import os
from copy import deepcopy
from itertools import product
from pathlib import Path

import imageio
import numpy as np
import xarray as xr

from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.spec import load_resource_description
from bioimageio.spec.model.nodes import Model
from tqdm import tqdm


#
# utility functions for prediction
#


def require_axes(im, axes):
    is_volume = "z" in axes
    # we assume images / volumes are loaded as one of
    # yx, yxc, zyxc
    if im.ndim == 2:
        im_axes = ("y", "x")
    elif im.ndim == 3:
        im_axes = ("z", "y", "x") if is_volume else ("y", "x", "c")
    elif im.ndim == 4:
        raise NotImplementedError
    else:  # ndim >= 5 not implemented
        raise RuntimeError

    # add singleton channel dimension if not present
    if "c" not in im_axes:
        im = im[..., None]
        im_axes = im_axes + ("c",)

    # add singleton batch dim
    im = im[None]
    im_axes = ("b",) + im_axes

    # permute the axes correctly
    assert set(axes) == set(im_axes)
    axes_permutation = tuple(im_axes.index(ax) for ax in axes)
    im = im.transpose(axes_permutation)
    return im


def pad(im, axes, padding, pad_right=True):
    assert im.ndim == len(axes), f"{im.ndim}, {len(axes)}"

    padding_ = deepcopy(padding)
    mode = padding_.pop("mode", "dynamic")
    assert mode in ("dynamic", "fixed")

    is_volume = "z" in axes
    if is_volume:
        assert len(padding_) == 3
    else:
        assert len(padding_) == 2

    if isinstance(pad_right, bool):
        pad_right = len(axes) * [pad_right]

    pad_width = []
    crop = {}
    for ax, dlen, pr in zip(axes, im.shape, pad_right):

        if ax in "zyx":
            pad_to = padding_[ax]

            if mode == "dynamic":
                r = dlen % pad_to
                pwidth = 0 if r == 0 else (pad_to - r)
            else:
                if pad_to < dlen:
                    msg = f"Padding for axis {ax} failed; pad shape {pad_to} is smaller than the image shape {dlen}."
                    raise RuntimeError(msg)
                pwidth = pad_to - dlen

            pad_width.append([0, pwidth] if pr else [pwidth, 0])
            crop[ax] = slice(0, dlen) if pr else slice(pwidth, None)

        elif ax in "zyx" and mode == "fixed":
            pad_to = padding_[ax]

        else:
            pad_width.append([0, 0])
            crop[ax] = slice(None)

    im = np.pad(im, pad_width, mode="symmetric")
    return im, crop


def load_image(in_path, axes):
    ext = os.path.splitext(in_path)[1]
    if ext == ".npy":
        im = np.load(in_path)
    else:
        is_volume = "z" in axes
        im = imageio.volread(in_path) if is_volume else imageio.imread(in_path)
        im = require_axes(im, axes)
    return im


def _to_channel_last(image):
    chan_id = image.dims.index("c")
    if chan_id != image.ndim - 1:
        target_axes = tuple(ax for ax in image.dims if ax != "c") + ("c",)
        image = image.transpose(*target_axes)
    return image


def save_image(out_path, image):
    ext = os.path.splitext(out_path)[1]
    if ext == ".npy":
        np.save(out_path, image)
    else:
        is_volume = "z" in image.dims

        # squeeze batch or channel axes if they are singletons
        squeeze = {ax: 0 if (ax in "bc" and sh == 1) else slice(None) for ax, sh in zip(image.dims, image.shape)}
        image = image[squeeze]

        if "b" in image.dims:
            raise RuntimeError(f"Cannot save prediction with batchsize > 1 as {ext}-file")
        if "c" in image.dims:  # image formats need channel last
            image = _to_channel_last(image)

        save_function = imageio.volsave if is_volume else imageio.imsave
        # most image formats only support channel dimensions of 1, 3 or 4;
        # if not we need to save the channels separately
        ndim = 3 if is_volume else 2
        save_as_single_image = image.ndim == ndim or (image.shape[-1] in (3, 4))

        if save_as_single_image:
            save_function(out_path, image)
        else:
            out_prefix, ext = os.path.splitext(out_path)
            for c in range(image.shape[-1]):
                chan_out_path = f"{out_prefix}-c{c}{ext}"
                save_function(chan_out_path, image[..., c])


def apply_crop(data, crop):
    crop = tuple(crop[ax] for ax in data.dims)
    return data[crop]


def get_tiling(shape, tile_shape, halo, input_axes):
    assert len(shape) == len(input_axes)

    shape_ = [sh for sh, ax in zip(shape, input_axes) if ax in "xyz"]
    spatial_axes = [ax for ax in input_axes if ax in "xyz"]
    tile_shape_ = [tile_shape[ax] for ax in spatial_axes]
    halo_ = [halo[ax] for ax in spatial_axes]
    assert len(shape_) == len(tile_shape_) == len(spatial_axes) == len(halo_)

    ranges = [range(sh // tsh if sh % tsh == 0 else sh // tsh + 1) for sh, tsh in zip(shape_, tile_shape_)]
    start_points = product(*ranges)

    for start_point in start_points:
        positions = [sp * tsh for sp, tsh in zip(start_point, tile_shape_)]

        outer_tile = {
            ax: slice(max(pos - ha, 0), min(pos + tsh + ha, sh))
            for ax, pos, tsh, sh, ha in zip(spatial_axes, positions, tile_shape_, shape_, halo_)
        }
        outer_tile["b"] = slice(None)
        outer_tile["c"] = slice(None)

        inner_tile = {
            ax: slice(pos, min(pos + tsh, sh))
            for ax, pos, tsh, sh in zip(spatial_axes, positions, tile_shape_, shape_)
        }
        inner_tile["b"] = slice(None)
        inner_tile["c"] = slice(None)

        local_tile = {
            ax: slice(
                inner_tile[ax].start - outer_tile[ax].start,
                -(outer_tile[ax].stop - inner_tile[ax].stop) if outer_tile[ax].stop != inner_tile[ax].stop else None
            )
            for ax in spatial_axes
        }
        local_tile["b"] = slice(None)
        local_tile["c"] = slice(None)

        yield outer_tile, inner_tile, local_tile


def predict_with_tiling_impl(prediction_pipeline, input_, output, tile_shape, halo, input_axes):
    assert input_.ndim == len(input_axes), f"{input_.ndim}, {len(input_axes)}"

    input_ = xr.DataArray(input_, dims=input_axes)
    tiles = get_tiling(input_.shape, tile_shape, halo, input_axes)

    def load_tile(tile):
        inp = input_[tile]
        # whether to pad on the right or left of the dim for the spatial dims
        # + placeholders for batch and axis dimension, where we don't pad
        pad_right = [None, None] + [tile[ax].start == 0 for ax in input_axes if ax in "xyz"]
        return inp, pad_right

    # we need to use padded prediction for the individual tiles in case the
    # border tiles don't match the requested tile shape
    padding = {ax: tile_shape[ax] + 2 * halo[ax] for ax in input_axes if ax in "xyz"}
    padding["mode"] = "fixed"
    for outer_tile, inner_tile, local_tile in tiles:
        inp, pad_right = load_tile(outer_tile)
        out = predict_with_padding(prediction_pipeline, inp, padding, pad_right)
        output[inner_tile] = out[local_tile]


#
# prediction functions
# TODO support models with multiple in/outputs
#


def predict(prediction_pipeline, inputs):
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]
    if len(inputs) > 1:
        raise NotImplementedError(len(inputs))
    axes = tuple(prediction_pipeline.input_axes)
    tagged_data = [xr.DataArray(ipt, dims=axes) for ipt in inputs]
    return prediction_pipeline.forward(*tagged_data)


def predict_with_padding(prediction_pipeline, inputs, padding, pad_right=True):
    if isinstance(inputs, (np.ndarray, xr.DataArray)):
        inputs = [inputs]
    axes = tuple(prediction_pipeline.input_axes)
    inputs = [pad(inp, axes, padding, pad_right=pad_right) for inp in inputs]
    inputs, crops = [inp[0] for inp in inputs], [inp[1] for inp in inputs]
    result = predict(prediction_pipeline, inputs)
    if isinstance(result, (list, tuple)):
        result = [apply_crop(res, crop) for res, crop in zip(result, crops)]
    else:
        result = apply_crop(result, crops[0])
    return result


def predict_with_tiling(prediction_pipeline, inputs, tiling):
    if isinstance(inputs, (list, tuple)):
        if len(inputs) > 1:
            raise NotImplementedError(len(inputs))
        input_ = inputs[0]
    else:
        input_ = inputs
    input_axes = tuple(prediction_pipeline.input_axes)

    output_axes = tuple(prediction_pipeline.output_axes)
    # NOTE there could also be models with a fixed output shape, but this is currently
    # not reflected in prediction_pipeline, need to adapt this here once fixed
    scale, offset = prediction_pipeline.scale, prediction_pipeline.offset
    scale, offset = {sc[0]: sc[1] for sc in scale}, {off[0]: off[1] for off in offset}

    # for now, we only support tiling if the spatial shape doesn't change
    # supporting this should not be so difficult, we would just need to apply the inverse
    # to "out_shape = scale * in_shape + 2 * offset" ("in_shape = (out_shape - 2 * offset) / scale")
    # to 'outer_tile' in 'get_tiling'
    if any(scale[ax] != 1 for ax in output_axes if ax in "xyz") or\
       any(offset[ax] != 0 for ax in output_axes if ax in "xyz"):
        raise NotImplementedError("Tiling with a different output shape is not yet supported")

    out_shape = tuple(
        int(scale[ax] * input_.shape[input_axes.index(ax)] + 2 * offset[ax]) for ax in output_axes
    )
    # TODO the dtype information is missing from prediction pipeline
    out_dtype = "float32"
    output = xr.DataArray(np.zeros(out_shape, dtype=out_dtype), dims=output_axes)

    halo = tiling["halo"]
    tile_shape = tiling["tile"]

    predict_with_tiling_impl(prediction_pipeline, input_, output, tile_shape, halo, input_axes)
    return output


def parse_padding(padding, model):

    input_spec = model.inputs[0]
    pad_keys = tuple(input_spec.axes) + ("mode",)

    def check_padding(padding):
        assert all(k in pad_keys for k in padding.keys())

    if padding is None:  # no padding
        return padding
    elif isinstance(padding, dict):  # pre-defined padding
        check_padding(padding)
    elif isinstance(padding, bool):  # determine padding from spec
        if padding:
            axes = input_spec.axes
            shape = input_spec.shape
            if isinstance(shape, list):  # fixed padding
                padding = {ax: sh for ax, sh in zip(axes, shape) if ax in "xyz"}
                padding["mode"] = "fixed"
            else:  # dynamic padding
                step = shape.step
                padding = {ax: st for ax, st in zip(axes, step) if ax in "xyz"}
                padding["mode"] = "dynamic"
            check_padding(padding)
        else:  # no padding
            padding = None
    else:
        raise ValueError(f"Invalid argument for padding: {padding}")
    return padding


def parse_tiling(tiling, model):
    input_spec = model.inputs[0]
    output_spec = model.outputs[0]

    def check_tiling(tiling):
        assert "halo" in tiling and "tile" in tiling

    if tiling is None:  # no tiling
        return tiling
    elif isinstance(tiling, dict):
        check_tiling(tiling)
    elif isinstance(tiling, bool):
        if tiling:
            # NOTE we assume here that shape in input and output are the same
            # for different input and output shapes, we should actually tile in the
            # output space and then request the corresponding input tiles
            # so we would need to apply the output scale and offset to the
            # input shape to compute the tile size and halo here
            axes = input_spec.axes
            shape = input_spec.shape
            if not isinstance(shape, list):
                # NOTE this might result in very small tiles.
                # it would be good to have some heuristic to determine a suitable tilesize
                # from shape.min and shape.step
                shape = shape.min
            halo = output_spec.halo
            tiling = {
                "halo": {ax: ha for ax, ha in zip(axes, halo) if ax in "xyz"},
                "tile": {ax: sh for ax, sh in zip(axes, shape) if ax in "xyz"},
            }
        else:
            tiling = None
    else:
        raise ValueError(f"Invalid argument for tiling: {tiling}")
    return tiling


# TODO support models with multiple in/outputs
def predict_image(model_rdf, inputs, outputs, padding=None, tiling=None, devices=None):
    """Run prediction for a single set of inputs with a bioimage.io model."""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    if len(inputs) > 1:
        raise NotImplementedError(len(inputs))
    if isinstance(outputs, (str, Path)):
        outputs = [outputs]
    if len(outputs) > 1:
        raise NotImplementedError(len(outputs))

    model = load_resource_description(Path(model_rdf))
    assert isinstance(model, Model)
    if len(model.inputs) != len(inputs):
        raise ValueError
    if len(model.outputs) != len(outputs):
        raise ValueError

    prediction_pipeline = create_prediction_pipeline(bioimageio_model=model, devices=devices)
    axes = tuple(prediction_pipeline.input_axes)

    padding = parse_padding(padding, model)
    tiling = parse_tiling(tiling, model)
    if padding is not None and tiling is not None:
        raise ValueError("Only one of padding or tiling is supported")

    input_data = [load_image(inp, axes) for inp in inputs]
    if padding is not None:
        result = predict_with_padding(prediction_pipeline, input_data, padding)
    elif tiling is not None:
        result = predict_with_tiling(prediction_pipeline, input_data, tiling)
    else:
        result = predict(prediction_pipeline, input_data)

    if isinstance(result, list):
        assert len(result) == len(outputs)
        for res, out in zip(result, outputs):
            save_image(out, res)
    else:
        assert len(outputs) == 1
        save_image(outputs[0], result)


def predict_images(model_rdf, inputs, outputs, verbose=False, padding=None, tiling=None, devices=None):
    """Predict multiple inputs with a bioimage.io model.

    Only works for models with a single input and output tensor.
    """
    model = load_resource_description(Path(model_rdf))
    assert isinstance(model, Model)
    if len(model.inputs) > 1 or len(model.outputs) > 1:
        raise RuntimeError("predict_images only supports models that have a single input/output tensor")

    prediction_pipeline = create_prediction_pipeline(bioimageio_model=model, devices=devices)
    axes = tuple(prediction_pipeline.input_axes)

    padding = parse_padding(padding, model)
    tiling = parse_tiling(tiling, model)
    if padding is not None and tiling is not None:
        raise ValueError("Only one of padding or tiling is supported")

    prog = zip(inputs, outputs)
    if verbose:
        prog = tqdm(prog, total=len(inputs))

    for inp, outp in prog:
        inp = load_image(inp, axes)
        if padding is not None:
            res = predict_with_padding(prediction_pipeline, inp, padding)
        elif tiling is not None:
            res = predict_with_tiling(prediction_pipeline, inp, tiling)
        else:
            res = predict(prediction_pipeline, inp)
        save_image(outp, res)
