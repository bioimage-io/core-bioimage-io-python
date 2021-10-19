import collections
import os
import warnings
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Sequence, Tuple, Union

import imageio
import numpy as np
import xarray as xr
from tqdm import tqdm

from bioimageio.core import load_resource_description
from bioimageio.core.prediction_pipeline import PredictionPipeline, create_prediction_pipeline

#
# utility functions for prediction
#
from bioimageio.core.resource_io.nodes import (
    ImplicitOutputShape,
    InputTensor,
    Model,
    OutputTensor,
    ResourceDescription,
    URI,
)
from bioimageio.spec.model.raw_nodes import WeightsFormat
from bioimageio.spec.shared.raw_nodes import ResourceDescription as RawResourceDescription


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


def pad(im, axes: Sequence[str], padding, pad_right=True) -> Tuple[np.ndarray, Dict[str, slice]]:
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
        else:
            pad_width.append([0, 0])
            crop[ax] = slice(None)

    im = np.pad(im, pad_width, mode="symmetric")
    return im, crop


def load_image(in_path, axes: Sequence[str]) -> xr.DataArray:
    ext = os.path.splitext(in_path)[1]
    if ext == ".npy":
        im = np.load(in_path)
    else:
        is_volume = "z" in axes
        im = imageio.volread(in_path) if is_volume else imageio.imread(in_path)
        im = require_axes(im, axes)
    return xr.DataArray(im, dims=axes)


def load_tensors(sources, tensor_specs: List[Union[InputTensor, OutputTensor]]) -> List[xr.DataArray]:
    return [load_image(s, sspec.axes) for s, sspec in zip(sources, tensor_specs)]


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
            ax: slice(pos, min(pos + tsh, sh)) for ax, pos, tsh, sh in zip(spatial_axes, positions, tile_shape_, shape_)
        }
        inner_tile["b"] = slice(None)
        inner_tile["c"] = slice(None)

        local_tile = {
            ax: slice(
                inner_tile[ax].start - outer_tile[ax].start,
                -(outer_tile[ax].stop - inner_tile[ax].stop) if outer_tile[ax].stop != inner_tile[ax].stop else None,
            )
            for ax in spatial_axes
        }
        local_tile["b"] = slice(None)
        local_tile["c"] = slice(None)

        yield outer_tile, inner_tile, local_tile


def predict_with_tiling_impl(
    prediction_pipeline,
    inputs: List[xr.DataArray],
    outputs: List[xr.DataArray],
    tile_shapes: List[dict],
    halos: List[dict],
):
    if len(inputs) > 1:
        raise NotImplementedError("Tiling with multiple inputs not implemented yet")

    if len(outputs) > 1:
        raise NotImplementedError("Tiling with multiple outputs not implemented yet")

    assert len(tile_shapes) == len(outputs)
    assert len(halos) == len(outputs)

    input_ = inputs[0]
    output = outputs[0]
    tile_shape = tile_shapes[0]
    halo = halos[0]

    tiles = get_tiling(shape=input_.shape, tile_shape=tile_shape, halo=halo, input_axes=input_.dims)

    assert all(isinstance(ax, str) for ax in input_.dims)
    input_axes: Tuple[str, ...] = input_.dims  # noqa

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
        assert len(out) == 1
        out = out[0]
        output[inner_tile] = out[local_tile]


#
# prediction functions
#


def predict(prediction_pipeline: PredictionPipeline, inputs) -> List[xr.DataArray]:
    if not isinstance(inputs, (tuple, list)):
        inputs = [inputs]

    assert len(inputs) == len(prediction_pipeline.input_specs)
    tagged_data = [
        xr.DataArray(ipt, dims=ipt_spec.axes) for ipt, ipt_spec in zip(inputs, prediction_pipeline.input_specs)
    ]
    return prediction_pipeline.forward(*tagged_data)


def predict_with_padding(prediction_pipeline, inputs, padding, pad_right=True) -> List[xr.DataArray]:
    if not isinstance(inputs, (tuple, list)):
        inputs = [inputs]

    assert len(inputs) == len(prediction_pipeline.input_specs)

    if not isinstance(padding, (tuple, list)):
        padding = [padding]

    assert len(padding) == len(prediction_pipeline.input_specs)
    inputs, crops = zip(
        *[
            pad(inp, spec.axes, p, pad_right=pad_right)
            for inp, spec, p in zip(inputs, prediction_pipeline.input_specs, padding)
        ]
    )

    result = predict(prediction_pipeline, inputs)
    return [apply_crop(res, crop) for res, crop in zip(result, crops)]


def predict_with_tiling(prediction_pipeline: PredictionPipeline, inputs, tiling) -> List[xr.DataArray]:
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    assert len(inputs) == len(prediction_pipeline.input_specs)
    named_inputs: OrderedDict[str, xr.DataArray] = collections.OrderedDict(
        **{
            ipt_spec.name: xr.DataArray(ipt_data, dims=tuple(ipt_spec.axes))
            for ipt_data, ipt_spec in zip(inputs, prediction_pipeline.input_specs)
        }
    )

    outputs = []
    for output_spec in prediction_pipeline.output_specs:
        if isinstance(output_spec.shape, ImplicitOutputShape):
            scale = dict(zip(output_spec.axes, output_spec.shape.scale))
            offset = dict(zip(output_spec.axes, output_spec.shape.offset))

            # for now, we only support tiling if the spatial shape doesn't change
            # supporting this should not be so difficult, we would just need to apply the inverse
            # to "out_shape = scale * in_shape + 2 * offset" ("in_shape = (out_shape - 2 * offset) / scale")
            # to 'outer_tile' in 'get_tiling'
            if any(sc != 1 for ax, sc in scale.items() if ax in "xyz") or any(
                off != 0 for ax, off in offset.items() if ax in "xyz"
            ):
                raise NotImplementedError("Tiling with a different output shape is not yet supported")

            ref_input = named_inputs[output_spec.shape.reference_tensor]
            ref_input_shape = dict(zip(ref_input.dims, ref_input.shape))
            output_shape = tuple(int(scale[ax] * ref_input_shape[ax] + 2 * offset[ax]) for ax in output_spec.axes)
        else:
            output_shape = tuple(output_spec.shape)

        outputs.append(xr.DataArray(np.zeros(output_shape, dtype=output_spec.data_type), dims=tuple(output_spec.axes)))

    predict_with_tiling_impl(
        prediction_pipeline,
        list(named_inputs.values()),
        outputs,
        tile_shapes=[tiling["tile"]],  # todo: update tiling for multiple inputs/outputs
        halos=[tiling["halo"]],
    )

    return outputs


def parse_padding(padding, model):
    if len(model.inputs) > 1:
        raise NotImplementedError("Padding for multiple inputs not yet implemented")

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
    if len(model.inputs) > 1:
        raise NotImplementedError("Tiling for multiple inputs not yet implemented")

    if len(model.outputs) > 1:
        raise NotImplementedError("Tiling for multiple outputs not yet implemented")

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


def predict_image(model_rdf, inputs, outputs, padding=None, tiling=None, weight_format=None, devices=None):
    """Run prediction for a single set of inputs with a bioimage.io model."""
    if not isinstance(inputs, (tuple, list)):
        inputs = [inputs]

    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]

    model = load_resource_description(model_rdf)
    assert isinstance(model, Model)
    if len(model.inputs) != len(inputs):
        raise ValueError
    if len(model.outputs) != len(outputs):
        raise ValueError

    prediction_pipeline = create_prediction_pipeline(
        bioimageio_model=model, weight_format=weight_format, devices=devices
    )

    padding = parse_padding(padding, model)
    tiling = parse_tiling(tiling, model)

    _predict_sample(prediction_pipeline, inputs, outputs, padding, tiling)


def _predict_sample(prediction_pipeline, inputs, outputs, padding, tiling):
    if padding is not None and tiling is not None:
        raise ValueError("Only one of padding or tiling is supported")

    input_data = load_tensors(inputs, prediction_pipeline.input_specs)
    if padding is not None:
        result = predict_with_padding(prediction_pipeline, input_data, padding)
    elif tiling is not None:
        result = predict_with_tiling(prediction_pipeline, input_data, tiling)
    else:
        result = predict(prediction_pipeline, input_data)

    assert isinstance(result, list)
    assert len(result) == len(outputs)
    for res, out in zip(result, outputs):
        save_image(out, res)


def predict_images(
    model_rdf,
    inputs: Sequence[Union[Tuple[Path, ...], List[Path], Path]],
    outputs: Sequence[Union[Tuple[Path, ...], List[Path], Path]],
    padding=None,
    tiling=None,
    weight_format=None,
    devices=None,
    verbose=False,
):
    """Predict multiple inputs with a bioimage.io model."""

    model = load_resource_description(model_rdf)
    assert isinstance(model, Model)

    prediction_pipeline = create_prediction_pipeline(
        bioimageio_model=model, weight_format=weight_format, devices=devices
    )

    padding = parse_padding(padding, model)
    tiling = parse_tiling(tiling, model)

    prog = zip(inputs, outputs)
    if verbose:
        prog = tqdm(prog, total=len(inputs))

    for inp, outp in prog:
        if not isinstance(inp, (tuple, list)):
            inp = [inp]

        if not isinstance(outp, (tuple, list)):
            outp = [outp]

        _predict_sample(prediction_pipeline, inp, outp, padding, tiling)


def test_model(
    model_rdf: Union[URI, Path, str],
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[List[str]] = None,
    decimal: int = 4,
) -> bool:
    """Test whether the test output(s) of a model can be reproduced.

    Returns True if the test passes, otherwise returns False and issues a warning.
    """
    model = load_resource_description(model_rdf)
    assert isinstance(model, Model)
    summary = test_resource(model, weight_format=weight_format, devices=devices, decimal=decimal)
    if summary["error"] is None:
        return True
    else:
        warnings.warn(summary["error"])
        return False


def test_resource(
    model_rdf: Union[RawResourceDescription, ResourceDescription, URI, Path, str],
    *,
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[List[str]] = None,
    decimal: int = 4,
):
    """Test RDF dynamically

    Returns summary with "error", keys
    """
    model = load_resource_description(model_rdf)

    error: Optional[str] = None
    if isinstance(model, Model):
        prediction_pipeline = create_prediction_pipeline(
            bioimageio_model=model, devices=devices, weight_format=weight_format
        )
        inputs = [np.load(str(in_path)) for in_path in model.test_inputs]
        results = predict(prediction_pipeline, inputs)
        if isinstance(results, (np.ndarray, xr.DataArray)):
            results = [results]

        expected = [np.load(str(out_path)) for out_path in model.test_outputs]
        if len(results) != len(expected):
            error = f"Number of outputs and number of expected outputs disagree: {len(results)} != {len(expected)}"
        else:
            for res, exp in zip(results, expected):
                try:
                    np.testing.assert_array_almost_equal(res, exp, decimal=decimal)
                except AssertionError as e:
                    error = f"Output and expected output disagree:\n {e}"

    # todo: add tests for non-model resources

    return {"error": error}
