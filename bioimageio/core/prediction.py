import os
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


def pad(im, axes, padding):
    is_volume = "z" in axes
    if is_volume:
        assert len(padding) == 3
    else:
        assert len(padding) == 2

    pad_width = []
    crop = {}
    for ax, dlen in zip(axes, im.shape):
        if ax in "zyx":
            pad_to = padding[ax]
            r = dlen % pad_to
            pwidth = 0 if r == 0 else (pad_to - r)
            pad_width.append([0, pwidth])
            crop[ax] = slice(0, dlen)
        else:
            pad_width.append([0, 0])
            crop[ax] = slice(None)

    im = np.pad(im, pad_width)
    return im, crop


def load_image(in_path, axes):
    ext = os.path.splitext(in_path)
    if ext == ".npy":
        im = np.load(in_path)
    else:
        is_volume = "z" in axes
        im = imageio.volread(in_path) if is_volume else imageio.imread(in_path)
        im = require_axes(im, axes)
    return im


def save_image(out_path, image, axes):
    ext = os.path.splitext(out_path)[1]
    if ext == ".npy":
        np.save(out_path, image)
    else:
        is_volume = "z" in axes

        # to channel last
        chan_id = axes.index("c")
        if chan_id != len(axes) - 1:
            target_axes = tuple(ax for ax in axes if ax != "c") + ("c",)
            axes_permutation = tuple(axes.index(ax) for ax in target_axes)
            image = image.transpose(axes_permutation)
            axes = target_axes

        # squeeze singleton axes
        squeeze = []
        for ax, dlen in zip(axes, image.shape):
            # squeeze batch or channle axes if they are singletons
            if ax in "bc" and dlen == 1:
                squeeze.append(0)
            else:
                squeeze.append(slice(None))
        image = image[tuple(squeeze)]

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


#
# prediction functions
#


def predict(prediction_pipeline, inputs):
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]
        return_array = True
    else:
        return_array = False
    if len(inputs) > 1:
        raise NotImplementedError(len(inputs))
    axes = tuple(prediction_pipeline.input_axes)
    tagged_data = [xr.DataArray(ipt, dims=axes) for ipt in inputs]
    result = prediction_pipeline.forward(*tagged_data)
    if return_array:
        return result[0]
    return result


def pad_predict_crop(prediction_pipeline, inputs, padding):
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]
        return_array = True
    else:
        return_array = False
    axes = tuple(prediction_pipeline.input_axes)
    inputs = [pad(inp, axes, padding) for inp in inputs]
    inputs, crops = [inp[0] for inp in inputs], [inp[1] for inp in inputs]
    result = predict(prediction_pipeline, inputs)
    result = [apply_crop(res, crop) for res, crop in zip(result, crops)]
    if return_array:
        return result[0]
    return result


# TODO add support for tiling
def predict_image(model_rdf, inputs, outputs, padding=None, devices=None):
    """Run prediction for a single set of inputs with a bioimage.io model.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    if len(inputs) > 1:
        raise NotImplementedError(len(inputs))

    model = load_resource_description(Path(model_rdf))
    assert isinstance(model, Model)
    prediction_pipeline = create_prediction_pipeline(bioimageio_model=model, devices=devices)

    axes = tuple(prediction_pipeline.input_axes)
    input_data = [load_image(inp, axes) for inp in inputs]

    if padding is None:
        res = predict(prediction_pipeline, input_data)
    else:
        res = pad_predict_crop(prediction_pipeline, input_data, padding)
    save_image(res, outputs[0], axes)


# TODO add support for tiling
def predict_images(model_rdf, inputs, outputs, verbose=False, padding=None, devices=None):
    """Predict multiple inputs with a bioimage.io model.

    Only works for models with a single input and output tensor.
    """
    model = load_resource_description(Path(model_rdf))
    assert isinstance(model, Model)
    prediction_pipeline = create_prediction_pipeline(bioimageio_model=model, devices=devices)
    axes = tuple(prediction_pipeline.input_axes)

    prog = zip(inputs, outputs)
    if verbose:
        prog = tqdm(prog, total=len(inputs))

    for inp, outp in prog:
        inp = load_image(inp, axes)
        if padding is None:
            res = predict(prediction_pipeline, inp)
        else:
            res = pad_predict_crop(prediction_pipeline, inp, padding)
        save_image(res, outp, axes)
