import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import imageio
import numpy as np
import xarray as xr
from tqdm import tqdm

from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.spec import load_resource_description
from bioimageio.spec.model.nodes import Model

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="bioimage model resource (ziped package or rdf.yaml)", required=True)
parser.add_argument("-i", "--input", help="input folder with images for prediction", required=True)
parser.add_argument("-o", "--output", help="output folder to save the predictions", required=True)
parser.add_argument("-e", "--output_extension", help="", default=None)
parser.add_argument("--devices", nargs="+", help="Devices to run this model", default=None)
parser.add_argument("--wildcard", default=None, help="glob wildcard to select files in the input folder")
# implement tiling and enable this as well???
# json-encoded dict
parser.add_argument("--padding", type=str, default=None)


def _require_axes(im, axes, is_volume):
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


def _pad(im, axes, padding, is_volume):
    if is_volume:
        assert len(padding) == 3
    else:
        assert len(padding) == 2

    pad_width = []
    crop = []
    for ax, dlen in zip(axes, im.shape):
        if ax in "zyx":
            pad_to = padding[ax]
            r = dlen % pad_to
            pwidth = 0 if r == 0 else (pad_to - r)
            pad_width.append([0, pwidth])
            crop.append(slice(0, dlen))
        else:
            pad_width.append([0, 0])
            crop.append(slice(None))

    im = np.pad(im, pad_width)
    return im, tuple(crop)


def _load_image(in_path, axes, padding):
    ext = os.path.splitext(in_path)
    is_volume = "z" in axes

    if ext == ".npy":
        im = np.load(in_path)
    else:
        im = imageio.volread(in_path) if is_volume else imageio.imread(in_path)
        im = _require_axes(im, axes, is_volume)

    if padding is None:
        crop = None
    else:
        im, crop = _pad(im, axes, padding, is_volume)

    return [xr.DataArray(im, dims=axes)], crop


def _save_image(out_path, image, axes, ext):
    if ext is None:
        ext = os.path.splitext(out_path)
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


def _predict(prediction_pipeline, in_path, out_path, model, padding, output_ext):
    axes = tuple(model.inputs[0].axes)
    input_, crop = _load_image(in_path, axes, padding)

    res = prediction_pipeline.forward(*input_)
    # NOTE this only makes sense if the model returns the same shape
    if crop is not None:
        res = res[crop]

    axes = tuple(model.outputs[0].axes)
    _save_image(out_path, res.data, axes, output_ext)


def main():
    args = parser.parse_args()
    model = load_resource_description(Path(args.model))
    assert isinstance(model, Model)
    if len(model.inputs) > 1 or len(model.outputs) > 1:
        raise ValueError("Bulk prediction is only supported for models with a single input and output")

    wildcard = "*" if args.wildcard is None else args.wildcard
    input_files = glob(os.path.join(args.input, wildcard))

    prediction_pipeline = create_prediction_pipeline(bioimageio_model=model, devices=args.devices)
    os.makedirs(args.output, exist_ok=True)

    padding = args.padding
    if padding is not None:
        padding = json.loads(padding.replace("'", '"'))
        assert isinstance(padding, dict)

    for input_ in tqdm(input_files):
        output = os.path.join(args.output, os.path.split(input_)[1])
        _predict(prediction_pipeline, input_, output, model, padding, args.output_extension)

    return 0


if __name__ == "__main__":
    sys.exit(main())
