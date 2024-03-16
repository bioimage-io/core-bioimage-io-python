import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import imageio
import numpy as np
from tifffile import tifffile
from xarray import DataArray
from bioimageio.core.resource_io.nodes import InputTensor, OutputTensor


#
# helper functions to transform input images / output tensors to the required axes
#


def transform_input_image(image: np.ndarray, tensor_axes: str, image_axes: Optional[str] = None):
    """Transform input image into output tensor with desired axes.

    Args:
        image: the input image
        tensor_axes: the desired tensor axes
        input_axes: the axes of the input image (optional)
    """
    # if the image axes are not given deduce them from the required axes and image shape
    if image_axes is None:
        has_z_axis = "z" in tensor_axes
        ndim = image.ndim
        if ndim == 2:
            image_axes = "yx"
        elif ndim == 3:
            image_axes = "zyx" if has_z_axis else "cyx"
        elif ndim == 4:
            image_axes = "czyx"
        elif ndim == 5:
            image_axes = "bczyx"
        else:
            raise ValueError(f"Invalid number of image dimensions: {ndim}")
    tensor = DataArray(image, dims=tuple(image_axes))
    # expand the missing image axes
    missing_axes = tuple(set(tensor_axes) - set(image_axes))
    tensor = tensor.expand_dims(dim=missing_axes)
    # transpose to the correct axis order
    tensor = tensor.transpose(*tuple(tensor_axes))
    # return numpy array
    return tensor.values


def _drop_axis_default(axis_name, axis_len):
    # spatial axes: drop at middle coordnate
    # other axes (channel or batch): drop at 0 coordinate
    return axis_len // 2 if axis_name in "zyx" else 0


def transform_output_tensor(tensor: np.ndarray, tensor_axes: str, output_axes: str, drop_function=_drop_axis_default):
    """Transform output tensor into image with desired axes.

    Args:
        tensor: the output tensor
        tensor_axes: bioimageio model spec
        output_axes: the desired output axes
        drop_function: function that determines how to drop unwanted axes
    """
    if len(tensor_axes) != tensor.ndim:
        raise ValueError(f"Number of axes {len(tensor_axes)} and dimension of tensor {tensor.ndim} don't match")
    shape = {ax_name: sh for ax_name, sh in zip(tensor_axes, tensor.shape)}
    output = DataArray(tensor, dims=tuple(tensor_axes))
    # drop unwanted axes
    drop_axis_names = tuple(set(tensor_axes) - set(output_axes))
    drop_axes = {ax_name: drop_function(ax_name, shape[ax_name]) for ax_name in drop_axis_names}
    output = output[drop_axes]
    # transpose to the desired axis order
    output = output.transpose(*tuple(output_axes))
    # return numpy array
    return output.values


def to_channel_last(image):
    chan_id = image.dims.index("c")
    if chan_id != image.ndim - 1:
        target_axes = tuple(ax for ax in image.dims if ax != "c") + ("c",)
        image = image.transpose(*target_axes)
    return image


#
# helper functions for loading and saving images
#


def load_image(in_path, axes: Optional[Sequence[str]] = None) -> DataArray:
    ext = os.path.splitext(in_path)[1]
    if ext == ".npy":
        im = np.load(in_path)
    else:
        is_volume = "z" in axes
        im = imageio.volread(in_path) if is_volume else imageio.imread(in_path)
        im = transform_input_image(im, axes)
    return DataArray(im, dims=axes)


def load_tensors(sources, tensor_specs: List[Union[InputTensor, OutputTensor]]) -> List[DataArray]:
    return [load_image(s, sspec.axes) for s, sspec in zip(sources, tensor_specs)]


def save_image(out_path: os.PathLike, image: DataArray, pixel_size=None):
    out_path = Path(out_path)
    if out_path.suffix == ".npy":
        if pixel_size is not None:
            warnings.warn("Ignoring 'pixel_size'")
        np.save(str(out_path), image)
    elif out_path.suffix in (".tif", ".tiff"):
        save_imagej_tiff_image(out_path, image)
    else:
        if pixel_size is not None:
            warnings.warn("Ignoring 'pixel_size'")
        is_volume = "z" in image.dims

        # squeeze batch or channel axes if they are singletons
        squeeze = {ax: 0 if (ax in "bc" and sh == 1) else slice(None) for ax, sh in zip(image.dims, image.shape)}
        image = image[squeeze]

        if "b" in image.dims:
            raise RuntimeError(f"Cannot save prediction with batchsize > 1 as {out_path.suffix}-file")
        if "c" in image.dims:  # image formats need channel last
            image = to_channel_last(image)

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


def save_imagej_tiff_image(path, image: DataArray, pixel_size: Optional[Dict[str, float]] = None):
    pixel_size = pixel_size or image.attrs.get("pixel_size")
    assert (
        pixel_size is None
        or isinstance(pixel_size, dict)
        and all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in pixel_size.items())
    )
    assert im.ndim in (4, 5), f"{im.ndim}"

    # convert the image to expected (Z)CYX axis order
    if im.ndim == 4:
        assert set(axes) == {"b", "x", "y", "c"}, f"{axes}"
        resolution_axes_ij = "cyxb"
    else:
        assert set(axes) == {"b", "x", "y", "z", "c"}, f"{axes}"
        resolution_axes_ij = "bzcyx"

    def add_missing_axes(im_axes):
        needed_axes = ["b", "c", "x", "y", "z", "s"]
        for ax in needed_axes:
            if ax not in im_axes:
                im_axes += ax
        return im_axes

    axes_ij = "bzcyxs"
    # Expand the image to ImageJ dimensions
    im = np.expand_dims(im, axis=tuple(range(len(axes), len(axes_ij))))

    axis_permutation = tuple(add_missing_axes(axes).index(ax) for ax in axes_ij)
    im = im.transpose(axis_permutation)

    tiff_metadata = {}
    if pixel_size is None:
        resolution = None
    else:
        spatial_axes = list(set(resolution_axes_ij) - set("bc"))
        resolution = tuple(1.0 / pixel_size[ax] for ax in resolution_axes_ij if ax in spatial_axes)
    # does not work for double
    if np.dtype(im.dtype) == np.dtype("float64"):
        im = im.astype("float32")
    tifffile.imwrite(path, im, imagej=True, resolution=resolution)


#
# helper function for padding
#


def pad(image, axes: Sequence[str], padding, pad_right=True) -> Tuple[np.ndarray, Dict[str, slice]]:
    assert image.ndim == len(axes), f"{image.ndim}, {len(axes)}"

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
    for ax, dlen, pr in zip(axes, image.shape, pad_right):
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

    image = np.pad(image, pad_width, mode="symmetric")
    return image, crop
