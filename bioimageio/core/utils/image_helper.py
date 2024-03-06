# TODO: update

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import imageio
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import assert_never

from bioimageio.spec.model import v0_4
from bioimageio.spec.model.v0_4 import InputTensorDescr as InputTensorDescr04
from bioimageio.spec.model.v0_4 import OutputTensorDescr as OutputTensorDescr04
from bioimageio.spec.model.v0_5 import (
    AnyAxis,
    AxisId,
    BatchAxis,
    ChannelAxis,
    Identifier,
    InputAxis,
    InputTensorDescr,
    OutputTensorDescr,
    SpaceInputAxis,
    convert_axes,
)
from bioimageio.spec.utils import load_array, save_array

InputTensor = Union[InputTensorDescr04, InputTensorDescr]
OutputTensor = Union[OutputTensorDescr04, OutputTensorDescr]


def transpose_image(
    image: NDArray[Any],
    desired_axes: Union[v0_4.AxesStr, Sequence[AnyAxis]],
    current_axes: Optional[Union[v0_4.AxesStr, Sequence[AnyAxis]]] = None,
) -> xr.DataArray:
    """Transpose an image to match desired axes.

    Args:
        image: the input image
        desired_axes: the desired image axes
        current_axes: the axes of the input image
    """
    # if the image axes are not given deduce them from the required axes and image shape
    if current_axes is None:
        if isinstance(desired_axes, str):
            desired_space_axes = [a for a in desired_axes if a in "zyx"]
        else:
            desired_space_axes = [a for a in desired_axes if a.type == "space"]

        ndim = image.ndim
        if ndim == 2 and len(desired_space_axes) >= 2:
            current_axes = (
                SpaceInputAxis(id=AxisId("y"), size=image.shape[0]),
                SpaceInputAxis(id=AxisId("x"), size=image.shape[1]),
            )
        elif ndim == 3 and len(desired_space_axes) == 2:
            current_axes = (
                ChannelAxis(channel_names=[Identifier(f"channel{i}") for i in range(image.shape[0])]),
                SpaceInputAxis(id=AxisId("y"), size=image.shape[1]),
                SpaceInputAxis(id=AxisId("x"), size=image.shape[2]),
            )
        elif ndim == 3 and len(desired_space_axes) == 3:
            current_axes = (
                SpaceInputAxis(id=AxisId("z"), size=image.shape[0]),
                SpaceInputAxis(id=AxisId("y"), size=image.shape[1]),
                SpaceInputAxis(id=AxisId("x"), size=image.shape[2]),
            )
        elif ndim == 4:
            current_axes = (
                ChannelAxis(channel_names=[Identifier(f"channel{i}") for i in range(image.shape[0])]),
                SpaceInputAxis(id=AxisId("z"), size=image.shape[1]),
                SpaceInputAxis(id=AxisId("y"), size=image.shape[2]),
                SpaceInputAxis(id=AxisId("x"), size=image.shape[3]),
            )
        elif ndim == 5:
            current_axes = (
                BatchAxis(),
                ChannelAxis(channel_names=[Identifier(f"channel{i}") for i in range(image.shape[1])]),
                SpaceInputAxis(id=AxisId("z"), size=image.shape[2]),
                SpaceInputAxis(id=AxisId("y"), size=image.shape[3]),
                SpaceInputAxis(id=AxisId("x"), size=image.shape[4]),
            )
        else:
            raise ValueError(f"Could not guess a mapping of {image.shape} to {desired_axes}")

    current_axes_ids = tuple(current_axes) if isinstance(current_axes, str) else tuple(a.id for a in current_axes)
    desired_axes_ids = tuple(desired_axes) if isinstance(desired_axes, str) else tuple(a.id for a in desired_axes)
    tensor = xr.DataArray(image, dims=current_axes_ids)
    # expand the missing image axes
    missing_axes = tuple(set(desired_axes_ids) - set(current_axes_ids))
    tensor = tensor.expand_dims(dim=missing_axes)
    # transpose to the correct axis order
    return tensor.transpose(*tuple(desired_axes_ids))


def convert_axes_for_known_shape(axes: v0_4.AxesStr, shape: Sequence[int]):
    return convert_axes(axes, shape=shape, tensor_type="input", halo=None, size_refs={})


def load_tensor(
    path: Path,
    desired_axes: Union[v0_4.AxesStr, Sequence[AnyAxis]],
    current_axes: Optional[Union[v0_4.AxesStr, Sequence[AnyAxis]]] = None,
) -> xr.DataArray:

    ext = path.suffix
    if ext == ".npy":
        im = load_array(path)
    else:
        guess_axes = current_axes or desired_axes
        if isinstance(guess_axes, str):
            is_volume = "z" in guess_axes or "t" in guess_axes
        else:
            is_volume = len([a for a in guess_axes if a.type in ("time", "space")]) > 2

        im = imageio.volread(path) if is_volume else imageio.imread(path)
        im = transpose_image(im, desired_axes=desired_axes, current_axes=current_axes)

    return xr.DataArray(
        im, dims=tuple(desired_axes) if isinstance(desired_axes, str) else tuple(a.id for a in desired_axes)
    )


def pad(
    tensor: xr.DataArray,
    pad_with: Mapping[AxisId, Union[int, Tuple[int, int]]],
    mode: Literal["edge", "reflect", "symmetric"] = "symmetric",
):
    return tensor.pad(pad_with=pad_with, mode=mode)


def pad_to(
    tensor: xr.DataArray,
    sizes: Mapping[AxisId, int],
    pad_where: Union[
        Literal["before", "center", "after"], Mapping[AxisId, Literal["before", "center", "after"]]
    ] = "center",
    mode: Literal["edge", "reflect", "symmetric"] = "symmetric",
):
    """pad `tensor` to match `shape`"""
    if isinstance(pad_where, str):
        pad_axis_where: Mapping[AxisId, Literal["before", "center", "after"]] = {
            AxisId(str(a)): pad_where for a in tensor.dims
        }
    else:
        pad_axis_where = pad_where

    pad_with: Dict[AxisId, Union[int, Tuple[int, int]]] = {}
    for a, s_is in tensor.sizes.items():
        a = AxisId(str(a))
        if a not in sizes or sizes[a] == s_is:
            pad_with[a] = 0
        elif s_is < sizes[a]:
            raise ValueError(f"Cannot pad axis {a} of size {s_is} to smaller size {sizes[a]}")
        elif a not in pad_axis_where:
            raise ValueError(f"Don't know where to pad axis {a}, `pad_where`={pad_where}")
        else:
            pad_this_axis_where = pad_axis_where[a]
            p = sizes[a] - s_is
            if pad_this_axis_where == "before":
                pad_with[a] = (p, 0)
            elif pad_this_axis_where == "after":
                pad_with[a] = (0, p)
            elif pad_this_axis_where == "center":
                pad_with[a] = (left := p // 2, p - left)
            else:
                assert_never(pad_this_axis_where)

    return pad(tensor, pad_with, mode)


def pad_old(image, axes: Sequence[str], padding, pad_right=True) -> Tuple[np.ndarray, Dict[str, slice]]:
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

    pad_width: Sequence[Tuple[int, int]] = []
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
