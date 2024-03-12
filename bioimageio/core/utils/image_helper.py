import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import imageio
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import assert_never

from bioimageio.core.common import Axis
from bioimageio.spec.model import v0_4
from bioimageio.spec.model.v0_4 import InputTensorDescr as InputTensorDescr04
from bioimageio.spec.model.v0_4 import OutputTensorDescr as OutputTensorDescr04
from bioimageio.spec.model.v0_5 import (
    AnyAxis,
    AxisId,
    BatchAxis,
    ChannelAxis,
    Identifier,
    InputTensorDescr,
    OutputTensorDescr,
    SpaceInputAxis,
    convert_axes,
)
from bioimageio.spec.utils import load_array

InputTensor = Union[InputTensorDescr04, InputTensorDescr]
OutputTensor = Union[OutputTensorDescr04, OutputTensorDescr]


def interprete_array_with_desired_axes(
    nd_array: NDArray[Any],
    desired_axes: Union[v0_4.AxesStr, Sequence[AnyAxis]],
) -> xr.DataArray:
    if isinstance(desired_axes, str):
        desired_space_axes = [a for a in desired_axes if a in "zyx"]
    else:
        desired_space_axes = [a for a in desired_axes if a.type == "space"]

    return interprete_array(nd_array, len(desired_space_axes))


def interprete_array(
    nd_array: NDArray[Any],
    n_expected_space_axes: Optional[int] = None,
) -> xr.DataArray:

    ndim = nd_array.ndim
    if ndim == 2 and (n_expected_space_axes is None or n_expected_space_axes >= 2):
        current_axes = (
            SpaceInputAxis(id=AxisId("y"), size=nd_array.shape[0]),
            SpaceInputAxis(id=AxisId("x"), size=nd_array.shape[1]),
        )
    elif ndim == 3 and (
        (n_expected_space_axes is None and any(s <= 3 for s in nd_array.shape)) or n_expected_space_axes == 2
    ):
        current_axes = (
            ChannelAxis(channel_names=[Identifier(f"channel{i}") for i in range(nd_array.shape[0])]),
            SpaceInputAxis(id=AxisId("y"), size=nd_array.shape[1]),
            SpaceInputAxis(id=AxisId("x"), size=nd_array.shape[2]),
        )
    elif ndim == 3 and (n_expected_space_axes is None or n_expected_space_axes == 3):
        current_axes = (
            SpaceInputAxis(id=AxisId("z"), size=nd_array.shape[0]),
            SpaceInputAxis(id=AxisId("y"), size=nd_array.shape[1]),
            SpaceInputAxis(id=AxisId("x"), size=nd_array.shape[2]),
        )
    elif ndim == 4:
        current_axes = (
            ChannelAxis(channel_names=[Identifier(f"channel{i}") for i in range(nd_array.shape[0])]),
            SpaceInputAxis(id=AxisId("z"), size=nd_array.shape[1]),
            SpaceInputAxis(id=AxisId("y"), size=nd_array.shape[2]),
            SpaceInputAxis(id=AxisId("x"), size=nd_array.shape[3]),
        )
    elif ndim == 5:
        current_axes = (
            BatchAxis(),
            ChannelAxis(channel_names=[Identifier(f"channel{i}") for i in range(nd_array.shape[1])]),
            SpaceInputAxis(id=AxisId("z"), size=nd_array.shape[2]),
            SpaceInputAxis(id=AxisId("y"), size=nd_array.shape[3]),
            SpaceInputAxis(id=AxisId("x"), size=nd_array.shape[4]),
        )
    else:
        raise ValueError(
            f"Could not guess an axis mapping for {nd_array.shape} with {n_expected_space_axes} expected space axes"
        )

    current_axes_ids = tuple(current_axes) if isinstance(current_axes, str) else tuple(a.id for a in current_axes)
    return xr.DataArray(nd_array, dims=current_axes_ids)


def axis_descr_to_ids(axes: Union[v0_4.AxesStr, Sequence[AnyAxis]]) -> Tuple[AxisId, ...]:
    if isinstance(axes, str):
        return tuple(map(AxisId, axes))
    else:
        return tuple(a.id for a in axes)


def transpose_tensor(
    tensor: xr.DataArray,
    axes: Sequence[AxisId],
) -> xr.DataArray:
    """Transpose `array` to `axes` order.

    Args:
        tensor: the input array
        axes: the desired array axes
    """

    # expand the missing image axes
    current_axes = tuple(AxisId(str(d)) for d in tensor.dims)
    missing_axes = tuple(str(a) for a in axes if a not in current_axes)
    tensor = tensor.expand_dims(missing_axes)
    # transpose to the correct axis order
    return tensor.transpose(*axes)


def convert_v0_4_axes_for_known_shape(axes: v0_4.AxesStr, shape: Sequence[int]):
    return convert_axes(axes, shape=shape, tensor_type="input", halo=None, size_refs={})


def load_tensor(
    path: Path,
    axes: Optional[Sequence[Axis]] = None,
) -> xr.DataArray:

    ext = path.suffix
    if ext == ".npy":
        array = load_array(path)
    else:
        is_volume = True if axes is None else sum(a.type != "channel" for a in axes) > 2
        array = imageio.volread(path) if is_volume else imageio.imread(path)

    if axes is None:
        return interprete_array(array)
    else:
        return xr.DataArray(array, dims=tuple(a.id for a in axes))


def pad(
    tensor: xr.DataArray,
    pad_width: Mapping[AxisId, Union[int, Tuple[int, int]]],
    mode: Literal["edge", "reflect", "symmetric"] = "symmetric",
):
    return tensor.pad(pad_width={str(k): v for k, v in pad_width.items()}, mode=mode)


def resize_to(
    tensor: xr.DataArray,
    sizes: Mapping[AxisId, int],
    *,
    pad_where: Union[
        Literal["before", "center", "after"], Mapping[AxisId, Literal["before", "center", "after"]]
    ] = "center",
    crop_where: Union[
        Literal["before", "center", "after"], Mapping[AxisId, Literal["before", "center", "after"]]
    ] = "center",
    pad_mode: Literal["edge", "reflect", "symmetric"] = "symmetric",
):
    """crop and pad `tensor` to match `sizes`"""
    crop_to_sizes: Dict[AxisId, int] = {}
    pad_to_sizes: Dict[AxisId, int] = {}
    new_axes = dict(sizes)
    for a, s_is in tensor.sizes.items():
        a = AxisId(str(a))
        _ = new_axes.pop(a, None)
        if a not in sizes or sizes[a] == s_is:
            pass
        elif s_is < sizes[a]:
            crop_to_sizes[a] = sizes[a]
        else:
            pad_to_sizes[a] = sizes[a]

    if crop_to_sizes:
        tensor = crop_to(tensor, crop_to_sizes, crop_where=crop_where)

    if pad_to_sizes:
        tensor = pad_to(tensor, pad_to_sizes, pad_where=pad_where, mode=pad_mode)

    if new_axes:
        tensor = tensor.expand_dims({str(k): v for k, v in new_axes})

    return tensor


def crop_to(
    tensor: xr.DataArray,
    sizes: Mapping[AxisId, int],
    crop_where: Union[
        Literal["before", "center", "after"], Mapping[AxisId, Literal["before", "center", "after"]]
    ] = "center",
):
    """crop `tensor` to match `sizes`"""
    axes = [AxisId(str(a)) for a in tensor.dims]
    if crop_where in ("before", "center", "after"):
        crop_axis_where: Mapping[AxisId, Literal["before", "center", "after"]] = {a: crop_where for a in axes}
    else:
        crop_axis_where = crop_where

    slices: Dict[AxisId, slice] = {}

    for a, s_is in tensor.sizes.items():
        a = AxisId(str(a))
        if a not in sizes or sizes[a] == s_is:
            pass
        elif sizes[a] > s_is:
            warnings.warn(f"Cannot crop axis {a} of size {s_is} to larger size {sizes[a]}")
        elif a not in crop_axis_where:
            raise ValueError(f"Don't know where to crop axis {a}, `crop_where`={crop_where}")
        else:
            crop_this_axis_where = crop_axis_where[a]
            if crop_this_axis_where == "before":
                slices[a] = slice(s_is - sizes[a], s_is)
            elif crop_this_axis_where == "after":
                slices[a] = slice(0, sizes[a])
            elif crop_this_axis_where == "center":
                slices[a] = slice(start := (s_is - sizes[a]) // 2, sizes[a] + start)
            else:
                assert_never(crop_this_axis_where)

    return tensor.isel({str(a): s for a, s in slices.items()})


def pad_to(
    tensor: xr.DataArray,
    sizes: Mapping[AxisId, int],
    pad_where: Union[
        Literal["before", "center", "after"], Mapping[AxisId, Literal["before", "center", "after"]]
    ] = "center",
    mode: Literal["edge", "reflect", "symmetric"] = "symmetric",
):
    """pad `tensor` to match `sizes`"""
    axes = [AxisId(str(a)) for a in tensor.dims]
    if pad_where in ("before", "center", "after"):
        pad_axis_where: Mapping[AxisId, Literal["before", "center", "after"]] = {a: pad_where for a in axes}
    else:
        pad_axis_where = pad_where

    pad_width: Dict[AxisId, Union[int, Tuple[int, int]]] = {}
    for a, s_is in tensor.sizes.items():
        a = AxisId(str(a))
        if a not in sizes or sizes[a] == s_is:
            pad_width[a] = 0
        elif s_is < sizes[a]:
            pad_width[a] = 0
            warnings.warn(f"Cannot pad axis {a} of size {s_is} to smaller size {sizes[a]}")
        elif a not in pad_axis_where:
            raise ValueError(f"Don't know where to pad axis {a}, `pad_where`={pad_where}")
        else:
            pad_this_axis_where = pad_axis_where[a]
            p = sizes[a] - s_is
            if pad_this_axis_where == "before":
                pad_width[a] = (p, 0)
            elif pad_this_axis_where == "after":
                pad_width[a] = (0, p)
            elif pad_this_axis_where == "center":
                pad_width[a] = (left := p // 2, p - left)
            else:
                assert_never(pad_this_axis_where)

    return pad(tensor, pad_width, mode)
