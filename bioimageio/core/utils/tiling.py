import warnings
from typing import Dict, Literal, Mapping, Tuple, Union

from typing_extensions import assert_never

from bioimageio.core.common import Tensor
from bioimageio.spec.model.v0_5 import AxisId


def pad(
    tensor: Tensor,
    pad_width: Mapping[AxisId, Union[int, Tuple[int, int]]],
    mode: Literal["edge", "reflect", "symmetric"] = "symmetric",
):
    return tensor.pad(pad_width={str(k): v for k, v in pad_width.items()}, mode=mode)


def pad_to(
    tensor: Tensor,
    sizes: Mapping[AxisId, int],
    pad_where: Union[
        Literal["before", "center", "after"],
        Mapping[AxisId, Literal["before", "center", "after"]],
    ] = "center",
    mode: Literal["edge", "reflect", "symmetric"] = "symmetric",
):
    """pad `tensor` to match `sizes`"""
    axes = [AxisId(str(a)) for a in tensor.dims]
    if pad_where in ("before", "center", "after"):
        pad_axis_where: Mapping[AxisId, Literal["before", "center", "after"]] = {
            a: pad_where for a in axes
        }
    else:
        pad_axis_where = pad_where

    pad_width: Dict[AxisId, Union[int, Tuple[int, int]]] = {}
    for a, s_is in tensor.sizes.items():
        a = AxisId(str(a))
        if a not in sizes or sizes[a] == s_is:
            pad_width[a] = 0
        elif s_is > sizes[a]:
            pad_width[a] = 0
            warnings.warn(
                f"Cannot pad axis {a} of size {s_is} to smaller size {sizes[a]}"
            )
        elif a not in pad_axis_where:
            raise ValueError(
                f"Don't know where to pad axis {a}, `pad_where`={pad_where}"
            )
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


def crop_to(
    tensor: Tensor,
    sizes: Mapping[AxisId, int],
    crop_where: Union[
        Literal["before", "center", "after"],
        Mapping[AxisId, Literal["before", "center", "after"]],
    ] = "center",
):
    """crop `tensor` to match `sizes`"""
    axes = [AxisId(str(a)) for a in tensor.dims]
    if crop_where in ("before", "center", "after"):
        crop_axis_where: Mapping[AxisId, Literal["before", "center", "after"]] = {
            a: crop_where for a in axes
        }
    else:
        crop_axis_where = crop_where

    slices: Dict[AxisId, slice] = {}

    for a, s_is in tensor.sizes.items():
        a = AxisId(str(a))
        if a not in sizes or sizes[a] == s_is:
            pass
        elif sizes[a] > s_is:
            warnings.warn(
                f"Cannot crop axis {a} of size {s_is} to larger size {sizes[a]}"
            )
        elif a not in crop_axis_where:
            raise ValueError(
                f"Don't know where to crop axis {a}, `crop_where`={crop_where}"
            )
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


def resize_to(
    tensor: Tensor,
    sizes: Mapping[AxisId, int],
    *,
    pad_where: Union[
        Literal["before", "center", "after"],
        Mapping[AxisId, Literal["before", "center", "after"]],
    ] = "center",
    crop_where: Union[
        Literal["before", "center", "after"],
        Mapping[AxisId, Literal["before", "center", "after"]],
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
        elif s_is > sizes[a]:
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
