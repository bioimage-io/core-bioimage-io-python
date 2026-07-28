from __future__ import annotations

import collections.abc
from itertools import permutations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    get_args,
)

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from numpy.typing import DTypeLike, NDArray
from typing_extensions import Self, TypeAlias, assert_never

from bioimageio.spec._internal.type_guards import is_dict
from bioimageio.spec.model import v0_5

from ._magic_tensor_ops import MagicTensorOpsMixin
from .axis import AxisId, AxisInfo, AxisLike, PerAxis
from .common import (
    CropWhere,
    DTypeStr,
    PadMode,
    PadWhere,
    PadWidth,
    PadWidthLike,
    QuantileMethod,
    SliceInfo,
)

if TYPE_CHECKING:
    import dask.array as da
    from numpy.typing import NDArray
    from xarray.core.types import DaCompatible

Array: TypeAlias = "NDArray[Any] | da.Array"


def _resolve_pad_mode(mode: PadMode):
    constant_value = None
    if isinstance(mode, str):
        mode_name = mode
    elif isinstance(mode, v0_5.ConstantPadding):
        mode_name = mode.mode
        constant_value = mode.value
    elif isinstance(
        mode, (v0_5.EdgePadding, v0_5.ReflectPadding, v0_5.SymmetricPadding)
    ):
        mode_name = mode.mode
    else:
        assert_never(mode)

    return mode_name, constant_value


# TODO: complete docstrings
# TODO: in the long run---with improved typing in xarray---we should probably replace `Tensor` with xr.DataArray
class Tensor(MagicTensorOpsMixin):
    """A wrapper around an xr.DataArray for better integration with bioimageio.spec
    and improved type annotations."""

    _Compatible: TypeAlias = Union["Tensor", xr.DataArray, DaCompatible]

    def __init__(
        self,
        array: Union[Array, xr.DataArray],
        dims: Sequence[Union[AxisId, AxisLike]],
    ) -> None:
        super().__init__()
        axes = tuple(
            a if isinstance(a, AxisId) else AxisInfo.create(a).id for a in dims
        )
        if isinstance(array, xr.DataArray):
            self._data: xr.DataArray = array.transpose(*axes)
            assert isinstance(self._data, xr.DataArray)
        else:
            self._data = xr.DataArray(array, dims=axes)

    def __repr__(self) -> str:
        return f"<Tensor {repr(self._data)}>"

    def __array__(self, dtype: Optional[DTypeLike] = None):
        return np.asarray(self._data, dtype=dtype)

    def __getitem__(
        self,
        key: Union[
            SliceInfo,
            slice,
            int,
            PerAxis[Union[SliceInfo, slice, int]],
            Tensor,
            xr.DataArray,
        ],
    ) -> Self:
        if isinstance(key, SliceInfo):
            key = slice(*key)
        elif isinstance(key, collections.abc.Mapping):
            key = {
                a: s if isinstance(s, int) else s if isinstance(s, slice) else slice(*s)
                for a, s in key.items()
            }
        elif isinstance(key, Tensor):
            key = key._data

        return self.__class__.from_xarray(self._data[key])

    def __setitem__(
        self,
        key: Union[PerAxis[Union[SliceInfo, slice]], Tensor, xr.DataArray],
        value: Union[Tensor, xr.DataArray, float, int],
    ) -> None:
        if isinstance(key, Tensor):
            key = key._data
        elif isinstance(key, xr.DataArray):
            pass
        else:
            key = {a: s if isinstance(s, slice) else slice(*s) for a, s in key.items()}

        if isinstance(value, Tensor):
            value = value._data

        self._data[key] = value

    def __len__(self) -> int:
        return len(self.data)

    def _iter(self: Any) -> Iterator[Any]:
        for n in range(len(self)):
            yield self[n]

    def __iter__(self: Any) -> Iterator[Any]:
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        return self._iter()

    def _binary_op(
        self,
        other: _Compatible,
        f: Callable[[Any, Any], Any],
        reflexive: bool = False,
    ) -> Self:
        data: xr.DataArray = self._data._binary_op(  # pyright: ignore[reportPrivateUsage]
            (other._data if isinstance(other, Tensor) else other),
            f,
            reflexive,
        )
        assert isinstance(data, xr.DataArray)
        return self.__class__.from_xarray(data)

    def _inplace_binary_op(
        self,
        other: _Compatible,
        f: Callable[[Any, Any], Any],
    ) -> Self:
        _ = self._data._inplace_binary_op(  # pyright: ignore[reportPrivateUsage]
            (
                other_d
                if (other_d := getattr(other, "data")) is not None
                and isinstance(
                    other_d,
                    xr.DataArray,
                )
                else other
            ),
            f,
        )
        return self

    def _unary_op(self, f: Callable[[Any], Any], *args: Any, **kwargs: Any) -> Self:
        data: xr.DataArray = self._data._unary_op(  # pyright: ignore[reportPrivateUsage]
            f, *args, **kwargs
        )
        assert isinstance(data, xr.DataArray)
        return self.__class__.from_xarray(data)

    @classmethod
    def from_xarray(cls, data_array: xr.DataArray) -> Self:
        """create a `Tensor` from an xarray data array

        note for internal use: this factory method is round-trip save
            for any `Tensor`'s  `data` property (an xarray.DataArray).
        """
        return cls(array=data_array, dims=tuple(AxisId(d) for d in data_array.dims))

    @classmethod
    def from_numpy(
        cls,
        array: NDArray[Any],
        *,
        dims: Optional[Union[AxisLike, Sequence[AxisLike]]],
    ) -> Tensor:
        return cls.from_array(array, dims=dims)

    @classmethod
    def from_array(
        cls,
        array: Array,
        *,
        dims: Optional[Union[AxisLike, Sequence[AxisLike]]],
    ) -> Tensor:
        """create a `Tensor` from a numpy array

        Args:
            array: the nd numpy array
            dims: A description of the array's axes.
                If None axes are guessed (which might fail and raise a ValueError.)
                If dims do not match array shape, permutations and singleton dimensions are tried to find a match.
        Raises:
            ValueError: if `dims` is None and dims guessing fails.
        """

        if dims is None:
            return cls._interprete_array_wo_known_axes(array)
        elif isinstance(dims, collections.abc.Sequence):
            dim_seq = list(dims)
        else:
            dim_seq = [dims]

        axis_infos = [AxisInfo.create(a) for a in dim_seq]
        original_shape = tuple(array.shape)

        successful_view = _get_array_view(array, axis_infos)
        if successful_view is None:
            raise ValueError(
                f"Array shape {original_shape} does not map to axes {dims}"
            )

        return Tensor(successful_view, dims=tuple(a.id for a in axis_infos))

    @property
    def data(self):
        return self._data

    @property
    def dims(self):  # TODO: rename to `axes`?
        """Tuple of dimension names associated with this tensor."""
        return cast(Tuple[AxisId, ...], self._data.dims)

    @property
    def dtype(self) -> DTypeStr:
        dt = str(self.data.dtype)
        assert dt in get_args(DTypeStr)
        return dt  # pyright: ignore[reportReturnType]

    @property
    def ndim(self):
        """Number of tensor dimensions."""
        return self._data.ndim

    @property
    def shape(self):
        """Tuple of tensor axes lengths"""
        return self._data.shape

    @property
    def shape_tuple(self):
        """Tuple of tensor axes lengths"""
        return self._data.shape

    @property
    def size(self):
        """Number of elements in the tensor.

        Equal to math.prod(tensor.shape), i.e., the product of the tensors’ dimensions.
        """
        return self._data.size

    @property
    def sizes(self):
        """Ordered, immutable mapping from axis ids to axis lengths."""
        return cast(Mapping[AxisId, int], self.data.sizes)

    @property
    def tagged_shape(self):
        """(alias for `sizes`) Ordered, immutable mapping from axis ids to lengths."""
        return self.sizes

    def to_numpy(self) -> NDArray[Any]:
        """Return the data of this tensor as a numpy array."""
        return self.data.to_numpy()

    def argmax(self) -> Mapping[AxisId, int]:
        ret = self._data.argmax(...)
        assert is_dict(ret)
        return {cast(AxisId, k): cast(int, v.item()) for k, v in ret.items()}

    def astype(self, dtype: DTypeStr, *, copy: bool = False):
        """Return tensor cast to `dtype`

        note: if dtype is already satisfied copy if `copy`"""
        return self.__class__.from_xarray(self._data.astype(dtype, copy=copy))

    def clip(self, min: Optional[float] = None, max: Optional[float] = None):
        """Return a tensor whose values are limited to [min, max].
        At least one of max or min must be given."""
        return self.__class__.from_xarray(self._data.clip(min, max))

    def crop_to(
        self,
        sizes: PerAxis[int],
        crop_where: Union[
            CropWhere,
            PerAxis[CropWhere],
        ] = "left_and_right",
    ) -> Self:
        """crop to match `sizes`"""
        if isinstance(crop_where, str):
            crop_axis_where: PerAxis[CropWhere] = {a: crop_where for a in self.dims}
        else:
            crop_axis_where = crop_where

        slices: Dict[AxisId, SliceInfo] = {}

        for a, s_is in self.sizes.items():
            if a not in sizes or sizes[a] == s_is:
                pass
            elif sizes[a] > s_is:
                logger.warning(
                    "Cannot crop axis {} of size {} to larger size {}",
                    a,
                    s_is,
                    sizes[a],
                )
            elif a not in crop_axis_where:
                raise ValueError(
                    f"Don't know where to crop axis {a}, `crop_where`={crop_where}"
                )
            else:
                crop_this_axis_where = crop_axis_where[a]
                if crop_this_axis_where == "left":
                    slices[a] = SliceInfo(s_is - sizes[a], s_is)
                elif crop_this_axis_where == "right":
                    slices[a] = SliceInfo(0, sizes[a])
                elif crop_this_axis_where == "left_and_right":
                    slices[a] = SliceInfo(
                        start := (s_is - sizes[a]) // 2, sizes[a] + start
                    )
                else:
                    assert_never(crop_this_axis_where)

        return self[slices]

    def expand_dims(self, dims: Union[Sequence[AxisId], PerAxis[int]]) -> Self:
        return self.__class__.from_xarray(self._data.expand_dims(dims=dims))

    def item(
        self,
        key: Union[
            None, SliceInfo, slice, int, PerAxis[Union[SliceInfo, slice, int]]
        ] = None,
    ):
        """Copy a tensor element to a standard Python scalar and return it."""
        if key is None:
            ret = self._data.item()
        else:
            ret = self[key]._data.item()

        assert isinstance(ret, (bool, float, int))
        return ret

    def mean(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        return self.__class__.from_xarray(self._data.mean(dim=dim))

    def pad(
        self,
        pad_width: PerAxis[PadWidthLike],
        mode: PadMode = "symmetric",
    ) -> Self:
        pad_width = {a: PadWidth.create(p) for a, p in pad_width.items()}
        mode_name, constant_value = _resolve_pad_mode(mode)
        return self.__class__.from_xarray(
            self._data.pad(
                pad_width=pad_width, mode=mode_name, constant_values=constant_value
            )
        )

    def pad_to(
        self,
        sizes: PerAxis[int],
        pad_where: Union[PadWhere, PerAxis[PadWhere]] = "left_and_right",
        mode: PadMode = "symmetric",
    ) -> Self:
        """pad `tensor` to match `sizes`"""
        if isinstance(pad_where, str):
            pad_axis_where: PerAxis[PadWhere] = {a: pad_where for a in self.dims}
        else:
            pad_axis_where = pad_where

        pad_width: Dict[AxisId, PadWidth] = {}
        for a, s_is in self.sizes.items():
            if a not in sizes or sizes[a] == s_is:
                pad_width[a] = PadWidth(0, 0)
            elif s_is > sizes[a]:
                pad_width[a] = PadWidth(0, 0)
                logger.warning(
                    "Cannot pad axis {} of size {} to smaller size {}",
                    a,
                    s_is,
                    sizes[a],
                )
            elif a not in pad_axis_where:
                raise ValueError(
                    f"Don't know where to pad axis {a}, `pad_where`={pad_where}"
                )
            else:
                pad_this_axis_where = pad_axis_where[a]
                d = sizes[a] - s_is
                if pad_this_axis_where == "left":
                    pad_width[a] = PadWidth(d, 0)
                elif pad_this_axis_where == "right":
                    pad_width[a] = PadWidth(0, d)
                elif pad_this_axis_where == "left_and_right":
                    pad_width[a] = PadWidth(left := d // 2, d - left)
                else:
                    assert_never(pad_this_axis_where)

        return self.pad(pad_width, mode)

    def quantile(
        self,
        q: Union[float, Sequence[float]],
        dim: Optional[Union[AxisId, Sequence[AxisId]]] = None,
        method: QuantileMethod = "linear",
    ) -> Self:
        assert (
            isinstance(q, (float, int))
            and q >= 0.0
            or not isinstance(q, (float, int))
            and all(qq >= 0.0 for qq in q)
        )
        assert (
            isinstance(q, (float, int))
            and q <= 1.0
            or not isinstance(q, (float, int))
            and all(qq <= 1.0 for qq in q)
        )
        assert dim is None or (
            (quantile_dim := AxisId("quantile")) != dim and quantile_dim not in set(dim)
        )
        return self.__class__.from_xarray(
            self._data.quantile(q, dim=dim, method=method)
        )

    def resize_to(
        self,
        sizes: PerAxis[int],
        *,
        pad_where: Union[
            PadWhere,
            PerAxis[PadWhere],
        ] = "left_and_right",
        crop_where: Union[
            CropWhere,
            PerAxis[CropWhere],
        ] = "left_and_right",
        pad_mode: PadMode = "symmetric",
    ):
        """return cropped/padded tensor with `sizes`"""
        crop_to_sizes: Dict[AxisId, int] = {}
        pad_to_sizes: Dict[AxisId, int] = {}
        new_axes = dict(sizes)
        for a, s_is in self.sizes.items():
            a = AxisId(str(a))
            _ = new_axes.pop(a, None)
            if a not in sizes or sizes[a] == s_is:
                pass
            elif s_is > sizes[a]:
                crop_to_sizes[a] = sizes[a]
            else:
                pad_to_sizes[a] = sizes[a]

        tensor = self
        if crop_to_sizes:
            tensor = tensor.crop_to(crop_to_sizes, crop_where=crop_where)

        if pad_to_sizes:
            tensor = tensor.pad_to(pad_to_sizes, pad_where=pad_where, mode=pad_mode)

        if new_axes:
            tensor = tensor.expand_dims(new_axes)

        return tensor

    def std(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        return self.__class__.from_xarray(self._data.std(dim=dim))

    def sum(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        """Reduce this Tensor's data by applying sum along some dimension(s)."""
        return self.__class__.from_xarray(self._data.sum(dim=dim))

    def assign_batch_multi_index(self, multi_index: "pd.MultiIndex") -> Self:
        """Set the batch multi-index for this tensor.

        Args:
            multi_index: The multi-index to set.
        """
        if AxisId("batch") not in self.dims:
            raise ValueError(
                "Cannot set batch multi-index on a tensor without a 'batch' axis."
            )

        return self.__class__.from_xarray(
            self._data.assign_coords({AxisId("batch"): multi_index})
        )

    def unstack_batch_multi_index(
        self, *, errors: Literal["raise", "ignore"] = "raise"
    ) -> Self:
        """Unstack the batch multi-index of this tensor.

        Returns:
            A new tensor with the batch multi-index unstacked into separate axes.
        """
        if AxisId("batch") not in self.dims:
            if errors == "raise":
                raise ValueError(
                    "Cannot unstack batch multi-index on a tensor without a 'batch' axis."
                )
            elif errors == "ignore":
                return self
            else:
                assert_never(errors)

        if not isinstance(self._data.indexes.get(AxisId("batch")), pd.MultiIndex):
            if errors == "raise":
                raise ValueError(
                    "Cannot unstack batch multi-index on a tensor whose 'batch' axis does not have a MultiIndex."
                )
            elif errors == "ignore":
                return self
            else:
                assert_never(errors)

        old_dims = self.dims
        array = self._data.unstack(AxisId("batch"))
        assert isinstance(array, xr.DataArray)
        added_dims = [AxisId(d) for d in array.dims if d not in self._data.dims]

        # restore expected axis order, replace batch dim with added dims
        new_dims: List[AxisId] = []
        for d in old_dims:
            if d in array.dims:
                new_dims.append(d)
            elif d == AxisId("batch"):
                new_dims.extend(added_dims)
            else:
                raise ValueError(f"Expected axis {d} not found in unstacked array.")

        array = array.transpose(*new_dims)
        if AxisId("original_batch") in array.dims:
            array = array.rename({AxisId("original_batch"): AxisId("batch")})

        assert isinstance(array, xr.DataArray)
        return self.__class__.from_xarray(array)

    def transpose(
        self,
        axes: Sequence[AxisId],
        *,
        extra_dims: Literal[
            "raise", "squeeze", "stack", "squeeze_or_stack"
        ] = "squeeze",
        missing_dims: Literal[
            "raise", "expand", "unstack", "unstack_or_expand"
        ] = "unstack_or_expand",
    ) -> Self:
        """Return a transposed tensor, missing axes are expanded (if `unstack_missing_dims_from_batch` is False) or unstacked from batch (if `unstack_missing_dims_from_batch` is True), extra axes are stacked to batch (if `stack_extra_dims_to_batch` is True). Additional axes raise (if `stack_extra_dims_to_batch` is True).

        Args:
            axes: The desired tensor axes
            extra_dims:
                Extra dimensions are any dimensions in the tensor that are not specified in `axes`.
                If "raise", any extra dimensions will raise an error.
                If "squeeze", any extra singleton dimensions will be squeezed, non-singleton dimensions will raise an error.
                If "stack", any extra dimensions will be stacked to the batch dimension. Such a stacked batch dimension then has a multi-index that can be unstacked using `Tensor.unstack_batch_multi_index()`.
                If "squeeze_or_stack", any extra singleton dimensions will be squeezed, non-singleton dimensions will be stacked to the batch dimension.
            missing_dims:
                Missing dimensions are any dimensions specified in `axes` that are not present in the tensor.
                If "raise", any missing dimensions will raise an error.
                If "expand", any missing dimensions will be added as singleton dimensions.
                If "unstack", any missing dimensions will be unstacked from the batch dimension. For this option a batch dimension with a multi-index must be present from previous stacking operations or assigned by `Tensor.assign_batch_multi_index()`.
                If "unstack_or_expand", any missing dimensions will be unstacked from the batch dimension if it has a multi-index, otherwise they will be added as singleton dimensions.
        """
        array = self._data

        unhandled_missing_dims = [a for a in axes if a not in array.dims]
        if unhandled_missing_dims and missing_dims == "raise":
            raise ValueError(f"Found missing dimensions {unhandled_missing_dims}.")

        unstack_error = None
        if unhandled_missing_dims and missing_dims in ("unstack", "unstack_or_expand"):
            lets_unstack = AxisId("batch") in array.dims
            if not lets_unstack:
                unstack_error = f"Missing dimensions {unhandled_missing_dims} found, but 'batch' axis is not in the tensor. Cannot unstack missing dimensions from batch."
                if missing_dims == "unstack":
                    raise ValueError(unstack_error)

            if lets_unstack and not isinstance(
                array.indexes.get(AxisId("batch")), pd.MultiIndex
            ):
                lets_unstack = False
                unstack_error = f"Missing dimensions {unhandled_missing_dims} found, but 'batch' axis does not have a MultiIndex. Cannot unstack missing dimensions from non-multi-index batch."
                if missing_dims == "unstack":
                    raise ValueError(unstack_error)
        else:
            lets_unstack = False

        if lets_unstack:
            array: xr.DataArray = array.unstack(AxisId("batch"))

            if AxisId("original_batch") in array.dims:
                if AxisId("batch") in axes:
                    array = array.rename({AxisId("original_batch"): AxisId("batch")})
                else:
                    array = array.squeeze(AxisId("original_batch"))

            unhandled_missing_dims = [a for a in axes if a not in array.dims]

        if unhandled_missing_dims and missing_dims in ("expand", "unstack_or_expand"):
            array = array.expand_dims(unhandled_missing_dims)
            unhandled_missing_dims = []

        if unhandled_missing_dims:
            if unstack_error is not None:
                raise ValueError(unstack_error)

            raise ValueError(f"Missing dimensions {unhandled_missing_dims}.")

        assert isinstance(array, xr.DataArray)
        unhandled_extra_dims = [a for a in array.dims if a not in axes]

        if unhandled_extra_dims and extra_dims == "raise":
            raise ValueError(f"Found extra dimensions {unhandled_extra_dims}.")

        if unhandled_extra_dims and extra_dims in ("squeeze", "squeeze_or_stack"):
            for d in list(unhandled_extra_dims):
                if array.sizes[d] == 1:
                    array = array.squeeze(d)
                    unhandled_extra_dims.remove(d)
                elif extra_dims == "squeeze":
                    raise ValueError(
                        f"Extra dimension {d} found but stack_extra_dims_to_batch is False and the dimension is not a singleton."
                    )

        if unhandled_extra_dims and extra_dims in ("stack", "squeeze_or_stack"):
            if AxisId("batch") not in axes:
                raise ValueError(
                    f"Extra dimensions {unhandled_extra_dims} found but 'batch' axis is not in the desired axes {axes}."
                    + " Cannot stack extra dimensions to batch."
                )

            if AxisId("batch") in array.dims:
                array = array.rename({AxisId("batch"): AxisId("original_batch")})
                unhandled_extra_dims.insert(0, AxisId("original_batch"))

            array = array.stack({AxisId("batch"): unhandled_extra_dims})
            unhandled_extra_dims = []

        if unhandled_extra_dims:
            raise ValueError(
                f"Non-singleton extra dimensions {unhandled_extra_dims} found, but `extra_dims` not in ('stack', 'squeeze_or_stack')."
            )

        # transpose to the correct axis order
        return self.__class__.from_xarray(array.transpose(*axes))

    def var(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        return self.__class__.from_xarray(self._data.var(dim=dim))

    @classmethod
    def _interprete_array_wo_known_axes(cls, array: Array) -> Tensor:
        ndim = array.ndim
        shape = [s if isinstance(s, int) else -1 for s in array.shape]

        if ndim == 2:
            current_axes = (
                v0_5.SpaceInputAxis(id=v0_5.AxisId("y"), size=shape[0]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("x"), size=shape[1]),
            )
        elif ndim == 3 and any(s <= 3 for s in shape):
            current_axes = (
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(shape[0])
                    ]
                ),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("y"), size=shape[1]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("x"), size=shape[2]),
            )
        elif ndim == 3:
            current_axes = (
                v0_5.SpaceInputAxis(id=v0_5.AxisId("z"), size=shape[0]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("y"), size=shape[1]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("x"), size=shape[2]),
            )
        elif ndim == 4:
            current_axes = (
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(shape[0])
                    ]
                ),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("z"), size=shape[1]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("y"), size=shape[2]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("x"), size=shape[3]),
            )
        elif ndim == 5:
            current_axes = (
                v0_5.BatchAxis(),
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(shape[1])
                    ]
                ),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("z"), size=shape[2]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("y"), size=shape[3]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("x"), size=shape[4]),
            )
        else:
            raise ValueError(f"Could not guess an axis mapping for {shape}")

        dims = tuple(a.id for a in current_axes)
        if isinstance(array, da.Array):
            return cls.from_xarray(xr.DataArray(array, dims=dims))
        else:
            return cls(array, dims=dims)


def _add_singletons(arr: NDArray[Any], axis_infos: Sequence[AxisInfo]):
    if len(arr.shape) > len(axis_infos):
        # remove singletons
        for i, s in enumerate(arr.shape):
            if s == 1:
                arr = np.take(arr, 0, axis=i)
                if len(arr.shape) == len(axis_infos):
                    break

    # add singletons if nececsary
    for i, a in enumerate(axis_infos):
        if len(arr.shape) >= len(axis_infos):
            break

        if a.size.min == 1:
            arr = np.expand_dims(arr, i)

    return arr


def _get_array_view(
    original_array: Array, axis_infos: Sequence[AxisInfo]
) -> Optional[Array]:
    perms = list(permutations(range(len(original_array.shape))))

    for perm in perms:
        view: Array = original_array.transpose(perm)  # pyright: ignore
        view = _add_singletons(view, axis_infos)  # pyright: ignore
        if len(view.shape) != len(axis_infos):
            return None

        for s, a in zip(view.shape, axis_infos):
            if (
                s < a.size.min
                or (a.size.max is not None and s > a.size.max)
                or (a.size.step is not None and (s - a.size.min) % a.size.step != 0)
            ):
                break
        else:
            return view

    return None
