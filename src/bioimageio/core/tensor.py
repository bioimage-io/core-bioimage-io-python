from __future__ import annotations

import collections.abc
from collections.abc import Collection, Iterator, Mapping, Sequence
from itertools import permutations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Union,
    cast,
    get_args,
)

import dask.array
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from numpy.typing import DTypeLike, NDArray
from typing_extensions import Self, TypeAlias, assert_never

from bioimageio.spec.model import v0_5
from bioimageio.spec.model.v0_5 import SpaceUnit, TimeUnit

from ._magic_tensor_ops import MagicTensorOpsMixin
from ._utils.color import get_default_channel_colors
from ._utils.type_guards import is_dict, is_mapping, is_sequence
from ._utils.type_hints import cast_and_assert
from .axis import BATCH_AXIS_ID, AxisId, AxisInfo, AxisLike, PerAxis
from .common import (
    CropWhere,
    DTypeStr,
    Halo,
    HaloLike,
    MemberId,
    PadMode,
    PadWhere,
    PadWidth,
    PadWidthLike,
    PerMember,
    QuantileMethod,
    SliceInfo,
    resolve_pad_mode,
)

MultiIndex = pd.MultiIndex

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from xarray.core.types import DaCompatible

Array: TypeAlias = "NDArray[Any] | dask.array.Array"


# TODO: complete docstrings
# TODO: in the long run---with improved typing in xarray---we should probably replace `Tensor` with xr.DataArray
class Tensor(MagicTensorOpsMixin):
    """A wrapper around an xr.DataArray for better integration with bioimageio.spec
    and improved type annotations."""

    _Compatible: TypeAlias = Union["Tensor", xr.DataArray, "DaCompatible"]

    def __init__(
        self,
        array: Array | xr.DataArray,
        dims: Sequence[AxisId | AxisLike],
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
        return f"<Tensor {self._data!r}>"

    def __array__(self, dtype: DTypeLike | None = None):
        return np.asarray(self._data, dtype=dtype)

    def __getitem__(
        self,
        key: SliceInfo
        | slice
        | int
        | PerAxis[SliceInfo | slice | int]
        | Tensor
        | xr.DataArray,
    ) -> Self:
        if isinstance(key, SliceInfo):
            key = slice(key.start, key.stop)
        elif isinstance(key, collections.abc.Mapping):
            key = {
                a: s
                if isinstance(s, int)
                else s
                if isinstance(s, slice)
                else slice(s.start, s.stop)
                for a, s in key.items()
            }
        elif isinstance(key, Tensor):
            key = key._data

        selected = cast_and_assert(xr.DataArray, self._data[key])
        return self.__class__.from_xarray(selected)

    def __setitem__(
        self,
        key: PerAxis[SliceInfo | slice] | Tensor | xr.DataArray,
        value: Tensor | xr.DataArray | float,
    ) -> None:
        if isinstance(key, Tensor):
            key = key._data
        elif isinstance(key, xr.DataArray):
            pass
        else:
            key = {
                a: s if isinstance(s, slice) else slice(s.start, s.stop)
                for a, s in key.items()
            }

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
        data = cast_and_assert(
            xr.DataArray,
            self._data._binary_op(  # pyright: ignore[reportPrivateUsage]
                (other._data if isinstance(other, Tensor) else other),
                f,
                reflexive,
            ),
        )
        return self.__class__.from_xarray(data)

    def get_physical_scale(self) -> PerAxis[float]:
        """Return the physical scale of each axis in units (see `get_physical_scale_unit`)."""
        _scale = self._data.attrs.get("physical_scale", None)
        scale: Mapping[Any, Any] = _scale if is_mapping(_scale) else {}
        del _scale
        return {
            a if isinstance(a, AxisId) else AxisId(a): 1.0 if s is None else float(s)
            for a, s in scale.items()
        }

    def set_physical_scale(self, scale: PerAxis[float | None]) -> None:
        """Set the physical scale of each axis in units (see `set_physical_scale_unit`)."""
        if unknown_axes := [d for d in scale if d not in self.dims]:
            raise ValueError(
                f"Cannot set physical scale for axes {unknown_axes} not in tensor axes {self.dims}"
            )

        self._data.attrs["physical_scale"] = {
            d: 1.0 if (s := scale.get(d)) is None else float(s) for d in self.dims
        }

    def get_physical_scale_unit(self) -> PerAxis[SpaceUnit | TimeUnit | str]:
        """Return the physical unit of each axis."""
        _unit = self._data.attrs.get("physical_scale_unit", None)
        unit: Mapping[Any, Any] = _unit if is_mapping(_unit) else {}
        del _unit
        return {d: "" if (u := unit.get(d)) is None else str(u) for d in self.dims}

    def set_physical_scale_unit(
        self, unit: PerAxis[SpaceUnit | TimeUnit | str | None]
    ) -> None:
        """Set the physical unit of each axis."""
        if unknown_axes := [d for d in unit if d not in self.dims]:
            raise ValueError(
                f"Cannot set physical scale unit for axes {unknown_axes} not in tensor axes {self.dims}"
            )
        self._data.attrs["physical_scale_unit"] = {
            d: "" if (u := unit.get(d)) is None else str(u) for d in self.dims
        }

    @property
    def channel_names(self) -> list[str] | None:
        """Return the channel names if available, otherwise None."""

        names = self._data.attrs.get("channel_names", None)
        if not is_sequence(names):
            return None

        if len(names) != self.sizes.get(AxisId("channel"), 0):
            logger.warning(
                "Number of channel names ({}) does not match number of channels ({})",
                len(names),
                self.sizes.get(AxisId("channel"), 0),
            )
            return None

        return [str(name) for name in names]

    @channel_names.setter
    def channel_names(self, names: list[str] | None) -> None:
        """Set the channel names."""
        if names is not None and len(names) != self.sizes.get(AxisId("channel"), 0):
            logger.warning(
                "Number of channel names ({}) does not match number of channels ({})",
                len(names),
                self.sizes.get(AxisId("channel"), 0),
            )
        self._data.attrs["channel_names"] = names

    @property
    def channel_colors(self) -> list[str] | None:
        """Return channel colors ."""

        n_channels = self.sizes.get(AxisId("channel"), 0)
        if n_channels == 0:
            return None

        colors = self._data.attrs.get("channel_colors", None)
        if not is_sequence(colors):
            colors = None

        if colors is not None and len(colors) < n_channels:
            logger.warning(
                "Overwriting existing channel colors with default colors due to too few channel colors: {} colors for {} channels",
                len(colors),
                n_channels,
            )
            colors = None

        if colors is None:
            default_channel_colors = get_default_channel_colors(n_channels)
            self.channel_colors = default_channel_colors
            return default_channel_colors
        else:
            return [str(c) for c in colors[:n_channels]]

    @channel_colors.setter
    def channel_colors(self, colors: list[str] | None) -> None:
        """Set the channel colors."""
        n_channels = self.sizes.get(AxisId("channel"), 0)
        if colors is not None and len(colors) < n_channels:
            logger.warning(
                "Number of channel colors ({}) lower than number of channels ({})",
                len(colors),
                n_channels,
            )
        self._data.attrs["channel_colors"] = colors

    @property
    def name(self) -> str | None:
        """Return the name of the tensor if available, otherwise None."""
        name = self._data.attrs.get("name")
        return None if name is None else str(name)

    @name.setter
    def name(self, name: str | None) -> None:
        """Set the name of the tensor."""
        self._data.attrs["name"] = name

    def squeeze(self, dim: AxisId | Sequence[AxisId] | None = None) -> Self:
        """Return a tensor with all the singleton dimensions removed."""
        squeezed = self._data.squeeze(dim=dim)  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(squeezed, xr.DataArray)
        return self.__class__.from_xarray(squeezed)

    def _inplace_binary_op(
        self,
        other: _Compatible,
        f: Callable[[Any, Any], Any],
    ) -> Self:
        _ = self._data._inplace_binary_op(  # pyright: ignore[reportPrivateUsage]
            (
                other_d
                if (other_d := other.data) is not None
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
        dims = [AxisId(d) for d in data_array.dims]
        dim_map = {d: a for d, a in zip(data_array.dims, dims) if d != a}
        if dim_map:
            data_array = data_array.rename(dim_map)  # pyright: ignore[reportUnknownVariableType]
            assert isinstance(data_array, xr.DataArray)

        return cls(array=data_array, dims=dims)

    @classmethod
    def from_numpy(
        cls,
        array: NDArray[Any],
        *,
        dims: AxisLike | Sequence[AxisLike] | None,
    ) -> Self:
        return cls.from_array(array, dims=dims)

    @classmethod
    def from_array(
        cls,
        array: Array,
        *,
        dims: AxisLike | Sequence[AxisLike] | None,
    ) -> Self:
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
            dims_formatted = "".join(f"\n{d}" for d in dim_seq)
            raise ValueError(
                f"Array shape {original_shape} does not map to axes: {dims_formatted}"
            )

        return cls(successful_view, dims=tuple(a.id for a in axis_infos))

    @property
    def data(self):
        return self._data

    @property
    def dims(self):  # TODO: rename to `axes`?
        """Tuple of dimension names associated with this tensor."""
        return cast(tuple[AxisId, ...], self._data.dims)

    @property
    def dtype(self) -> DTypeStr:
        dt = str(self.data.dtype)  # pyright: ignore[reportUnknownArgumentType]
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
        """Return the data of this tensor as a numpy array.

        Use `as_numpy()` to get a new `Tensor` instance with the data as a numpy array (wrapped in an xarray DataArray)."""
        return self.data.to_numpy()

    def as_numpy(self) -> Self:
        """Return a new Tensor instance with this tensor's data as a numpy array (wrapped in an xarray DataArray).
        Use `to_numpy()` to get the raw numpy array."""
        return self.from_xarray(cast_and_assert(xr.DataArray, self.data.as_numpy()))

    def crop_halo(self, halo: PerAxis[HaloLike] | None) -> Self:
        """helper to crop a halo from the tensor"""
        if not halo:
            return self

        return self[
            {
                d: slice(h[0], self.tagged_shape[d] - h[1])
                if (h := Halo.create(halo.get(d, 0))) != Halo(0, 0)
                else slice(None)
                for d in self.dims
            }
        ]

    def chunk(
        self,
        chunk_sizes: PerAxis[int],
        halo: PerAxis[HaloLike] | None = None,
        pad_mode: PadMode | Literal["none"] = "none",
    ) -> Self:
        """Return a tensor with the same data but chunked according to `chunk_sizes`."""
        if da is None:
            raise ImportError(
                "Dask is not installed. Please install dask to use the chunking functionality."
            )

        if pad_mode != "none" and halo is not None:
            data = self.pad(pad_width=halo, mode=pad_mode).data
        else:
            data = self.data

        chunked = cast_and_assert(xr.DataArray, data.chunk(chunk_sizes))
        if halo is not None:
            for a, h in halo.items():
                if a not in self.dims:
                    raise ValueError(f"Axis {a} not in tensor dimensions {self.dims}")

                if isinstance(h, int):
                    if h < 0:
                        raise ValueError(
                            f"Halo size for axis {a} must be non-negative, got {h}"
                        )
                elif isinstance(h, tuple):
                    if len(h) != 2 or any(x < 0 for x in h):
                        raise ValueError(
                            f"Halo size for axis {a} must be a non-negative integer or a tuple of two non-negative integers, got {h}"
                        )

                    if len(set(h)) != 1 and pad_mode != "none":
                        raise ValueError(
                            f"Halo size for chunking with pad_mode {pad_mode} must be symmetric, got {h}"
                        )

                else:
                    raise TypeError(
                        f"Halo size for axis {a} must be a non-negative integer or a tuple of two non-negative integers, got {h}"
                    )

            depth = [
                h
                if isinstance((h := halo.get(d, 0)), int)
                else h[0]
                if len(set(h)) == 1
                else {"-": h[0], "+": h[1]}
                for d in self.dims
            ]

            enlarged_chunk_shape = tuple(
                chunk_sizes.get(a, self.tagged_shape[a])
                + (2 * h if isinstance(h, int) else h["-"] + h["+"])
                for a, h in zip(self.dims, depth)
            )

            def map_overlap_chunk(x: NDArray[Any]) -> NDArray[Any]:
                return x

            overlapping = cast_and_assert(
                da.Array,
                da.map_overlap(
                    map_overlap_chunk,
                    chunked.data,
                    depth=depth,
                    boundary="none",  # we explicitly pad above to avoid mapping `pad_mode` to `boundary`
                    dtype=chunked.data.dtype,
                    chunks=enlarged_chunk_shape,
                    trim=False,
                    allow_rechunk=False,
                    # meta=np.array((), dtype=chunked.data.dtype),  # possibly not a numpy array
                ),
            )
            ret = self.__class__.from_array(overlapping, dims=self.dims)
            if self.batch_multi_index is not None:
                ret = ret.assign_batch_multi_index(self.batch_multi_index)
        else:
            ret = self.__class__.from_xarray(chunked)

        assert ret.name == self.name
        assert ret.channel_names == self.channel_names
        assert ret.channel_colors == self.channel_colors
        assert (
            ret.batch_multi_index is None
            and self.batch_multi_index is None
            or (ret.batch_multi_index == self.batch_multi_index).all()
        )
        return ret

    def argmax(self) -> Mapping[AxisId, int]:
        ret = self._data.argmax(...)
        assert is_dict(ret)
        return {cast(AxisId, k): cast(int, v.item()) for k, v in ret.items()}

    def astype(self, dtype: DTypeStr, *, copy: bool = False):
        """Return tensor cast to `dtype`

        note: if dtype is already satisfied copy if `copy`"""
        return self.__class__.from_xarray(
            cast_and_assert(xr.DataArray, self._data.astype(dtype, copy=copy))
        )

    def clip(self, min: float | None = None, max: float | None = None):
        """Return a tensor whose values are limited to [min, max].
        At least one of max or min must be given."""
        return self.__class__.from_xarray(
            cast_and_assert(xr.DataArray, self._data.clip(min, max))
        )

    def crop_to(
        self,
        sizes: PerAxis[int],
        crop_where: CropWhere | PerAxis[CropWhere] = "left_and_right",
    ) -> Self:
        """crop to match `sizes`"""
        if isinstance(crop_where, str):
            crop_axis_where: PerAxis[CropWhere] = {a: crop_where for a in self.dims}
        else:
            crop_axis_where = crop_where

        slices: dict[AxisId, SliceInfo] = {}

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

    def expand_dims(self, dims: Sequence[AxisId] | PerAxis[int]) -> Self:
        return self.__class__.from_xarray(
            cast_and_assert(xr.DataArray, self._data.expand_dims(dims=dims))
        )

    def item(
        self,
        key: None | SliceInfo | slice | int | PerAxis[SliceInfo | slice | int] = None,
    ):
        """Copy a tensor element to a standard Python scalar and return it."""
        if key is None:
            ret = self._data.item()
        else:
            ret = self[key]._data.item()

        assert isinstance(ret, (bool, float, int))
        return ret

    def mean(self, dim: AxisId | Sequence[AxisId] | None = None) -> Self:
        return self.__class__.from_xarray(self._data.mean(dim=dim))

    def pad(
        self,
        pad_width: PerAxis[PadWidthLike],
        mode: PadMode = "symmetric",
    ) -> Self:
        pad_width = {a: PadWidth.create(p) for a, p in pad_width.items()}
        mode_name, constant_value = resolve_pad_mode(mode)

        padded_tensor = self.__class__.from_xarray(
            cast_and_assert(
                xr.DataArray,
                self._data.pad(
                    pad_width=pad_width,
                    mode=mode_name,
                    constant_values=constant_value,
                    keep_attrs=True,
                ),
            )
        )

        if (
            BATCH_AXIS_ID in self.dims
            # maybe future xarray versions will keep a MultiIndex when padding, but currently they don't
            and padded_tensor.batch_multi_index is not None
            and pad_width.get(BATCH_AXIS_ID, 0) in (0, (0, 0), PadWidth(0, 0))
            and (idx := self.batch_multi_index) is not None
        ):
            # add multi-index removed by padding when batch axis is not padded
            padded_tensor = padded_tensor.assign_batch_multi_index(idx)

        return padded_tensor

    def pad_to(
        self,
        sizes: PerAxis[int],
        pad_where: PadWhere | PerAxis[PadWhere] = "left_and_right",
        mode: PadMode = "symmetric",
    ) -> Self:
        """pad `tensor` to match `sizes`"""
        if isinstance(pad_where, str):
            pad_axis_where: PerAxis[PadWhere] = {a: pad_where for a in self.dims}
        else:
            pad_axis_where = pad_where

        pad_width: dict[AxisId, PadWidth] = {}
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
        q: float | Sequence[float],
        dim: AxisId | Sequence[AxisId] | None = None,
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
            cast_and_assert(
                xr.DataArray, self._data.quantile(q, dim=dim, method=method)
            )
        )

    def resize_to(
        self,
        sizes: PerAxis[int],
        *,
        pad_where: PadWhere | PerAxis[PadWhere] = "left_and_right",
        crop_where: CropWhere | PerAxis[CropWhere] = "left_and_right",
        pad_mode: PadMode = "symmetric",
    ):
        """return cropped/padded tensor with `sizes`"""
        crop_to_sizes: dict[AxisId, int] = {}
        pad_to_sizes: dict[AxisId, int] = {}
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

    def std(self, dim: AxisId | Sequence[AxisId] | None = None) -> Self:
        return self.__class__.from_xarray(
            cast_and_assert(xr.DataArray, self._data.std(dim=dim))
        )

    def sum(self, dim: AxisId | Sequence[AxisId] | None = None) -> Self:
        """Reduce this Tensor's data by applying sum along some dimension(s)."""
        return self.__class__.from_xarray(
            cast_and_assert(xr.DataArray, self._data.sum(dim=dim))
        )

    @property
    def batch_multi_index(self) -> MultiIndex | None:
        """Return the batch multi-index if available, otherwise None."""
        if BATCH_AXIS_ID not in self.dims:
            return None

        idx = self._data.indexes.get(BATCH_AXIS_ID)  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(idx, MultiIndex):
            return None

        return idx

    def assign_batch_multi_index(self, multi_index: MultiIndex) -> Self:
        """Set the batch multi-index for this tensor.

        Args:
            multi_index: The multi-index to set.
        """
        if BATCH_AXIS_ID not in self.dims:
            raise ValueError(
                "Cannot set batch multi-index on a tensor without a 'batch' axis."
            )

        # # TODO: check if we really need to create/assign coordinates object
        # alternatively create new multi-index: self._data.set_xindex(multi_index.names)
        data = cast_and_assert(
            xr.DataArray,
            self._data.drop_vars(
                list(multi_index.names) + [BATCH_AXIS_ID], errors="ignore"
            ),
        )
        mindex_coords = cast_and_assert(
            xr.Coordinates,
            xr.Coordinates.from_pandas_multiindex(multi_index, BATCH_AXIS_ID),
        )
        data = cast_and_assert(xr.DataArray, data.assign_coords(mindex_coords))
        return self.__class__.from_xarray(data)

    def unstack_batch_multi_index(
        self,
        *,
        errors: Literal["raise", "warn", "ignore"] = "raise",
        unstacked_dim_order: Sequence[AxisId] | None = None,
    ) -> Self:
        """Unstack the batch multi-index of this tensor.

        Returns:
            A new tensor with the batch multi-index unstacked into separate axes.
        """
        if BATCH_AXIS_ID not in self.dims:
            error_msg = (
                "Cannot unstack batch multi-index on a tensor without a 'batch' axis."
            )
            if errors == "raise":
                raise ValueError(error_msg)
            elif errors == "warn":
                logger.warning(error_msg)
                return self
            elif errors == "ignore":
                return self
            else:
                assert_never(errors)

        if not isinstance(self._data.indexes.get(BATCH_AXIS_ID), MultiIndex):
            error_msg = "Cannot unstack batch multi-index on a tensor whose 'batch' axis does not have a MultiIndex."
            if errors == "raise":
                raise ValueError(error_msg)
            elif errors == "warn":
                logger.warning(error_msg)
                return self
            elif errors == "ignore":
                return self
            else:
                assert_never(errors)

        old_dims = self.dims
        array = cast_and_assert(xr.DataArray, self._data.unstack(BATCH_AXIS_ID))
        added_dims = [
            BATCH_AXIS_ID if d == "original_batch" else AxisId(d)
            for d in array.dims
            if d not in self._data.dims
        ]

        if unstacked_dim_order is None:
            # restore expected axis order, replace batch dim with added dims
            new_dims: Sequence[AxisId] = []
            for d in old_dims:
                if d in array.dims:
                    new_dims.append(d)
                elif d == BATCH_AXIS_ID:
                    new_dims.extend(added_dims)
                else:
                    raise ValueError(f"Expected axis {d} not found in unstacked array.")
        else:
            new_dims = unstacked_dim_order

        if AxisId("original_batch") in array.dims:
            array = cast_and_assert(
                xr.DataArray, array.rename({AxisId("original_batch"): BATCH_AXIS_ID})
            )

        array = cast_and_assert(xr.DataArray, array.transpose(*new_dims))
        return self.__class__.from_xarray(array)

    def transpose(
        self,
        axes: Sequence[AxisLike],
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
        transposed, _dim_order = self.transpose_and_return_unstacked_dim_order(
            axes, extra_dims=extra_dims, missing_dims=missing_dims
        )
        return transposed

    def transpose_and_return_unstacked_dim_order(
        self,
        axes: Sequence[AxisLike],
        *,
        extra_dims: Literal[
            "raise", "squeeze", "stack", "squeeze_or_stack"
        ] = "squeeze",
        missing_dims: Literal[
            "raise", "expand", "unstack", "unstack_or_expand"
        ] = "unstack_or_expand",
    ) -> tuple[Self, list[AxisId] | None]:
        axes = [a if isinstance(a, AxisId) else AxisId(a) for a in axes]
        array = self._data

        unhandled_missing_dims = [a for a in axes if a not in self.dims]
        if unhandled_missing_dims and missing_dims == "raise":
            raise ValueError(f"Found missing dimensions {unhandled_missing_dims}.")

        unstack_error = None
        if unhandled_missing_dims and missing_dims in ("unstack", "unstack_or_expand"):
            lets_unstack = BATCH_AXIS_ID in array.dims
            if not lets_unstack:
                unstack_error = f"Missing dimensions {unhandled_missing_dims} found, but 'batch' axis is not in the tensor. Cannot unstack missing dimensions from batch."
                if missing_dims == "unstack":
                    raise ValueError(unstack_error)

            if lets_unstack and not isinstance(
                array.indexes.get(BATCH_AXIS_ID), MultiIndex
            ):
                lets_unstack = False
                unstack_error = f"Missing dimensions {unhandled_missing_dims} found, but 'batch' axis does not have a MultiIndex. Cannot unstack missing dimensions from non-multi-index batch."
                if missing_dims == "unstack":
                    raise ValueError(unstack_error)
        else:
            lets_unstack = False

        if lets_unstack:
            array = cast_and_assert(xr.DataArray, array.unstack(BATCH_AXIS_ID))

            if AxisId("original_batch") in array.dims:
                if BATCH_AXIS_ID in axes:
                    array = cast_and_assert(
                        xr.DataArray,
                        array.rename({AxisId("original_batch"): BATCH_AXIS_ID}),
                    )
                else:
                    array = cast_and_assert(
                        xr.DataArray, array.squeeze(AxisId("original_batch"))
                    )

            unhandled_missing_dims = [a for a in axes if a not in array.dims]

        if unhandled_missing_dims and missing_dims in ("expand", "unstack_or_expand"):
            array = cast_and_assert(
                xr.DataArray, array.expand_dims(unhandled_missing_dims)
            )
            unhandled_missing_dims = []

        if unhandled_missing_dims:
            if unstack_error is not None:
                raise ValueError(unstack_error)

            raise ValueError(f"Missing dimensions {unhandled_missing_dims}.")

        assert isinstance(array, xr.DataArray)
        unstacked_dim_order = None
        unhandled_extra_dims = [a for a in array.dims if a not in axes]

        if unhandled_extra_dims and extra_dims == "raise":
            raise ValueError(f"Found extra dimensions {unhandled_extra_dims}.")

        if unhandled_extra_dims and extra_dims in ("squeeze", "squeeze_or_stack"):
            for d in list(unhandled_extra_dims):
                if array.sizes[d] == 1:
                    array = cast_and_assert(xr.DataArray, array.squeeze(d))
                    assert isinstance(array, xr.DataArray)
                    unhandled_extra_dims.remove(d)
                elif extra_dims == "squeeze":
                    raise ValueError(
                        f"Extra dimension {d} found but stack_extra_dims_to_batch is False and the dimension is not a singleton."
                    )

        if unhandled_extra_dims and extra_dims in ("stack", "squeeze_or_stack"):
            if BATCH_AXIS_ID not in axes:
                raise ValueError(
                    f"Extra dimensions {unhandled_extra_dims} found but 'batch' axis is not in the desired axes {axes}."
                    + " Cannot stack extra dimensions to batch."
                )

            unstacked_dim_order = [AxisId(d) for d in array.dims]
            if BATCH_AXIS_ID in array.dims:
                array = cast_and_assert(
                    xr.DataArray,
                    array.rename({BATCH_AXIS_ID: AxisId("original_batch")}),
                )
                unhandled_extra_dims.insert(0, AxisId("original_batch"))

            array = cast_and_assert(
                xr.DataArray, array.stack({BATCH_AXIS_ID: unhandled_extra_dims})
            )
            unhandled_extra_dims = []

        if unhandled_extra_dims:
            raise ValueError(
                f"Non-singleton extra dimensions {unhandled_extra_dims} found, but `extra_dims` not in ('stack', 'squeeze_or_stack')."
            )

        # transpose to the correct axis order
        return self.__class__.from_xarray(
            cast_and_assert(xr.DataArray, array.transpose(*axes))
        ), unstacked_dim_order

    def var(self, dim: AxisId | Sequence[AxisId] | None = None) -> Self:
        return self.__class__.from_xarray(
            cast_and_assert(xr.DataArray, self._data.var(dim=dim))
        )

    @classmethod
    def _interprete_array_wo_known_axes(cls, array: Array) -> Self:
        ndim = array.ndim
        shape = [s if isinstance(s, int) else -1 for s in array.shape]

        current_axes: list[v0_5.InputAxis]
        if ndim == 2:
            current_axes = [
                v0_5.SpaceInputAxis(id=v0_5.AxisId("y"), size=shape[0]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("x"), size=shape[1]),
            ]
        elif ndim == 3 and any(s <= 10 for s in shape):
            current_axes = [
                v0_5.SpaceInputAxis(id=v0_5.AxisId("y"), size=shape[1]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("x"), size=shape[2]),
            ]
            current_axes.insert(
                np.argmin(shape),
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(min(shape))
                    ]
                ),
            )
        elif ndim == 3:
            current_axes = [
                v0_5.SpaceInputAxis(id=v0_5.AxisId("z"), size=shape[0]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("y"), size=shape[1]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("x"), size=shape[2]),
            ]
        elif ndim == 4:
            current_axes = [
                v0_5.SpaceInputAxis(id=v0_5.AxisId("z"), size=shape[1]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("y"), size=shape[2]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("x"), size=shape[3]),
            ]
            current_axes.insert(
                np.argmin(shape),
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(min(shape))
                    ]
                ),
            )
        elif ndim == 5:
            current_axes = [
                v0_5.TimeInputAxis(size=shape[1]),
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(shape[1])
                    ]
                ),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("z"), size=shape[2]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("y"), size=shape[3]),
                v0_5.SpaceInputAxis(id=v0_5.AxisId("x"), size=shape[4]),
            ]
        else:
            raise ValueError(f"Could not guess an axis mapping for {shape}")

        dims = tuple(a.id for a in current_axes)
        return cls(array, dims=dims)


def restore_batch_multi_index(
    inputs: Collection[Tensor | MultiIndex | None], outputs: PerMember[Tensor]
) -> dict[MemberId, Tensor]:
    """Restore any batch multi-index found in the inputs to any output with a batch dimension of matching length."""
    not_processed_outputs = dict(outputs)
    ret: dict[MemberId, Tensor] = {}
    for m in tuple(outputs):
        if BATCH_AXIS_ID not in outputs[m].dims:
            ret[m] = not_processed_outputs.pop(m)

    input_has_multi_index = False
    for ipt in inputs:
        if isinstance(ipt, Tensor):
            idx = ipt.batch_multi_index
        else:
            idx = ipt

        if not isinstance(idx, MultiIndex):
            continue

        input_has_multi_index = True
        for m in tuple(not_processed_outputs):
            if len(idx) == not_processed_outputs[m].tagged_shape.get(BATCH_AXIS_ID):
                ret[m] = not_processed_outputs.pop(m).assign_batch_multi_index(idx)

    if not_processed_outputs:
        if input_has_multi_index:
            logger.warning(
                "Could not restore batch multi-index for outputs: {}",
                list(not_processed_outputs),
            )
        ret.update(not_processed_outputs)

    return ret


def get_unstacked_dim_order(
    stacked_order: Sequence[AxisId],
    *,
    unstacked_reference_order: Sequence[AxisId] | None,
    stacked_dims: Sequence[AxisId] | None,
) -> list[AxisId]:
    if stacked_dims is not None:
        # make sure that any "original_batch" axis id in the stacked_dims is replaced with the actual batch axis id
        stacked_dims = [
            BATCH_AXIS_ID if d == "original_batch" else d for d in stacked_dims
        ]

    if not stacked_dims and unstacked_reference_order is None:
        return list(stacked_order)
    if unstacked_reference_order is not None and list(stacked_order) == [
        d for d in unstacked_reference_order if d in stacked_order
    ]:
        # no differences in non-stacking axes, so we can preserve the unstacked axis order
        return [
            d
            for d in unstacked_reference_order
            if d in stacked_order or (stacked_dims is not None and d in stacked_dims)
        ]
    elif stacked_dims is None:
        # no stacked dims provided, so we can just return the stacked_out_order
        return list(stacked_order)
    elif BATCH_AXIS_ID not in stacked_order:
        raise ValueError(
            f"stacked_order={stacked_order} does not contain a batch axis, but stacked_dims={stacked_dims} are provided"
        )
    else:
        # unstack batch axis in its place
        ret = list(stacked_order)
        b_idx = ret.index(AxisId("batch"))
        ret[b_idx : b_idx + 1] = (
            [
                d for d in unstacked_reference_order or () if d in stacked_dims
            ]  # keep unstacked axis order from reference
            + [
                d
                for d in stacked_dims
                if unstacked_reference_order is None
                or d not in unstacked_reference_order
            ]  # add any remaining stacked dims that are not in the unstacked reference order
        )
        return ret


# def transpose_multiple_tensors(
#     tensors: PerMember[Tensor],
#     axes: PerMember[Sequence[AxisId]],
#     *,
#     extra_dims: Literal["raise", "squeeze", "stack", "squeeze_or_stack"] = "raise",
#     missing_dims: Literal["raise", "expand", "unstack", "unstack_or_expand"] = "raise",
# ) -> PerMember[Tensor]:
#     """Transpose multiple tensors and make
#     """
#     if any(unknown := [m not in tensors for m in axes]):
#         raise ValueError(f"Axes specified for unknown members: {unknown}")

#     transposed_tensors: dict[MemberId, Tensor] = {}
#     for m, t in tensors.items():
#         if m in axes:
#             transposed_tensors[m] = (
#                 t.transpose(
#                     axes=axes[m],
#                     extra_dims=extra_dims,
#                     missing_dims=missing_dims,
#                 )
#             )
#         else:
#             transposed_tensors[m] = t

#     if (
#         len(
#             batch_lengths := {
#                 t.sizes[AxisId("batch")]
#                 for t in transposed_tensors.values()
#                 if AxisId("batch") in t.dims
#             }
#         )
#         > 1
#     ):
#         raise ValueError(
#             f"Transposed sample members have incompatible batch lengths: {batch_lengths}."
#         )

#     return transposed_tensors


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
) -> Array | None:
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
