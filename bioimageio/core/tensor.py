from __future__ import annotations

import itertools
from math import prod
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    get_args,
)

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import Self, assert_never

from bioimageio.core.axis import PerAxis
from bioimageio.core.common import PadMode, PadWhere
from bioimageio.spec.model import v0_4, v0_5

from .axis import Axis, AxisId, AxisInfo, AxisLike
from .common import (
    DTypeStr,
    Halo,
    HaloLike,
    PadWidth,
    SliceInfo,
    TileNumber,
    TotalNumberOfTiles,
)

TensorId = v0_5.TensorId

T = TypeVar("T")
PerTensor = Mapping[TensorId, T]


class Tensor:
    def __init__(
        self,
        array: NDArray[Any],
        dims: Union[AxisId, Sequence[AxisId]],
        id: TensorId,
    ) -> None:
        super().__init__()
        self._data = xr.DataArray(array, dims=dims, name=id)
        self._id = id

    def __getitem__(self, key: PerAxis[Union[SliceInfo, slice]]) -> Self:
        key = {a: s if isinstance(s, slice) else slice(*s) for a, s in key.items()}
        return self.__class__.from_xarray(self._data[key])

    def __setitem__(self, key: PerAxis[Union[SliceInfo, slice]], value: Tensor) -> None:
        key = {a: s if isinstance(s, slice) else slice(*s) for a, s in key.items()}
        self._data[key] = value._data

    @classmethod
    def from_xarray(cls, data_array: xr.DataArray) -> Self:
        if data_array.name is None:
            raise ValueError(
                "Expected a named `data_array` to use `data_array.name` as tensor id"
            )

        return cls(
            array=data_array.data,
            dims=tuple(AxisId(d) for d in data_array.dims),
            id=TensorId(data_array.name),
        )

    @classmethod
    def from_numpy(
        cls, array: NDArray[Any], axes: Optional[Sequence[AxisLike]], id: TensorId
    ) -> Tensor:
        if axes is None:
            return cls._interprete_array_wo_known_axes(array, id=id)

        original_shape = tuple(array.shape)
        if len(array.shape) > len(axes):
            # remove singletons
            for i, s in enumerate(array.shape):
                if s == 1:
                    array = np.take(array, 0, axis=i)
                    if len(array.shape) == len(axes):
                        break

        # add singletons if nececsary
        for a in axes:
            a = AxisInfo.create(a)
            if len(array.shape) >= len(axes):
                break

            if a.maybe_singleton:
                array = array[None]

        if len(array.shape) != len(axes):
            raise ValueError(
                f"Array shape {original_shape} does not map to axes {axes}"
            )

        normalized_axes = normalize_axes(axes)
        assert len(normalized_axes) == len(axes)
        return Tensor(array, dims=tuple(a.id for a in normalized_axes))

    @property
    def data(self):
        return self._data

    @property
    def dims(self):
        return cast(Tuple[AxisId, ...], self._data.dims)

    @property
    def dtype(self) -> DTypeStr:
        dt = str(self.data.dtype)  # pyright: ignore[reportUnknownArgumentType]
        assert dt in get_args(DTypeStr)
        return dt  # pyright: ignore[reportReturnType]

    @property
    def id(self):
        return self._id

    @property
    def sizes(self):
        return cast(Mapping[AxisId, int], self.data.sizes)

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

    def mean(self, dim: Union[AxisId, Sequence[AxisId]]) -> Self:
        return self.__class__.from_xarray(self._data.mean(dims=dim))

    def std(self, dim: Union[AxisId, Sequence[AxisId]]) -> Self:
        return self.__class__.from_xarray(self._data.std(dims=dim))

    def var(self, dim: Union[AxisId, Sequence[AxisId]]) -> Self:
        return self.__class__.from_xarray(self._data.var(dims=dim))

    def pad(
        self,
        pad_width: PerAxis[PadWidth],
        mode: PadMode = "symmetric",
    ) -> Self:
        return self.__class__.from_xarray(
            self._data.pad(pad_width=pad_width, mode=mode)
        )

    def pad_to(
        self,
        sizes: PerAxis[int],
        pad_where: Union[PadWhere, PerAxis[PadWhere]] = "center",
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
                p = sizes[a] - s_is
                if pad_this_axis_where == "before":
                    pad_width[a] = PadWidth(p, 0)
                elif pad_this_axis_where == "after":
                    pad_width[a] = PadWidth(0, p)
                elif pad_this_axis_where == "center":
                    pad_width[a] = PadWidth(left := p // 2, p - left)
                else:
                    assert_never(pad_this_axis_where)

        return self.pad(pad_width, mode)

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
        pad_mode: PadMode = "symmetric",
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

    def tile(
        self,
        tile_size: PerAxis[int],
        halo: PerAxis[HaloLike],
        pad_mode: PadMode,
    ) -> Tuple[
        TotalNumberOfTiles,
        Generator[Tuple[TileNumber, Tensor, PerAxis[SliceInfo]], Any, None],
    ]:
        """tile this tensor into `tile_size` tiles that overlap by `halo`.
        At the tensor's edge the `halo` is padded with `pad_mode`.

        Args:
            tile_sizes: (Outer) output tile shape.
            halo: padding At the tensor's edge, overlap with neighboring tiles within
                the tensor; additional padding at the end of dimensions that do not
                evenly divide by the tile shape may result in larger halos for edge
                tiles.
            pad_mode: How to pad at the tensor's edge.
        """
        assert all(a in self.dims for a in tile_size), (self.dims, set(tile_size))
        assert all(a in self.dims for a in halo), (self.dims, set(halo))

        inner_1d_tiles: List[List[SliceInfo]] = []
        halo = {a: Halo.create(h) for a, h in halo.items()}
        for a, s in self.sizes.items():
            stride = tile_size[a] - sum(halo[a])
            tiles_1d = [SliceInfo(p, min(s, p + stride)) for p in range(0, s, stride)]
            inner_1d_tiles.append(tiles_1d)

        n_tiles = prod(map(len, inner_1d_tiles))

        return n_tiles, self._tile_generator(
            inner_1d_tiles=inner_1d_tiles, halo=halo, pad_mode=pad_mode
        )

    def transpose(
        self,
        axes: Sequence[AxisId],
    ) -> Self:
        """return a transposed tensor

        Args:
            axes: the desired tensor axes
        """
        # expand the missing image axes
        current_axes = tuple(
            d if isinstance(d, AxisId) else AxisId(d) for d in tensor.dims
        )
        missing_axes = tuple(a for a in axes if a not in current_axes)
        tensor = tensor.expand_dims(missing_axes)
        # transpose to the correct axis order
        return tensor.transpose(*map(str, axes))

    @classmethod
    def _interprete_array_wo_known_axes(cls, array: NDArray[Any], id: TensorId):
        ndim = array.ndim
        if ndim == 2:
            current_axes = (
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[0]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[1]),
            )
        elif ndim == 3 and any(s <= 3 for s in array.shape):
            current_axes = (
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(array.shape[0])
                    ]
                ),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[1]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[2]),
            )
        elif ndim == 3:
            current_axes = (
                v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[0]),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[1]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[2]),
            )
        elif ndim == 4:
            current_axes = (
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(array.shape[0])
                    ]
                ),
                v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[1]),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[2]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[3]),
            )
        elif ndim == 5:
            current_axes = (
                v0_5.BatchAxis(),
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(array.shape[1])
                    ]
                ),
                v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[2]),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[3]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[4]),
            )
        else:
            raise ValueError(f"Could not guess an axis mapping for {array.shape}")

        return cls(array, dims=tuple(a.id for a in current_axes), id=id)

    def _tile_generator(
        self,
        *,
        inner_1d_tiles: List[List[SliceInfo]],
        halo: PerAxis[Halo],
        pad_mode: PadMode,
    ):
        for i, nd_tile in enumerate(itertools.product(*inner_1d_tiles)):
            inner_slice: PerAxis[SliceInfo] = dict(zip(self.dims, nd_tile))
            outer_slice = {
                a: SliceInfo(
                    max(0, inner.start - halo[a].left),
                    min(self.sizes[a], inner.stop + halo[a].right),
                )
                for a, inner in inner_slice.items()
            }
            pad_width: PerAxis[PadWidth] = {
                a: PadWidth(
                    max(0, halo[a].left - inner.start),
                    max(0, inner.stop + halo[a].right - self.sizes[a]),
                )
                for a, inner in inner_slice.items()
            }

            yield i, self[outer_slice].pad(pad_width, pad_mode), inner_slice
