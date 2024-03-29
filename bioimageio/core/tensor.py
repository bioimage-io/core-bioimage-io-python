from __future__ import annotations

import itertools
from math import prod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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
from bioimageio.spec.model import v0_5

from ._magic_tensor_ops import MagicTensorOpsMixin
from .axis import Axis, AxisId, AxisInfo, AxisLike
from .common import (
    CropWhere,
    DTypeStr,
    Halo,
    HaloLike,
    PadWidth,
    PadWidthLike,
    SliceInfo,
    TileNumber,
    TotalNumberOfTiles,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
TensorId = v0_5.TensorId

T = TypeVar("T")

PerTensor = Mapping[TensorId, T]


_ScalarOrArray = Union["ArrayLike", np.generic, "NDArray[Any]"]  # TODO: add "DaskArray"


# TODO: make Tensor a numpy compatible array type, to use e.g. \
#   with `np.testing.assert_array_almost_equal`.
# TODO: complete docstrings
class Tensor(MagicTensorOpsMixin):
    """A wrapper around an xr.DataArray for better integration with bioimageio.spec
    and improved type annotations."""

    _Compatible = Union["Tensor", xr.DataArray, _ScalarOrArray]

    def __init__(
        self,
        array: NDArray[Any],
        dims: Union[AxisId, Sequence[AxisId]],
        id: Optional[TensorId] = None,
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

    def _binary_op(
        self,
        other: _Compatible,
        f: Callable[[Any, Any], Any],
        reflexive: bool = False,
    ) -> Self:
        data = self._data._binary_op(  # pyright: ignore[reportPrivateUsage]
            (other._data if isinstance(other, Tensor) else other),
            f,
            reflexive,
        )
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
        data = self._data._unary_op(  # pyright: ignore[reportPrivateUsage]
            f, *args, **kwargs
        )
        return self.__class__.from_xarray(data)

    @classmethod
    def from_xarray(cls, data_array: xr.DataArray) -> Self:
        """create a `Tensor` from an xarray data array

        note for internal use: this factory method is round-trip save
            for any `Tensor`'s  `data` property (an xarray.DataArray).
        """
        return cls(
            array=data_array.data,
            dims=tuple(AxisId(d) for d in data_array.dims),
            id=None if data_array.name is None else TensorId(data_array.name),
        )

    @classmethod
    def from_numpy(
        cls,
        array: NDArray[Any],
        *,
        dims: Optional[Union[AxisLike, Sequence[AxisLike]]],
        id: TensorId,
    ) -> Tensor:
        """create a `Tensor` from a numpy array

        Args:
            array: the nd numpy array
            axes: A description of the array's axes,
                if None axes are guessed (which might fail and raise a ValueError.)
            id: the created tensor's identifier

        Raises:
            ValueError: if `axes` is None and axes guessing fails.
        """

        if dims is None:
            return cls._interprete_array_wo_known_axes(array, id=id)
        elif isinstance(dims, (str, Axis, v0_5.AxisBase)):
            dims = [dims]

        axis_infos = [AxisInfo.create(a) for a in dims]
        original_shape = tuple(array.shape)
        if len(array.shape) > len(dims):
            # remove singletons
            for i, s in enumerate(array.shape):
                if s == 1:
                    array = np.take(array, 0, axis=i)
                    if len(array.shape) == len(dims):
                        break

        # add singletons if nececsary
        for a in axis_infos:

            if len(array.shape) >= len(dims):
                break

            if a.maybe_singleton:
                array = array[None]

        if len(array.shape) != len(dims):
            raise ValueError(
                f"Array shape {original_shape} does not map to axes {dims}"
            )

        return Tensor(array, dims=tuple(a.id for a in axis_infos), id=id)

    @property
    def data(self):
        return self._data

    @property
    def dims(self):  # TODO: rename to `axes`?
        """Tuple of dimension names associated with this tensor."""
        return cast(Tuple[AxisId, ...], self._data.dims)

    @property
    def shape(self):
        """Tuple of tensor dimension lenghts"""
        return self._data.shape

    @property
    def size(self):
        """Number of elements in the tensor.

        Equal to math.prod(tensor.shape), i.e., the product of the tensors’ dimensions.
        """
        return self._data.size

    def sum(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        """Reduce this Tensor's data by applying sum along some dimension(s)."""
        return self.__class__.from_xarray(self._data.sum(dim=dim))

    @property
    def ndim(self):
        """Number of tensor dimensions."""
        return self._data.ndim

    @property
    def dtype(self) -> DTypeStr:
        dt = str(self.data.dtype)  # pyright: ignore[reportUnknownArgumentType]
        assert dt in get_args(DTypeStr)
        return dt  # pyright: ignore[reportReturnType]

    @property
    def id(self):
        """the tensor's identifier"""
        return self._id

    @property
    def sizes(self):
        """Ordered, immutable mapping from axis ids to lengths."""
        return cast(Mapping[AxisId, int], self.data.sizes)

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

    def mean(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        return self.__class__.from_xarray(self._data.mean(dims=dim))

    def std(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        return self.__class__.from_xarray(self._data.std(dims=dim))

    def var(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        return self.__class__.from_xarray(self._data.var(dims=dim))

    def pad(
        self,
        pad_width: PerAxis[PadWidthLike],
        mode: PadMode = "symmetric",
    ) -> Self:
        pad_width = {a: PadWidth.create(p) for a, p in pad_width.items()}
        return self.__class__.from_xarray(
            self._data.pad(pad_width=pad_width, mode=mode)
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
        self, q: float, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None
    ) -> Self:
        assert q >= 0.0
        assert q <= 1.0
        return self.__class__.from_xarray(self._data.quantile(q, dim=dim))

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

    def tile(
        self,
        tile_size: PerAxis[int],
        halo: PerAxis[HaloLike],
        pad_mode: PadMode,
    ) -> Tuple[
        TotalNumberOfTiles,
        Generator[Tuple[TileNumber, Tensor, PedrAxis[SliceInfo]], Any, None],
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

        # fill in default halo (0) and tile_size (tensor size)
        halo = {a: Halo.create(halo.get(a, 0)) for a in self.dims}
        tile_size = {a: tile_size.get(a, s) for a, s in self.sizes.items()}

        inner_1d_tiles: List[List[SliceInfo]] = []
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
        # expand missing tensor axes
        missing_axes = tuple(a for a in axes if a not in self.dims)
        array = self._data
        if missing_axes:
            array = array.expand_dims(missing_axes)

        # transpose to the correct axis order
        return self.__class__.from_xarray(array.transpose(*axes))

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
