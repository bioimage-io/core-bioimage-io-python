from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union, cast

import numpy
from typing_extensions import Self
from xarray.core.utils import Frozen

from .axis import AxisId, PerAxis
from .common import Halo, HaloLike, PadMode, PadWidth, SliceInfo, TileNumber
from .stat_measures import Stat
from .tensor import PerTensor, Tensor, TensorId
from .tile import Tile, tile_tensor

TiledSample = Iterable[Tile]
"""A dataset sample split into tiles"""


@dataclass
class Sample:
    """A dataset sample"""

    data: PerTensor[Tensor]
    """the sample's tensors"""

    stat: Stat = field(default_factory=dict)
    """sample and dataset statistics"""

    @property
    def sizes(self) -> PerTensor[PerAxis[int]]:
        return {tid: t.sizes for tid, t in self.data.items()}

    def tile(
        self,
        tile_sizes: PerTensor[PerAxis[int]],
        minimum_halo: PerTensor[PerAxis[HaloLike]],
    ) -> TiledSample:
        assert not (
            missing := [t for t in tile_sizes if t not in self.data]
        ), f"`tile_sizes` specified for missing tensors: {missing}"
        assert not (
            missing := [t for t in minimum_halo if t not in tile_sizes]
        ), f"`minimum_halo` specified for tensors without `tile_sizes`: {missing}"

        tensor_ids = list(tile_sizes)

        tensor_tile_generators: Dict[
            TensorId, Iterable[Tuple[TileNumber, Tensor, PerAxis[SliceInfo]]]
        ] = {}
        n_tiles: Dict[TensorId, int] = {}
        for t in tensor_ids:
            n_tiles[t], tensor_tile_generators[t] = tile_tensor(
                self.data[t],
                tile_sizes=tile_sizes.get(t, self.data[t].sizes),
                minimum_halo=minimum_halo.get(t, {a: 0 for a in self.data[t].dims}),
                pad_mode=pad_mode,
            )

        n_tiles_common: Optional[int] = None
        single_tile_tensors: Dict[TensorId, Tuple[TensorTilePos, Tensor]] = {}
        tile_iterators: Dict[TensorId, Iterator[Tuple[int, TensorTilePos, Tensor]]] = {}
        for t, n in n_tiles.items():
            tile_iterator = iter(tensor_tile_generators[t])
            if n == 1:
                t0, pos, tensor_tile = next(tile_iterator)
                assert t0 == 0
                single_tile_tensors[t] = (pos, tensor_tile)
                continue

            if n_tiles_common is None:
                n_tiles_common = n
            elif n != n_tiles_common:
                raise ValueError(
                    f"{self} tiled by {tile_sizes} yields different numbers of tiles: {n_tiles}"
                )

            tile_iterators[t] = tile_iterator

        if n_tiles_common is None:
            assert not tile_iterators
            n_tiles_common = 1

        for t in range(n_tiles_common):
            data: Dict[TensorId, Tensor] = {}
            tile_pos: TilePos = {}
            inner_slice: TileSlice = {}
            outer_slice: TileSlice = {}
            for t, (tensor_tile, tensor_pos) in single_tile_tensors.items():
                data[t] = tensor_tile
                tile_pos[t] = tensor_pos
                inner_slice[t] = inner_tensor_slice
                outer_slice[t] = outer_tensor_slice

            for t, tile_iterator in tile_iterators.items():
                assert t not in data
                assert t not in tile_pos
                _t, tensor_pos, tensor_tile = next(tile_iterator)
                assert _t == t, (_t, t)
                data[t] = tensor_tile
                tile_pos[t] = tensor_pos

            yield Tile(
                data=data,
                pos=tile_pos,
                inner_slice=inner_slice,
                outer_slice=outer_slice,
                tile_number=t,
                tiles_in_self=n_tiles_common,
                stat=self.stat,
            )

    @classmethod
    def from_tiles(
        cls, tiles: Iterable[Tile], *, fill_value: float = float("nan")
    ) -> Self:
        # TODO: add `mode: Literal['in-memory', 'to-disk']` or similar to save out of mem samples
        data: TileData = {}
        stat: Stat = {}
        for tile in tiles:
            for t, tile_data in tile.inner_data.items():
                if t not in data:
                    axes = cast(Tuple[AxisId], tile_data.dims)
                    data[t] = Tensor(
                        numpy.full(
                            tuple(tile.sample_sizes[t][a] for a in axes),
                            fill_value,
                            dtype=tile_data.dtype,
                        ),
                        dims=axes,
                        id=t,
                    )

                data[t][tile.inner_slice[t]] = tile_data

            stat = tile.stat

        return cls(data=data, stat=stat)
