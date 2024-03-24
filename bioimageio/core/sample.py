from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union, cast
from typing_extensions import Self
from bioimageio.core.common import AxisId, Data, Stat, Tensor, TensorId

from .tile import SampleSizes, TensorTilePos, Tile, TilePos, tile_tensor

TiledSample = Iterable[Tile]
"""A dataset sample split into tiles"""


@dataclass
class Sample:
    """A dataset sample"""

    data: Data
    """the sample's tensors"""

    stat: Stat = field(default_factory=dict)
    """sample and dataset statistics"""

    @property
    def sizes(self) -> SampleSizes:
        return {tid: cast(Dict[AxisId, int], dict(t.sizes)) for tid, t in self.data.items()}

    def tile(
        self,
        tile_shape: Mapping[TensorId, Mapping[AxisId, int]],
        pad_width: Mapping[TensorId, Mapping[AxisId, Union[int, Tuple[int, int]]]],
    ) -> TiledSample:
        return tile_sample(self, tile_shape, pad_width)

    @classmethod
    def from_tiles(cls, tiles: Iterable[Tile]) -> Self:
        data: Data = {}
        stat: Stat = {}
        for tile in tiles:
            for tid, tile_data in tile.data.items():

            stat = tile.stat

        return cls(data=data, stat=stat)

def tile_sample(
    sample: Sample,
    tile_shape: Mapping[TensorId, Mapping[AxisId, int]],
    pad_width: Mapping[TensorId, Mapping[AxisId, Union[int, Tuple[int, int]]]],
):
    assert all(tid in sample.data for tid in tile_shape), (tile_shape, sample.data)
    assert all(tid in pad_width for tid in tile_shape), (tile_shape, pad_width)
    tensor_ids = list(tile_shape)

    tile_generators: Dict[TensorId, Iterable[Tuple[int, TensorTilePos, Tensor]]] = {}
    n_tiles: Dict[TensorId, int] = {}
    for tid in tensor_ids:
        n_tiles[tid], tile_generators[tid] = tile_tensor(
            sample.data[tid], tile_shape=tile_shape[tid], pad_width=pad_width[tid]
        )

    n_tiles_common: Optional[int] = None
    single_tile_tensors: Dict[TensorId, Tuple[TensorTilePos, Tensor]] = {}
    tile_iterators: Dict[TensorId, Iterator[Tuple[int, TensorTilePos, Tensor]]] = {}
    for tid, n in n_tiles.items():
        tile_iterator = iter(tile_generators[tid])
        if n == 1:
            t0, pos, tensor_tile = next(tile_iterator)
            assert t0 == 0
            single_tile_tensors[tid] = (pos, tensor_tile)
            continue

        if n_tiles_common is None:
            n_tiles_common = n
        elif n != n_tiles_common:
            raise ValueError(
                f"{sample} tiled by {tile_shape} yields different numbers of tiles: {n_tiles}"
            )

        tile_iterators[tid] = tile_iterator

    if n_tiles_common is None:
        assert not tile_iterators
        n_tiles_common = 1

    for t in range(n_tiles_common):
        data: Dict[TensorId, Tensor] = {}
        tile_pos: TilePos = {}
        for tid, (tensor_pos, tensor_tile) in single_tile_tensors.items():
            data[tid] = tensor_tile
            tile_pos[tid] = tensor_pos

        for tid, tile_iterator in tile_iterators.items():
            assert tid not in data
            assert tid not in tile_pos
            _t, tensor_pos, tensor_tile = next(tile_iterator)
            assert _t == t, (_t, t)
            data[tid] = tensor_tile
            tile_pos[tid] = tensor_pos

        yield Tile(
            data=data,
            pos=tile_pos,
            tile_number=t,
            tiles_in_sample=n_tiles_common,
            stat=sample.stat,
        )
