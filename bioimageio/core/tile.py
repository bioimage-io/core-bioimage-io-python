import itertools
from dataclasses import dataclass, field
from math import prod
from typing import Dict, Iterable, List, Mapping, Tuple, Union, cast

from xarray.core.utils import Frozen

from .common import AxisId, Data, LeftRight, SliceInfo, Stat, Tensor, TensorId

# TensorTilePos = Mapping[AxisId, int]
# TilePos = Mapping[TensorId, TensorTilePos]
TensorTileSlice = Mapping[AxisId, SliceInfo]
TileSlice = Mapping[TensorId, TensorTileSlice]
TensorSampleSize = Mapping[AxisId, int]
SampleSizes = Mapping[TensorId, TensorSampleSize]
TensorTileHalo = Mapping[AxisId, LeftRight]
TileHalo = Mapping[TensorId, TensorTileHalo]


@dataclass
class Tile:
    """A tile of a dataset sample"""

    data: Data
    """the tile's tensors"""

    inner_slice: TileSlice
    """slice of the inner tile (without padding and overlap) of the sample"""

    halo: TileHalo
    """pad/overlap to extend the (inner) tile (to the outer tile)"""

    outer_slice: Frozen[TensorId, Frozen[AxisId, SliceInfo]] = field(init=False)
    """slice of the outer tile (including overlap, but not padding) in the sample"""

    overlap: Frozen[TensorId, Frozen[AxisId, LeftRight]] = field(init=False)
    """overlap 'into a neighboring tile'"""

    padding: Frozen[TensorId, Frozen[AxisId, LeftRight]] = field(init=False)
    """pad (at sample edges where we cannot overlap to realize `halo`"""

    def __post_init__(self):
        self.outer_slice = Frozen(
            {
                t: Frozen(
                    {
                        a: SliceInfo(
                            max(0, self.inner_slice[t][a].start - self.halo[t][a].left),
                            min(
                                self.sample_sizes[t][a],
                                self.inner_slice[t][a].stop + self.halo[t][a].right,
                            ),
                        )
                        for a in self.inner_slice[t]
                    }
                )
                for t in self.inner_slice
            }
        )
        self.overlap = Frozen(
            {
                tid: Frozen(
                    {
                        a: LeftRight(
                            self.inner_slice[tid][a].start
                            - self.outer_slice[tid][a].start,
                            self.outer_slice[tid][a].stop
                            - self.inner_slice[tid][a].stop,
                        )
                        for a in self.inner_slice[tid]
                    }
                )
                for tid in self.inner_slice
            }
        )
        self.padding = Frozen(
            {
                tid: Frozen(
                    {
                        a: LeftRight(
                            self.halo[tid][a].left - self.overlap[tid][a].left,
                            self.halo[tid][a].right - self.overlap[tid][a].right,
                        )
                        for a in self.inner_slice[tid]
                    }
                )
                for tid in self.inner_slice
            }
        )

    tile_number: int
    """the n-th tile of the sample"""

    tiles_in_sample: int
    """total number of tiles of the sample"""

    sample_sizes: SampleSizes
    """the axis sizes of the sample"""

    stat: Stat = field(default_factory=dict)
    """sample and dataset statistics"""


def _tile_generator(tensor: Tensor, all_1d_tiles: List[List[Tuple[int, slice]]]):
    axes = cast(Tuple[AxisId, ...], tensor.dims)
    for i, tile in enumerate(itertools.product(*all_1d_tiles)):
        pos: TensorTilePos = {a: p for a, (p, _) in zip(axes, tile)}
        tile_slice = {a: s for a, (_, s) in zip(axes, tile)}
        yield i, pos, tensor[tile_slice]


def tile_tensor(
    tensor: Tensor,
    tile_shape: Mapping[AxisId, int],
    pad_width: Mapping[AxisId, Union[int, Tuple[int, int]]],
) -> Tuple[int, Iterable[Tuple[int, TensorTilePos, Tensor]]]:
    """tile a tensor

    Args:
        tile_shape: output tile shape
        pad_width: padding at edge of sample, overlap with neighboring tiles within the sample

    """
    assert all(aid in tensor.dims for aid in tile_shape), (tensor.dims, set(tile_shape))
    assert all(aid in tensor.dims for aid in pad_width), (tensor.dims, set(pad_width))
    assert all(aid in tile_shape for aid in tensor.dims), (tensor.dims, set(tile_shape))
    assert all(aid in pad_width for aid in tensor.dims), (tensor.dims, set(pad_width))

    axes = cast(Tuple[AxisId, ...], tensor.dims)

    all_1d_tiles: List[List[Tuple[int, slice]]] = []
    shape = tensor.shape
    for aid, s in zip(axes, shape):
        pad = _pad if isinstance(_pad := pad_width[aid], tuple) else (_pad, _pad)
        stride = tile_shape[aid] - sum(pad)
        tiles_1d = [
            (p, slice(max(0, p - pad[0]), min(s, p + pad[1])))
            for p in range(0, s, stride)
        ]
        all_1d_tiles.append(tiles_1d)

    n_tiles = prod(map(len, all_1d_tiles))

    return n_tiles, _tile_generator(tensor, all_1d_tiles)
