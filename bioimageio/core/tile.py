import itertools
from dataclasses import dataclass, field
from math import prod
from typing import Dict, Iterable, List, Mapping, Tuple, Union, cast

from .common import AxisId, Data, LeftRight, Stat, Tensor, TensorId

TensorTilePos = Dict[AxisId, int]
TilePos = Dict[TensorId, TensorTilePos]
TensorSampleSize = Dict[AxisId, int]
SampleSizes = Dict[TensorId, TensorSampleSize]


@dataclass
class Tile:
    """A tile of a dataset sample"""

    data: Data
    """the tile's tensors"""

    pos: TilePos
    """position of the inner origin (origin of tile if halo is cropped) within the sample"""

    halo: Dict[AxisId, LeftRight]
    """padded or overlapping border region"""

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
