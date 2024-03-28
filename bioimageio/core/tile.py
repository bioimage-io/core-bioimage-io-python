from dataclasses import dataclass, field

from bioimageio.core.common import TileNumber, TotalNumberOfTiles

from .axis import PerAxis
from .common import Halo, LeftRight, PadWidth, SliceInfo
from .stat_measures import Stat
from .tensor import PerTensor, Tensor


@dataclass
class AbstractTile:
    """A tile of a dataset sample without any data"""

    inner_slice: PerTensor[PerAxis[SliceInfo]]
    """slice of the inner tile (without padding and overlap) of the sample"""

    halo: PerTensor[PerAxis[Halo]]
    """pad/overlap to extend the (inner) tile (to the outer tile)"""

    tile_number: TileNumber
    """the n-th tile of the sample"""

    tiles_in_sample: TotalNumberOfTiles
    """total number of tiles of the sample"""

    sample_sizes: PerTensor[PerAxis[int]]
    """the axis sizes of the sample"""

    stat: Stat
    """sample and dataset statistics"""

    outer_slice: PerTensor[PerAxis[SliceInfo]] = field(init=False)
    """slice of the outer tile (including overlap, but not padding) in the sample"""

    local_slice: PerTensor[PerAxis[SliceInfo]] = field(init=False)
    """slice to extract the inner tile from the outer tile"""

    overlap: PerTensor[PerAxis[LeftRight]] = field(init=False)
    """overlap 'into a neighboring tile'"""

    padding: PerTensor[PerAxis[PadWidth]] = field(init=False)
    """pad (at sample edges where we cannot overlap to realize `halo`"""

    def __post_init__(self):
        self.outer_slice = {
            t: {
                a: SliceInfo(
                    max(0, self.inner_slice[t][a].start - self.halo[t][a].left),
                    min(
                        self.sample_sizes[t][a],
                        self.inner_slice[t][a].stop + self.halo[t][a].right,
                    ),
                )
                for a in self.inner_slice[t]
            }
            for t in self.inner_slice
        }
        self.local_slice = {
            t: {
                a: SliceInfo(
                    self.inner_slice[t][a].start - self.outer_slice[t][a].start,
                    self.inner_slice[t][a].stop - self.outer_slice[t][a].start,
                )
                for a in self.inner_slice[t]
            }
            for t in self.inner_slice
        }
        self.overlap = {
            t: {
                a: LeftRight(
                    self.inner_slice[t][a].start - self.outer_slice[t][a].start,
                    self.outer_slice[t][a].stop - self.inner_slice[t][a].stop,
                )
                for a in self.inner_slice[t]
            }
            for t in self.inner_slice
        }
        self.padding = {
            t: {
                a: PadWidth(
                    self.halo[t][a].left - self.overlap[t][a].left,
                    self.halo[t][a].right - self.overlap[t][a].right,
                )
                for a in self.inner_slice[t]
            }
            for t in self.inner_slice
        }


@dataclass
class Tile(AbstractTile):
    """A tile of a dataset sample"""

    data: PerTensor[Tensor]
    """the tile's tensors"""

    @property
    def inner_data(self):
        return {t: self.data[t][self.local_slice[t]] for t in self.data}

    def __post_init__(self):
        super().__post_init__()
        for t, d in self.data.items():
            assert t == d.id, f"tensor id mismatch: {t} != {d.id}"
            for a, s in d.sizes.items():
                slice_ = self.inner_slice[t][a]
                halo = self.halo[t][a]
                assert s == slice_.stop - slice_.start + halo.left + halo.right, (
                    s,
                    slice_,
                    halo,
                )
