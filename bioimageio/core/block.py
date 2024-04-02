import itertools
from dataclasses import dataclass, field
from math import prod
from typing import Any, Dict, Generator, List, Optional, Tuple

from .axis import AxisId, PerAxis
from .common import (
    BlockNumber,
    Halo,
    HaloLike,
    PadWidth,
    SliceInfo,
    TotalNumberOfBlocks,
)


@dataclass
class Block:
    """Block of a sample

    Figure for illustration:
    The first 2d block (dashed) of a sample (**bold**).
    The inner slice (thin) is expanded by a halo in both dimensions on both sides.
    The outer slice reaches from the sample origin (0, 0) to the right halo point.

    ```terminal
    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ─ ─ ─ ─ ─ ─ ─ ┐
    ╷ halo(left)                         ╷
    ╷                                    ╷
    ╷  (0, 0)┏━━━━━━━━━━━━━━━━━┯━━━━━━━━━┯━━━➔
    ╷        ┃                 │         ╷  sample
    ╷        ┃      inner      │         ╷
    ╷        ┃   (and outer)   │  outer  ╷
    ╷        ┃      slice      │  slice  ╷
    ╷        ┃                 │         ╷
    ╷        ┣─────────────────┘         ╷
    ╷        ┃   outer slice             ╷
    ╷        ┃               halo(right) ╷
    └ ─ ─ ─ ─┃─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
             ⬇
    ```

    note:
    - Inner and outer slices are specified in sample coordinates.
    - The outer_slice of a block at the sample edge may overlap by more than the
        halo with the neighboring block (the inner slices will not overlap though).

    """

    sample_shape: PerAxis[int]
    """the axis sizes of the whole (unblocked) sample"""

    inner_slice: PerAxis[SliceInfo]
    """inner region (without halo) wrt the sample"""

    halo: PerAxis[Halo]
    """halo enlarging the inner region to the block's sizes"""

    block_number: BlockNumber
    """the n-th block of the sample"""

    blocks_in_sample: TotalNumberOfBlocks
    """total number of blocks in the sample"""

    shape: PerAxis[int] = field(init=False)
    """axis lengths of the block"""

    padding: PerAxis[PadWidth] = field(init=False)
    """padding to realize the halo at the sample edge
    where we cannot simply enlarge the inner slice"""

    outer_slice: PerAxis[SliceInfo] = field(init=False)
    """slice of the outer block (without padding) wrt the sample"""

    inner_shape: PerAxis[int] = field(init=False)
    """axis lengths of the inner region (without halo)"""

    local_slice: PerAxis[SliceInfo] = field(init=False)
    """inner slice wrt the block, **not** the sample"""

    def __post_init__(self):
        assert all(
            a in self.sample_shape for a in self.inner_slice
        ), "block has axes not present in sample"
        assert all(
            a in self.inner_slice for a in self.halo
        ), "halo has axes not present in block"

        self.shape = {
            a: s.stop - s.start + sum(self.halo[a]) for a, s in self.inner_slice.items()
        }
        assert all(
            s <= self.sample_shape[a] for a, s in self.shape.items()
        ), "block larger than sample"

        self.inner_shape = {a: s.stop - s.start for a, s in self.inner_slice.items()}
        self.outer_slice = {
            a: SliceInfo(
                max(
                    0,
                    min(
                        self.inner_slice[a].start - self.halo[a].left,
                        self.sample_shape[a] - self.inner_shape[a] - self.halo[a].left,
                    ),
                ),
                min(
                    self.sample_shape[a],
                    self.inner_slice[a].stop + self.halo[a].right,
                ),
            )
            for a in self.inner_slice
        }
        self.padding = {
            a: PadWidth(
                max(
                    0,
                    self.halo[a].left
                    - (self.inner_slice[a].start + self.outer_slice[a].start),
                ),
                max(
                    0,
                    self.halo[a].right
                    - (self.outer_slice[a].stop + self.inner_slice[a].stop),
                ),
            )
            for a in self.inner_slice
        }
        self.local_slice = {
            a: SliceInfo(
                self.padding[a].left,
                self.padding[a].left + self.inner_shape[a],
            )
            for a in self.inner_slice
        }


def split_shape_into_blocks(
    shape: PerAxis[int],
    block_shape: PerAxis[int],
    halo: PerAxis[HaloLike],
    stride: Optional[PerAxis[int]] = None,
) -> Tuple[TotalNumberOfBlocks, Generator[Block, Any, None]]:
    assert all(a in shape for a in block_shape), (
        tuple(shape),
        set(block_shape),
    )
    assert all(a in shape for a in halo), (tuple(shape), set(halo))

    # fill in default halo (0) and tile_size (tensor size)
    halo = {a: Halo.create(h) for a, h in halo.items()}
    block_shape = {a: block_shape.get(a, s) for a, s in shape.items()}
    if stride is None:
        stride = {}

    inner_1d_slices: Dict[AxisId, List[SliceInfo]] = {}
    for a, s in shape.items():
        inner_size = block_shape[a] - sum(halo[a])
        stride_1d = stride.get(a, inner_size)
        inner_1d_slices[a] = [
            SliceInfo(min(p, s - inner_size), min(p + inner_size, s))
            for p in range(0, s, stride_1d)
        ]

    n_blocks = prod(map(len, inner_1d_slices.values()))

    return n_blocks, _block_generator(
        shape,
        blocks_in_sample=n_blocks,
        inner_1d_slices=inner_1d_slices,
        halo=halo,
    )


def _block_generator(
    sample_shape: PerAxis[int],
    *,
    blocks_in_sample: int,
    inner_1d_slices: Dict[AxisId, List[SliceInfo]],
    halo: PerAxis[HaloLike],
):
    assert all(a in sample_shape for a in halo)

    halo = {a: Halo.create(halo.get(a, 0)) for a in inner_1d_slices}
    for i, nd_tile in enumerate(itertools.product(*inner_1d_slices.values())):
        inner_slice: PerAxis[SliceInfo] = dict(zip(inner_1d_slices, nd_tile))

        yield Block(
            sample_shape=sample_shape,
            inner_slice=inner_slice,
            halo=halo,
            block_number=i,
            blocks_in_sample=blocks_in_sample,
        )
