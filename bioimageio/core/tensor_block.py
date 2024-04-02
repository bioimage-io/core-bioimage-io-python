from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional, Tuple

from typing_extensions import Self

from bioimageio.core.axis import PerAxis
from bioimageio.core.common import (
    Halo,
    HaloLike,
    PadMode,
    SliceInfo,
    TotalNumberOfBlocks,
)

from .block import Block, split_shape_into_blocks
from .stat_measures import Stat
from .tensor import Tensor


@dataclass(init=False)
class TensorBlock(Block):
    """A block with data"""

    stat: Stat
    """sample and dataset statistics"""

    data: Tensor
    """the block's tensor"""

    def __init__(
        self,
        data: Tensor,
        *,
        inner_slice: PerAxis[SliceInfo],
        halo: PerAxis[Halo],
        block_number: int,
        blocks_in_sample: int,
        stat: Stat,
    ):
        super().__init__(
            sample_shape=data.tagged_shape,
            inner_slice=inner_slice,
            halo=halo,
            block_number=block_number,
            blocks_in_sample=blocks_in_sample,
        )
        self.data = data
        self.stat = stat

    @property
    def inner_data(self):
        return {t: self.data[self.local_slice] for t in self.data}

    def __post_init__(self):
        super().__post_init__()
        for a, s in self.data.sizes.items():
            slice_ = self.inner_slice[a]
            halo = self.halo[a]
            assert s == slice_.stop - slice_.start + halo.left + halo.right, (
                s,
                slice_,
                halo,
            )

    @classmethod
    def from_sample(
        cls,
        sample: Tensor,
        block: Block,
        *,
        pad_mode: PadMode,
        stat: Stat,
    ) -> Self:
        return cls(
            data=sample[block.outer_slice].pad(block.padding, pad_mode),
            inner_slice=block.inner_slice,
            halo=block.halo,
            block_number=block.block_number,
            blocks_in_sample=block.blocks_in_sample,
            stat=stat,
        )


def split_tensor_into_blocks(
    sample: Tensor,
    block_shape: PerAxis[int],
    *,
    halo: PerAxis[HaloLike],
    stride: Optional[PerAxis[int]] = None,
    pad_mode: PadMode,
    stat: Stat,
) -> Tuple[TotalNumberOfBlocks, Generator[TensorBlock, Any, None]]:
    """divide a sample tensor into tensor blocks."""
    n_blocks, block_gen = split_shape_into_blocks(
        sample.tagged_shape, block_shape=block_shape, halo=halo
    )
    return n_blocks, _tensor_block_generator(
        sample, block_gen, pad_mode=pad_mode, stat=stat
    )


def _tensor_block_generator(
    sample: Tensor, blocks: Iterable[Block], *, pad_mode: PadMode, stat: Stat
):
    for block in blocks:
        yield TensorBlock.from_sample(sample, block, pad_mode=pad_mode, stat=stat)
