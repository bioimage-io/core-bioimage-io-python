from dataclasses import dataclass
from typing import (
    Any,
    Generator,
    Iterable,
    Optional,
    Tuple,
    Union,
)

from typing_extensions import Self

from .axis import PerAxis
from .block_meta import BlockMeta, LinearAxisTransform, split_shape_into_blocks
from .common import (
    Halo,
    HaloLike,
    PadMode,
    TotalNumberOfBlocks,
)
from .tensor import Tensor


@dataclass(init=False, frozen=True)
class Block(BlockMeta):
    """A block/tile of a (larger) tensor"""

    data: Tensor
    """the block's tensor, e.g. a (padded) slice of some larger, original tensor"""

    def __init__(
        self,
        data: Tensor,
        *,
        inner_slice: PerAxis[SliceInfo],
        halo: PerAxis[Halo],
        block_index: int,
        blocks_in_sample: int,
    ):
        object.__setattr__(self, "data", data)
        super().__init__(
            sample_shape=data.tagged_shape,
            inner_slice=inner_slice,
            halo=halo,
            block_index=block_index,
            blocks_in_sample=blocks_in_sample,
        )

    @property
    def inner_data(self):
        return self.data[self.local_slice]

    def __post_init__(self):
        super().__post_init__()
        for a, s in self.data.sizes.items():
            slice_ = self.inner_slice[a]
            halo = self.halo.get(a, Halo(0, 0))
            assert s == halo.left + (slice_.stop - slice_.start) + halo.right, (
                s,
                slice_,
                halo,
            )

    @classmethod
    def from_sample_member(
        cls,
        sample_member: Tensor,
        block: BlockMeta,
        *,
        pad_mode: PadMode,
    ) -> Self:
        return cls(
            data=sample_member[block.outer_slice].pad(block.padding, pad_mode),
            inner_slice=block.inner_slice,
            halo=block.halo,
            block_index=block.block_index,
            blocks_in_sample=block.blocks_in_sample,
        )

    def get_transformed(
        self, new_axes: PerAxis[Union[LinearAxisTransform, int]]
    ) -> Self:
        raise NotImplementedError


def split_tensor_into_blocks(
    tensor: Tensor,
    block_shape: PerAxis[int],
    *,
    halo: PerAxis[HaloLike],
    stride: Optional[PerAxis[int]] = None,
    pad_mode: PadMode,
) -> Tuple[TotalNumberOfBlocks, Generator[Block, Any, None]]:
    """divide a sample tensor into tensor blocks."""
    n_blocks, block_gen = split_shape_into_blocks(
        tensor.tagged_shape, block_shape=block_shape, halo=halo, stride=stride
    )
    return n_blocks, _block_generator(tensor, block_gen, pad_mode=pad_mode)


def _block_generator(sample: Tensor, blocks: Iterable[BlockMeta], *, pad_mode: PadMode):
    for block in blocks:
        yield Block.from_sample_member(sample, block, pad_mode=pad_mode)
