from __future__ import annotations

from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import (
    Any,
)

from typing_extensions import Self

from ._multi_index import MultiIndex
from .axis import BATCH_AXIS_ID, AxisId, PerAxis
from .block_meta import BlockMeta, LinearAxisTransform, split_shape_into_blocks
from .common import (
    Halo,
    HaloLike,
    PadMode,
    TotalNumberOfBlocks,
)
from .tensor import Tensor


@dataclass(frozen=True)
class Block(BlockMeta):
    """A block/tile of a (larger) tensor"""

    data: Tensor
    """the block's tensor, e.g. a (padded) slice of some larger, original tensor"""

    @property
    def inner_data(self):
        return self.data[self.local_slice]

    def __post_init__(self):
        super().__post_init__()
        assert not any(v == -1 for v in self.sample_shape.values()), self.sample_shape
        for a, s in self.data.sizes.items():
            slice_ = self.local_slice[a]
            halo = self.halo.get(a, Halo(0, 0))
            if s != (s_exp := halo.left + (slice_.stop - slice_.start) + halo.right):
                raise ValueError(
                    f"Block data shape does not match block meta information for axis {a}: data size is {s} but expected {s_exp} (halo: {halo}, slice: {slice_})",
                )

    @classmethod
    def from_sample_member(
        cls,
        sample_member: Tensor,
        block: BlockMeta,
        *,
        pad_mode: PadMode,
    ) -> Self:
        block_data = sample_member[block.outer_slice]
        padded_block_data = block_data.pad(block.padding, pad_mode)

        if block.unstacked_batch_slice is not None:
            if not isinstance(
                idx := block_data.data.indexes.get(BATCH_AXIS_ID),  # pyright: ignore[reportUnknownVariableType]
                MultiIndex,
            ):
                raise ValueError("Expected a MultiIndex for the batch axis")

            padded_block_data = padded_block_data.assign_batch_multi_index(idx)
            assert block.unstacked_batch_shape is not None
            assert block.inner_slice[BATCH_AXIS_ID] == block.outer_slice[BATCH_AXIS_ID]

        return cls(
            data=padded_block_data,
            sample_shape=sample_member.tagged_shape,
            unstacked_batch_shape=block.unstacked_batch_shape,
            unstacked_dim_order=block.unstacked_dim_order,
            inner_slice=block.inner_slice,
            unstacked_batch_slice=block.unstacked_batch_slice,
            halo=block.halo,
            block_index=block.block_index,
            blocks_in_sample=block.blocks_in_sample,
        )

    def get_transformed(self, new_axes: PerAxis[LinearAxisTransform | int]) -> Self:
        raise NotImplementedError

    @classmethod
    def from_meta(cls, meta: BlockMeta, data: Tensor) -> Self:
        return cls(
            sample_shape=meta.sample_shape,
            unstacked_batch_shape=meta.unstacked_batch_shape,
            unstacked_dim_order=meta.unstacked_dim_order,
            inner_slice=meta.inner_slice,
            unstacked_batch_slice=meta.unstacked_batch_slice,
            halo=meta.halo,
            block_index=meta.block_index,
            blocks_in_sample=meta.blocks_in_sample,
            data=data,
        )

    def get_meta(self) -> BlockMeta:
        return BlockMeta(
            sample_shape=self.sample_shape,
            unstacked_batch_shape=self.unstacked_batch_shape,
            unstacked_dim_order=self.unstacked_dim_order,
            inner_slice=self.inner_slice,
            unstacked_batch_slice=self.unstacked_batch_slice,
            halo=self.halo,
            block_index=self.block_index,
            blocks_in_sample=self.blocks_in_sample,
        )


def split_tensor_into_blocks(
    tensor: Tensor,
    block_shape: PerAxis[int],
    *,
    halo: PerAxis[HaloLike],
    stride: PerAxis[int] | None = None,
    pad_mode: PadMode,
) -> tuple[TotalNumberOfBlocks, Generator[Block, Any, None]]:
    """divide a sample tensor into tensor blocks."""
    if (idx := tensor.batch_multi_index) is None:
        unstacked_batch_shape = None
    else:
        unstacked_batch_shape = {
            BATCH_AXIS_ID if a == "original_batch" else AxisId(a): s
            for a, s in zip(idx.names, idx.levshape)
        }

    n_blocks, block_gen = split_shape_into_blocks(
        tensor.tagged_shape,
        unstacked_dim_order=None,
        unstacked_batch_shape=unstacked_batch_shape,
        block_shape=block_shape,
        halo=halo,
        stride=stride,
    )
    return n_blocks, _block_generator(tensor, block_gen, pad_mode=pad_mode)


def _block_generator(sample: Tensor, blocks: Iterable[BlockMeta], *, pad_mode: PadMode):
    for block in blocks:
        yield Block.from_sample_member(sample, block, pad_mode=pad_mode)
