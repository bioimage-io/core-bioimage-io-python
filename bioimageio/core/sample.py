from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generic, Iterable, Optional, Tuple, TypeVar

import numpy as np
from typing_extensions import Self

from bioimageio.core.block import Block

from .axis import PerAxis
from .block_meta import BlockMeta, split_multiple_shapes_into_blocks
from .common import (
    BlockNumber,
    HaloLike,
    MemberId,
    PadMode,
    PerMember,
    SampleId,
    TotalNumberOfBlocks,
)
from .stat_measures import Stat
from .tensor import Tensor

# TODO: allow for lazy samples to read/write to disk


@dataclass
class Sample:
    """A dataset sample"""

    members: Dict[MemberId, Tensor]
    """the sample's tensors"""

    stat: Stat = field(default_factory=dict)
    """sample and dataset statistics"""

    id: Optional[SampleId] = None
    """identifier within the sample's dataset"""

    @property
    def shape(self) -> PerMember[PerAxis[int]]:
        return {tid: t.sizes for tid, t in self.members.items()}

    def split_into_blocks(
        self,
        block_shapes: PerMember[PerAxis[int]],
        halo: PerMember[PerAxis[HaloLike]],
        pad_mode: PadMode,
        broadcast: bool = False,
    ) -> Tuple[TotalNumberOfBlocks, Iterable[SampleBlock]]:
        assert not (
            missing := [m for m in block_shapes if m not in self.members]
        ), f"`block_shapes` specified for unknown members: {missing}"
        assert not (
            missing := [m for m in halo if m not in block_shapes]
        ), f"`halo` specified for members without `block_shape`: {missing}"

        n_blocks, blocks = split_multiple_shapes_into_blocks(
            shapes=self.shape,
            block_shapes=block_shapes,
            halo=halo,
            broadcast=broadcast,
        )
        return n_blocks, sample_block_generator(blocks, origin=self, pad_mode=pad_mode)

    @classmethod
    def from_blocks(
        cls,
        sample_blocks: Iterable[SampleBlock],
        *,
        fill_value: float = float("nan"),
    ) -> Self:
        members: PerMember[Tensor] = {}
        for member_blocks in sample_blocks:
            for m, block in member_blocks.blocks.items():
                if m not in members:
                    members[m] = Tensor(
                        np.full(
                            tuple(block.sample_shape[a] for a in block.data.dims),
                            fill_value,
                            dtype=block.data.dtype,
                        ),
                        dims=block.data.dims,
                    )

                members[m][block.inner_slice] = block.inner_data

        return cls(members=members)


BlockT = TypeVar("BlockT", Block, BlockMeta)


@dataclass
class SampleBlockBase(Generic[BlockT]):
    """base class for `SampleBlockMeta` and `SampleBlock`"""

    blocks: Dict[MemberId, BlockT]

    block_number: BlockNumber = field(init=False)
    """the n-th block of the sample"""

    blocks_in_sample: TotalNumberOfBlocks = field(init=False)
    """total number of blocks in the sample"""

    def __post_init__(self):
        a_block = next(iter(self.blocks.values()))
        self.block_number = a_block.block_number
        self.blocks_in_sample = a_block.blocks_in_sample

    @property
    def shape(self) -> PerMember[PerAxis[int]]:
        return {mid: b.shape for mid, b in self.blocks.items()}

    @property
    def inner_shape(self) -> PerMember[PerAxis[int]]:
        return {mid: b.inner_shape for mid, b in self.blocks.items()}

    @property
    @abstractmethod
    def origin_shape(self) -> PerMember[PerAxis[int]]: ...


@dataclass
class SampleBlockMeta(SampleBlockBase[BlockMeta]):
    """Meta data of a dataset sample block"""

    origin: PerMember[PerAxis[int]]
    """the sampe shape the blocking for this block was based on"""

    @property
    def origin_shape(self):
        return self.origin


@dataclass
class SampleBlock(SampleBlockBase[Block]):
    """A block of a dataset sample"""

    origin: Sample
    """the sample this sample black was taken from"""

    @property
    def origin_shape(self):
        return self.origin.shape

    @property
    def members(self) -> PerMember[Tensor]:
        """the sample block's tensors"""
        return {m: b.data for m, b in self.blocks.items()}

    @property
    def stat(self):
        return self.origin.stat


def sample_block_meta_generator(
    blocks: Iterable[PerMember[BlockMeta]],
    *,
    origin: PerMember[PerAxis[int]],
):
    for member_blocks in blocks:
        yield SampleBlockMeta(
            blocks=dict(member_blocks),
            origin=origin,
        )


def sample_block_generator(
    blocks: Iterable[PerMember[BlockMeta]],
    *,
    origin: Sample,
    pad_mode: PadMode,
):
    for member_blocks in blocks:
        yield SampleBlock(
            blocks={
                m: Block.from_sample_member(
                    origin.members[m], block=member_blocks[m], pad_mode=pad_mode
                )
                for m in origin.members
            },
            origin=origin,
        )
