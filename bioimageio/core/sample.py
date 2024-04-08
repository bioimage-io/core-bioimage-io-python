from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generic, Iterable, Optional, Tuple, TypeVar, Union

import numpy as np
from typing_extensions import Self

from bioimageio.core.block import Block

from .axis import PerAxis
from .block_meta import (
    BlockMeta,
    LinearAxisTransform,
    split_multiple_shapes_into_blocks,
)
from .common import (
    BlockNumber,
    Halo,
    HaloLike,
    MemberId,
    PadMode,
    PerMember,
    SampleId,
    SliceInfo,
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
    ) -> Tuple[TotalNumberOfBlocks, Iterable[SampleBlockWithOrigin]]:
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
                    if -1 in block.sample_shape.values():
                        raise NotImplementedError(
                            "merging blocks with data dependent axis not yet implemented"
                        )

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

    sample_shape: PerMember[PerAxis[int]]
    """the sample shape this block represents a part of"""

    blocks: Dict[MemberId, BlockT]
    """Individual tensor blocks comprising this sample block"""

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
class LinearSampleAxisTransform(LinearAxisTransform):
    member: MemberId


@dataclass
class SampleBlockMeta(SampleBlockBase[BlockMeta]):
    """Meta data of a dataset sample block"""

    def get_transformed(
        self, new_axes: PerMember[PerAxis[Union[LinearSampleAxisTransform, int]]]
    ) -> Self:
        sample_shape = {
            m: {
                a: (
                    trf
                    if isinstance(trf, int)
                    else trf.compute(self.origin_shape[trf.member][trf.axis])
                )
                for a, trf in new_axes[m].items()
            }
            for m in new_axes
        }
        return self.__class__(
            blocks={
                m: BlockMeta(
                    sample_shape=sample_shape[m],
                    inner_slice={
                        a: (
                            SliceInfo(0, trf)
                            if isinstance(trf, int)
                            else SliceInfo(
                                trf.compute(
                                    self.blocks[trf.member].inner_slice[trf.axis].start
                                ),
                                trf.compute(
                                    self.blocks[trf.member].inner_slice[trf.axis].stop
                                ),
                            )
                        )
                        for a, trf in new_axes[m].items()
                    },
                    halo={
                        a: (
                            Halo(0, 0)
                            if isinstance(trf, int)
                            else Halo(
                                self.blocks[trf.member].halo[trf.axis].left,
                                self.blocks[trf.member].halo[trf.axis].right,
                            )
                        )
                        for a, trf in new_axes[m].items()
                    },
                    block_number=self.block_number,
                    blocks_in_sample=self.blocks_in_sample,
                )
                for m in new_axes
            },
            sample_shape=sample_shape,
        )

    def with_data(self, data: PerMember[Tensor], *, stat: Stat) -> SampleBlock:
        return SampleBlock(
            sample_shape=self.sample_shape,
            blocks={
                m: Block(
                    data[m],
                    inner_slice=b.inner_slice,
                    halo=b.halo,
                    block_number=b.block_number,
                    blocks_in_sample=b.blocks_in_sample,
                )
                for m, b in self.blocks.items()
            },
            stat=stat,
        )


@dataclass
class SampleBlock(SampleBlockBase[Block]):
    """A block of a dataset sample"""

    stat: Stat
    """computed statistics"""

    @property
    def members(self) -> PerMember[Tensor]:
        """the sample block's tensors"""
        return {m: b.data for m, b in self.blocks.items()}

    def get_transformed_meta(
        self, new_axes: PerMember[PerAxis[Union[LinearSampleAxisTransform, int]]]
    ) -> SampleBlockMeta:
        return SampleBlockMeta(
            blocks=dict(self.blocks), sample_shape=self.sample_shape
        ).get_transformed(new_axes)


@dataclass
class SampleBlockWithOrigin(SampleBlock):
    origin: Sample
    """the sample this sample black was taken from"""


def sample_block_meta_generator(
    blocks: Iterable[PerMember[BlockMeta]],
    *,
    sample_shape: PerMember[PerAxis[int]],
):
    for member_blocks in blocks:
        yield SampleBlockMeta(
            blocks=dict(member_blocks),
            sample_shape=sample_shape,
        )


def sample_block_generator(
    blocks: Iterable[PerMember[BlockMeta]],
    *,
    origin: Sample,
    pad_mode: PadMode,
):
    for member_blocks in blocks:
        yield SampleBlockWithOrigin(
            blocks={
                m: Block.from_sample_member(
                    origin.members[m], block=member_blocks[m], pad_mode=pad_mode
                )
                for m in origin.members
            },
            sample_shape=origin.shape,
            origin=origin,
            stat=origin.stat,
        )
