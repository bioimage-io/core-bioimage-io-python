from __future__ import annotations

import collections.abc
from dataclasses import dataclass
from math import ceil, floor
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pydantic
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import Self

from ._common_annotations import PerMemberAnno
from .axis import AxisId, PerAxis
from .block import Block
from .block_meta import (
    BlockMeta,
    LinearAxisTransform,
    split_multiple_shapes_into_blocks,
)
from .common import (
    BlockIndex,
    Halo,
    HaloLike,
    MemberId,
    PadMode,
    PadWidthLike,
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
    """A dataset sample.

    A `Sample` has `members`, which allows to combine multiple tensors into a single
    sample.
    For example a `Sample` from a dataset with masked images may contain a
    `MemberId("raw")` and `MemberId("mask")` image.
    """

    members: Dict[MemberId, Tensor]
    """The sample's tensors"""

    stat: Stat
    """Sample and dataset statistics"""

    id: SampleId
    """Identifies the `Sample` within the dataset -- typically a number or a string."""

    def __getitem__(
        self,
        key: PerMember[
            Union[
                SliceInfo,
                slice,
                int,
                PerAxis[Union[SliceInfo, slice, int]],
                Tensor,
                xr.DataArray,
            ]
        ],
    ) -> Self:
        return self.__class__(
            members={m: t[key[m]] for m, t in self.members.items() if m in key},
            stat=self.stat,
            id=self.id,
        )

    def set_block(self, block: SampleBlock) -> None:
        """Set values of `block`.

        Note:
            - Updates only existing sample members (extra block members are ignored)
            - Ignores missing block members (i.e. members in the sample but not in the block are not modified)

        Raises:
            ValueError if block and sample members do not overlap at all.
        """
        no_overlap = True
        for m in self.members:
            if m not in block.blocks:
                continue
            b = block.blocks[m]
            self.members[m][b.inner_slice] = b.inner_data
            no_overlap = False

        if no_overlap:
            raise ValueError(
                f"block with members {list(block.blocks)} does not overlap with sample members {list(self.members)}"
            )

    @property
    def shape(self) -> PerMember[PerAxis[int]]:
        return {tid: t.sizes for tid, t in self.members.items()}

    def as_arrays(self) -> Dict[MemberId, NDArray[Any]]:
        """Return sample as dictionary of arrays."""
        return {m: t.to_numpy() for m, t in self.members.items()}

    def split_into_blocks(
        self,
        block_shapes: PerMember[PerAxis[int]],
        halo: PerMember[PerAxis[HaloLike]],
        pad_mode: Union[PadMode, PerMember[PadMode]],
        broadcast: bool = False,
    ) -> Tuple[TotalNumberOfBlocks, Iterable[SampleBlockWithOrigin]]:
        assert not (missing := [m for m in block_shapes if m not in self.members]), (
            f"`block_shapes` specified for unknown members: {missing}"
        )
        assert not (missing := [m for m in halo if m not in block_shapes]), (
            f"`halo` specified for members without `block_shape`: {missing}"
        )

        n_blocks, blocks = split_multiple_shapes_into_blocks(
            shapes=self.shape,
            block_shapes=block_shapes,
            halo=halo,
            broadcast=broadcast,
        )
        return n_blocks, sample_block_generator(blocks, origin=self, pad_mode=pad_mode)

    def as_single_block(self, halo: Optional[PerMember[PerAxis[Halo]]] = None):
        if halo is None:
            halo = {}
        return SampleBlockWithOrigin(
            sample_shape=self.shape,
            sample_id=self.id,
            blocks={
                m: Block(
                    sample_shape=self.shape[m],
                    data=data,
                    inner_slice={
                        a: SliceInfo(0, s) for a, s in data.tagged_shape.items()
                    },
                    halo=halo.get(m, {}),
                    block_index=0,
                    blocks_in_sample=1,
                )
                for m, data in self.members.items()
            },
            stat=self.stat,
            origin=self,
            block_index=0,
            blocks_in_sample=1,
        )

    @classmethod
    def from_blocks(
        cls,
        sample_blocks: Iterable[SampleBlock],
        *,
        fill_value: float = float("nan"),
    ) -> Self:
        """Create a `Sample` from an iterable of `SampleBlock`s.

        Note:
            All sample blocks must have the same `sample_id`.

        Args:
            sample_blocks: The blocks to create the sample from.
            fill_value: The value to fill missing values with (default: `nan`).
        """
        output = None
        for output in cls.from_blocks_yield_intermediates(
            sample_blocks, fill_value=fill_value
        ):
            pass

        if output is None:
            raise ValueError("no sample blocks provided")

        return output

    @classmethod
    def from_blocks_yield_intermediates(
        cls,
        sample_blocks: Iterable[SampleBlock],
        *,
        fill_value: float = float("nan"),
    ):
        """Create a `Sample` from an iterable of `SampleBlock`s, yielding the intermediate sample after each block.

        Args:
            sample_blocks: The blocks to create the sample from.
            fill_value: The value to fill missing values with (default: `nan`).
        """
        output = cls(members={}, stat={}, id=None)
        for sample_block in sample_blocks:
            if output.id is None:
                output.sample_id = sample_block.sample_id
            else:
                assert output.id == sample_block.sample_id, (
                    "sample id changed between sample blocks"
                )

            output.stat = sample_block.stat

            for m, block in sample_block.blocks.items():
                if m not in output.members:
                    if -1 in block.sample_shape.values():
                        raise NotImplementedError(
                            "merging blocks with data dependent axis not yet implemented"
                        )

                    output.members[m] = Tensor(
                        np.full(
                            tuple(block.sample_shape[a] for a in block.data.dims),
                            fill_value,
                            dtype=block.data.dtype,
                        ),
                        dims=block.data.dims,
                    )

                output.members[m][block.inner_slice] = block.inner_data
            yield output

        yield output

    def pad(
        self,
        pad_width: PerMember[PerAxis[Union[int, PadWidthLike]]],
        mode: Union[PerMember[PadMode], PadMode],
    ) -> Self:
        """Convenience method to pad sample members."""
        default_mode = "symmetric"
        if isinstance(mode, collections.abc.Mapping):
            mode_per_member = mode
        else:
            mode_per_member: Mapping[MemberId, PadMode] = {}
            default_mode = mode

        return self.__class__(
            members={
                m: t.pad(
                    pad_width=pad_width.get(m, {}),
                    mode=mode_per_member.get(m, default_mode),
                )
                for m, t in self.members.items()
            },
            stat=self.stat,
            id=self.id,
        )


BlockT = TypeVar("BlockT", bound=BlockMeta)


@pydantic.dataclasses.dataclass(frozen=True)
class SampleBlockBase(Generic[BlockT]):
    """base class for `SampleBlockMeta` and `SampleBlock`"""

    sample_shape: PerMemberAnno[PerAxis[int]]
    """the sample shape this block represents a part of"""

    sample_id: SampleId
    """identifier for the sample within its dataset"""

    blocks: PerMemberAnno[BlockT]
    """Individual tensor blocks comprising this sample block"""

    block_index: BlockIndex
    """the n-th block of the sample"""

    blocks_in_sample: TotalNumberOfBlocks
    """total number of blocks in the sample"""

    @property
    def shape(self) -> PerMember[PerAxis[int]]:
        return MappingProxyType({mid: b.shape for mid, b in self.blocks.items()})

    @property
    def inner_shape(self) -> PerMember[PerAxis[int]]:
        return MappingProxyType({mid: b.inner_shape for mid, b in self.blocks.items()})


@dataclass
class LinearSampleAxisTransform(LinearAxisTransform):
    member: MemberId


@pydantic.dataclasses.dataclass(frozen=True)
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
                    else trf.compute(self.sample_shape[trf.member][trf.axis])
                )
                for a, trf in new_axes[m].items()
            }
            for m in new_axes
        }

        def get_member_halo(m: MemberId, round: Callable[[float], int]):
            return {
                a: (
                    Halo(0, 0)
                    if isinstance(trf, int)
                    or trf.axis not in self.blocks[trf.member].halo
                    else Halo(
                        round(self.blocks[trf.member].halo[trf.axis].left * trf.scale),
                        round(self.blocks[trf.member].halo[trf.axis].right * trf.scale),
                    )
                )
                for a, trf in new_axes[m].items()
            }

        halo: Dict[MemberId, Dict[AxisId, Halo]] = {}
        for m in new_axes:
            halo[m] = get_member_halo(m, floor)
            if halo[m] != get_member_halo(m, ceil):
                raise ValueError(
                    f"failed to unambiguously scale halo {halo[m]} with {new_axes[m]}"
                    + f" for {m}."
                )

        inner_slice = {
            m: {
                a: (
                    SliceInfo(0, trf)
                    if isinstance(trf, int)
                    else SliceInfo(
                        trf.compute(
                            self.blocks[trf.member].inner_slice[trf.axis].start
                        ),
                        trf.compute(self.blocks[trf.member].inner_slice[trf.axis].stop),
                    )
                )
                for a, trf in new_axes[m].items()
            }
            for m in new_axes
        }
        return self.__class__(
            blocks={
                m: BlockMeta(
                    sample_shape=sample_shape[m],
                    inner_slice=inner_slice[m],
                    halo=halo[m],
                    block_index=self.block_index,
                    blocks_in_sample=self.blocks_in_sample,
                )
                for m in new_axes
            },
            sample_shape=sample_shape,
            sample_id=self.sample_id,
            block_index=self.block_index,
            blocks_in_sample=self.blocks_in_sample,
        )

    def with_data(self, data: PerMember[Tensor], *, stat: Stat) -> SampleBlock:
        return SampleBlock(
            sample_shape={
                m: {
                    a: data[m].tagged_shape[a] if s == -1 else s
                    for a, s in member_shape.items()
                }
                for m, member_shape in self.sample_shape.items()
            },
            sample_id=self.sample_id,
            blocks={
                m: Block.from_meta(b, data=data[m]) for m, b in self.blocks.items()
            },
            stat=stat,
            block_index=self.block_index,
            blocks_in_sample=self.blocks_in_sample,
        )


@dataclass(frozen=True)
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
            sample_id=self.sample_id,
            blocks=dict(self.blocks),
            sample_shape=self.sample_shape,
            block_index=self.block_index,
            blocks_in_sample=self.blocks_in_sample,
        ).get_transformed(new_axes)

    @classmethod
    def from_meta(
        cls, meta: SampleBlockMeta, data: PerMember[Tensor], stat: Stat
    ) -> Self:
        return cls(
            sample_shape=meta.sample_shape,
            sample_id=meta.sample_id,
            blocks={
                m: Block.from_meta(b, data=data[m]) for m, b in meta.blocks.items()
            },
            stat=stat,
            block_index=meta.block_index,
            blocks_in_sample=meta.blocks_in_sample,
        )

    def get_meta(self) -> SampleBlockMeta:
        return SampleBlockMeta(
            sample_id=self.sample_id,
            blocks={m: b.get_meta() for m, b in self.blocks.items()},
            sample_shape=self.sample_shape,
            block_index=self.block_index,
            blocks_in_sample=self.blocks_in_sample,
        )


@dataclass(frozen=True)
class SampleBlockWithOrigin(SampleBlock):
    """A `SampleBlock` with a reference (`origin`) to the whole `Sample`"""

    origin: Sample
    """the sample this sample block was taken from"""


class _ConsolidatedMemberBlocks:
    def __init__(self, blocks: PerMember[BlockMeta]):
        super().__init__()
        block_indices = {b.block_index for b in blocks.values()}
        assert len(block_indices) == 1
        self.block_index = block_indices.pop()
        blocks_in_samples = {b.blocks_in_sample for b in blocks.values()}
        assert len(blocks_in_samples) == 1
        self.blocks_in_sample = blocks_in_samples.pop()


def sample_block_meta_generator(
    blocks: Iterable[PerMember[BlockMeta]],
    *,
    sample_shape: PerMember[PerAxis[int]],
    sample_id: SampleId,
):
    for member_blocks in blocks:
        cons = _ConsolidatedMemberBlocks(member_blocks)
        yield SampleBlockMeta(
            blocks=dict(member_blocks),
            sample_shape=sample_shape,
            sample_id=sample_id,
            block_index=cons.block_index,
            blocks_in_sample=cons.blocks_in_sample,
        )


def sample_block_generator(
    blocks: Iterable[PerMember[BlockMeta]],
    *,
    origin: Sample,
    pad_mode: Union[PadMode, PerMember[PadMode]],
) -> Iterable[SampleBlockWithOrigin]:
    for member_blocks in blocks:
        cons = _ConsolidatedMemberBlocks(member_blocks)
        yield SampleBlockWithOrigin(
            blocks={
                m: Block.from_sample_member(
                    origin.members[m],
                    block=member_blocks[m],
                    pad_mode=pad_mode.get(m, "symmetric")
                    if isinstance(pad_mode, collections.abc.Mapping)
                    else pad_mode,
                )
                for m in origin.members
            },
            sample_shape=origin.shape,
            origin=origin,
            stat=origin.stat,
            sample_id=origin.id,
            block_index=cons.block_index,
            blocks_in_sample=cons.blocks_in_sample,
        )
