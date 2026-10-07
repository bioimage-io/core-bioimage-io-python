from __future__ import annotations

import collections.abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
)

import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import Self, assert_never

from ._utils.resolve_per_member_key import resolve_per_member_var
from .axis import AxisId, PerAxis
from .common import (
    MemberId,
    PadMode,
    PadWidthLike,
    PerMember,
    SampleId,
    SliceInfo,
)
from .stat_measures import Stat
from .tensor import Tensor


@dataclass
class Sample:
    """A dataset sample.

    A `Sample` has `members`, which allows to combine multiple tensors into a single
    sample.
    For example a `Sample` from a dataset with masked images may contain a
    `MemberId("raw")` and `MemberId("mask")` image.
    """

    id: SampleId
    """Identifies the `Sample` within the dataset -- typically a number or a string."""

    members: dict[MemberId, Tensor]
    """The sample's tensors"""

    stat: Stat
    """Sample and dataset statistics"""

    # unstacked_dim_order: PerMember[Sequence[AxisId] | None] | None = None
    # """The order of the fully unstacked tensor"""

    def __getitem__(
        self,
        key: PerMember[PerAxis[SliceInfo | slice | int] | Tensor | xr.DataArray]
        | PerAxis[SliceInfo | slice | int]
        | Tensor
        | xr.DataArray,
    ) -> Self:
        per_member_key = resolve_per_member_var(
            key,
            valid_per_member_types=(
                SliceInfo,
                slice,
                int,
                Tensor,
                xr.DataArray,
            ),
            valid_per_axis_types=(SliceInfo, slice, int),
            members=self.members,
        )

        return self.__class__(
            members={
                m: t[per_member_key[m]]
                for m, t in self.members.items()
                if m in per_member_key
            },
            stat=self.stat,
            id=self.id,
        )

    def __postinit__(self):
        # default tensor names to member ids
        for m, t in self.members.items():
            if t.name is None:
                t.name = str(m)

    # def chunk(
    #     self,
    #     chunk_shapes: PerMember[PerAxis[int]],
    #     *,
    #     halo: PerMember[PerAxis[HaloLike]] | None = None,
    #     pad_mode: PadMode | PerMember[PadMode] = "symmetric",
    # ) -> Self:
    #     """Return a new sample with chunked sample members.

    #     Args:
    #         chunk_shapes: The chunk shapes per member and axis.
    #     """
    #     return self.__class__(
    #         members=chunk_tensors(
    #             self.members, chunk_shapes=chunk_shapes, halo=halo, pad_mode=pad_mode
    #         ),
    #         stat=self.stat,
    #         id=self.id,
    #     )

    # def chunkwise(
    #     self,
    #     func: Callable[[Sample], Sample],
    # ) -> Self:
    #     """Apply a function to each chunk of the chunked sample.

    #     See [chunk][] for chunking a sample.

    #     Args:
    #         func: The function to apply to each sample chunk.
    #     """
    #     return self.__class__(
    #         id=self.id,
    #         members=chunkwise(self.members, func),
    #         stat=self.stat,
    #     )

    # def split_into_blocks(
    #     self,
    #     block_shapes: PerMember[PerAxis[int]] | PerAxis[int],
    #     *,
    #     halo: PerMember[PerAxis[HaloLike]],
    #     broadcast: bool = False,
    # ) -> tuple[BlockLocShape, PerMember[BlockArray]]:
    #     """Split multiple shapes (e.g. sample member shapes) into aligned blocks.

    #     Args:
    #         block_shapes: the block shape per sample member (depending on `broaadcast` the block shapes must result in the same number of blocks for all sample members)
    #         halo: the halo around each block per sample member
    #         broadcast: whether to broadcast block shapes of members resulting in fewer blocks to match other sample members.
    #     """

    #     if unknown_blocks := [t for t in block_shapes if t not in self.shape]:
    #         raise ValueError(
    #             f"block shape specified for unknown tensors: {unknown_blocks}."
    #         )

    #     if not block_shapes:
    #         b_shapes = self.shape
    #     elif is_mapping(block_shapes, v_type=int):
    #         b_shapes = {m: block_shapes for m in self.shape}
    #     else:
    #         assert not any(isinstance(v, int) for v in block_shapes.values())
    #         b_shapes: PerMember[PerAxis[int]] = block_shapes  # pyright: ignore[reportAssignmentType]

    #     del block_shapes

    #     if not broadcast and (
    #         missing_blocks := [t for t in self.shape if t not in b_shapes]
    #     ):
    #         raise ValueError(
    #             f"No block shape specified for {missing_blocks}."
    #             + " Set `broadcast` to True if these tensors should be repeated"
    #             + " as a whole for each block."
    #         )

    #     if extra_halo := [t for t in halo if t not in b_shapes]:
    #         raise ValueError(
    #             f"`halo` specified for tensors without block shape: {extra_halo}."
    #         )

    #     blocks: PerMember[BlockArray] = {}
    #     block_locs: PerMember[BlockLocShape] = {}
    #     for m in b_shapes:
    #         block_locs[m], blocks[m] = self._split_member_into_blocks(
    #             member=m,
    #             block_shape=b_shapes[m],
    #             halo=halo.get(m, {}),
    #         )
    #         assert all(b > 0 for b in block_locs[m]), block_locs

    #     assert len(blocks) > 0, blocks
    #     assert len(block_locs) > 0, block_locs
    #     unique_block_locs = set(block_locs.values())

    #     if len(unique_block_locs) != 1:
    #         if not broadcast:
    #             raise ValueError(
    #                 f"Mismatch for number of blocks per dimension: { {t: b.shape for t, b in blocks.items()} }. Set `broadcast` to True if this is expected."
    #             )

    #         blocks = dict(zip(blocks.keys(), np.broadcast_arrays(*blocks.values())))
    #         block_locs = {m: b.shape for m, b in blocks.items()}
    #         unique_block_locs = set(block_locs.values())
    #         assert len(unique_block_locs) == 1, (
    #             f"Mismatch for total number of blocks (after braodcasting?!): {block_locs}"
    #         )

    #     return next(iter(unique_block_locs)), blocks

    # def foo(*bs: BlockMeta) -> SampleBlockMeta:
    #     return SampleBlockMeta.from_blocks(dict(zip(blocks.keys(), bs)))

    # foo_op = np.frompyfunc(foo, len(blocks), 1)
    # out = foo_op(*blocks.values())
    # assert out.shape == next(iter(blocks.values())).shape
    # return n, out

    # @property
    # def batch_multi_index(self) -> MultiIndex | None:
    #     """Return the batch multi-index of the sample, if it has one.

    #     Returns:
    #         The batch multi-index of the sample, or `None` if the sample does not have a batch dimension.
    #     """
    #     if not self.members:
    #         return None

    #     for tensor in self.members.values():
    #         idx = tensor.batch_multi_index
    #         if isinstance(idx, MultiIndex):
    #             return idx

    #     return None

    # def set_block(self, block: SampleBlock) -> None:
    #     """Set values of `block`.

    #     Note:
    #         - Updates only existing sample members (extra block members are ignored)
    #         - Ignores missing block members (i.e. members in the sample but not in the block are not modified)

    #     Raises:
    #         ValueError if block and sample members do not overlap at all.
    #     """
    #     no_overlap = True
    #     for m in self.members:
    #         if m not in block.blocks:
    #             continue
    #         b = block.blocks[m]
    #         self.members[m][b.inner_slice] = b.inner_data
    #         no_overlap = False

    #     if no_overlap:
    #         raise ValueError(
    #             f"block with members {list(block.blocks)} does not overlap with sample members {list(self.members)}"
    #         )

    @property
    def shape(self) -> PerMember[PerAxis[int]]:
        return {tid: t.sizes for tid, t in self.members.items()}

    # @property
    # def unstacked_batch_shape(self) -> PerAxis[int] | None:
    #     """The shape of the unstacked batch axis
    #     For example `{"t": 10, "z": 5}` for a sample with stacked "t" and "z" dimensions resulting in a batch length of 50).
    #     Note: As the batch axis has to match across members this is the same for all members."""
    #     if (idx := self.batch_multi_index) is None:
    #         return None

    #     return {
    #         BATCH_AXIS_ID if a == "original_batch" else AxisId(a): s
    #         for a, s in zip(idx.names, idx.levshape)
    #     }

    # @property
    # def unstacked_shape(self) -> PerMember[PerAxis[int]]:
    #     """combined (stacked) shape with the unstacked sample batch shape in order of unstacked_dim_order (if available)"""
    #     return {
    #         m: get_unstacked_shape(
    #             sh,
    #             unstacked_dim_order=self.unstacked_dim_order[m]
    #             if self.unstacked_dim_order is not None
    #             else None,
    #             unstacked_batch_shape=self.unstacked_batch_shape,
    #         )
    #         for m, sh in self.shape.items()
    #     }

    def as_arrays(self) -> dict[MemberId, NDArray[Any]]:
        """Return sample as dictionary of arrays."""
        return {m: t.to_numpy() for m, t in self.members.items()}

    # def split_into_blocks_meta(
    #     self,
    #     block_shapes: PerMember[PerAxis[int]],
    #     halo: PerMember[PerAxis[HaloLike]],
    #     broadcast: bool = False,
    # ) -> tuple[BlockLocShape, NDArray[Any]]:  # really NDArray[SampleBlockMeta]
    #     assert not (missing := [m for m in block_shapes if m not in self.members]), (
    #         f"`block_shapes` specified for unknown members: {missing}"
    #     )
    #     assert not (missing := [m for m in halo if m not in block_shapes]), (
    #         f"`halo` specified for members without `block_shape`: {missing}"
    #     )

    #     n, blocks = split_multiple_shapes_into_blocks(
    #         shapes=self.shape,
    #         unstacked_batch_shape=self.unstacked_batch_shape,
    #         unstacked_dim_order=self.unstacked_dim_order,
    #         block_shapes=block_shapes,
    #         halo=halo,
    #         broadcast=broadcast,
    #     )
    #     return n, sample_block_meta_generator(blocks, sample_id=self.id)

    # def split_into_blocks(
    #     self,
    #     block_shapes: PerMember[PerAxis[int]],
    #     halo: PerMember[PerAxis[HaloLike]],
    #     pad_mode: PadMode | PerMember[PadMode],
    #     broadcast: bool = False,
    # ) -> tuple[BlockLocShape, Iterable[SampleBlockWithOrigin]]:
    #     n_blocks, blocks = self.split_into_blocks_meta(block_shapes, halo, broadcast)
    #     return n_blocks, sample_block_generator(
    #         blocks.flatten(), origin=self, pad_mode=pad_mode
    #     )

    # def get_blocks(
    #     self,
    #     block_shapes: PerMember[PerAxis[int]],
    #     *,
    #     halo: PerMember[PerAxis[HaloLike]],
    #     pad_mode: PadMode | PerMember[PadMode],
    #     broadcast: bool = False,
    # ) -> tuple[BlockLocShape, PerMember[BlockMetaArray]]:
    #     n_blocks, blocks = self.split_into_blocks_meta(block_shapes, halo, broadcast)
    #     return n_blocks, sample_block_generator(
    #         blocks.flatten(), origin=self, pad_mode=pad_mode
    #     )

    # def as_single_block(self, halo: PerMember[PerAxis[Halo]] | None = None):
    #     if halo is None:
    #         halo = {}

    #     return SampleBlockWithOrigin(
    #         sample_shape=self.shape,
    #         sample_id=self.id,
    #         blocks={
    #             m: Block(
    #                 sample_shape=self.shape[m],
    #                 unstacked_sample_batch_shape=self.unstacked_batch_shape,
    #                 unstacked_dim_order=self.unstacked_dim_order[m]
    #                 if self.unstacked_dim_order is not None
    #                 else None,
    #                 data=data,
    #                 inner_slice={a: SliceInfo(0, s) for a, s in self.shape[m].items()},
    #                 unstacked_batch_slice=None
    #                 if self.unstacked_batch_shape is None
    #                 else {
    #                     a: SliceInfo(0, s)
    #                     for a, s in self.unstacked_batch_shape.items()
    #                 },
    #                 halo=halo.get(m, {}),
    #                 block_loc=0,
    #                 block_loc_shape=1,
    #             )
    #             for m, data in self.members.items()
    #         },
    #         stat=self.stat,
    #         origin=self,
    #         block_loc=0,
    #         blocks_loc_shape=1,
    #     )

    # @classmethod
    # def from_blocks(
    #     cls,
    #     sample_blocks: Iterable[SampleBlock],
    #     *,
    #     fill_value: float = float("nan"),
    # ) -> Self:
    #     """Create a `Sample` from an iterable of `SampleBlock`s.

    #     Note:
    #         All sample blocks must have the same `sample_id`.

    #     Args:
    #         sample_blocks: The blocks to create the sample from.
    #         fill_value: The value to fill missing values with (default: `nan`).
    #     """
    #     output = None
    #     for output in cls.from_blocks_yield_intermediates(
    #         sample_blocks, fill_value=fill_value
    #     ):
    #         pass

    #     if output is None:
    #         raise ValueError("no sample blocks provided")

    #     return output

    # @classmethod
    # def from_blocks_yield_intermediates(
    #     cls,
    #     sample_blocks: Iterable[SampleBlock],
    #     *,
    #     fill_value: float = float("nan"),
    # ):
    #     """Create a `Sample` from an iterable of `SampleBlock`s, yielding the intermediate sample after each block.

    #     Args:
    #         sample_blocks: The blocks to create the sample from.
    #         fill_value: The value to fill missing values with (default: `nan`).
    #     """
    #     output = None
    #     for sample_block in sample_blocks:
    #         if output is None:
    #             output = cls(
    #                 members={},
    #                 stat={},
    #                 id=None,
    #                 unstacked_dim_order=sample_block.unstacked_dim_order,
    #             )
    #         if output.id is None:
    #             output.id = sample_block.sample_id
    #         else:
    #             assert output.id == sample_block.sample_id, (
    #                 "sample id changed between sample blocks"
    #             )

    #         output.stat = sample_block.stat

    #         for m, block in sample_block.blocks.items():
    #             if m not in output.members:
    #                 if -1 in block.sample_shape.values():
    #                     raise NotImplementedError(
    #                         "merging blocks with data dependent axis not yet implemented"
    #                     )

    #                 if (
    #                     block.unstacked_sample_batch_shape is not None
    #                     and block.unstacked_dim_order is not None
    #                 ):
    #                     sizes = {
    #                         **block.sample_shape,
    #                         **block.unstacked_sample_batch_shape,
    #                     }
    #                     full_unstacked_shape = {
    #                         a: sizes[a] for a in block.unstacked_dim_order
    #                     }
    #                     t = create_tensor_with_multiindex(
    #                         full_unstacked_shape,
    #                         fill_value=fill_value,
    #                         dtype=np.dtype(block.data.dtype),
    #                         transpose_to=tuple(block.sample_shape.keys()),
    #                     )
    #                 else:
    #                     t = Tensor(
    #                         np.full(
    #                             shape=tuple(block.sample_shape.values()),
    #                             fill_value=fill_value,
    #                             dtype=block.data.dtype,
    #                         ),
    #                         dims=tuple(block.sample_shape.keys()),
    #                     )

    #                 output.members[m] = t

    #             output.members[m][block.inner_slice] = block.inner_data
    #         yield output

    #     yield output

    def pad(
        self,
        pad_width: PerMember[PerAxis[int | PadWidthLike]],
        mode: PerMember[PadMode] | PadMode,
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

    # def transpose(
    #     self,
    #     axes: PerMember[Sequence[AxisId]],
    #     *,
    #     extra_dims: Literal["raise", "squeeze", "stack", "squeeze_or_stack"] = "raise",
    #     missing_dims: Literal[
    #         "raise", "expand", "unstack", "unstack_or_expand"
    #     ] = "raise",
    # ) -> Self:
    #     """Return a new sample with transposed sample members.

    #     Raises:
    #         ValueError: If not all batch dimensions have the same length after transposition (and possibly stacking/unstacking extra dimensions).

    #     """
    #     if any(unknown := [m not in self.members for m in axes]):
    #         raise ValueError(f"Axes specified for unknown members: {unknown}")

    #     members: dict[MemberId, Tensor] = {}
    #     individual_unstacked_dim_order: dict[MemberId, list[AxisId] | None] = {}
    #     for m, t in self.members.items():
    #         if m in axes:
    #             members[m], individual_unstacked_dim_order[m] = (
    #                 t.transpose_and_return_unstacked_dim_order(
    #                     axes=axes[m],
    #                     extra_dims=extra_dims,
    #                     missing_dims=missing_dims,
    #                 )
    #             )
    #         else:
    #             members[m] = t
    #             individual_unstacked_dim_order[m] = None

    #     # if any(order is None for order in individual_unstacked_dim_order.values()):
    #     #     unstacked_dim_order = None
    #     # else:
    #     #     unstacked_dim_order = {
    #     #         m: order
    #     #         for m, order in individual_unstacked_dim_order.items()
    #     #         if order is not None
    #     #     }

    #     if (
    #         len(
    #             batch_lengths := {
    #                 t.sizes[AxisId("batch")]
    #                 for t in members.values()
    #                 if AxisId("batch") in t.dims
    #             }
    #         )
    #         > 1
    #     ):
    #         raise ValueError(
    #             f"Transposed sample members have incompatible batch lengths: {batch_lengths}."
    #         )

    #     return self.__class__(members=members, stat=dict(self.stat), id=self.id)

    # def assign_batch_multi_index(self, multi_index: MultiIndex) -> Self:
    #     """Return a new sample with the batch multi-index assigned to all sample members.

    #     Raises:
    #         ValueError: If not all sample members have a batch dimension.
    #     """
    #     if len(
    #         no_batch := [
    #             m for m, t in self.members.items() if AxisId("batch") not in t.dims
    #         ]
    #     ) == len(self.members):
    #         raise ValueError(f"No member has a batch dimension: {no_batch}")

    #     return self.__class__(
    #         members={
    #             m: t
    #             if AxisId("batch") not in t.dims
    #             else t.assign_batch_multi_index(multi_index)
    #             for m, t in self.members.items()
    #         },
    #         stat=dict(self.stat),
    #         id=self.id,
    #         unstacked_dim_order=self.unstacked_dim_order,
    #     )

    def unstack_batch_multi_index(
        self,
        *,
        errors: Literal["raise", "warn", "ignore"] = "raise",
        unstacked_dim_order: PerMember[Sequence[AxisId]] | None = None,
    ) -> Self:
        """Unstack the batch multi-index of all sample members.

        Args:
            errors: Whether to raise an error if a member does not have a batch multi-index. Default is "raise".

        Returns:
            A new `Sample` with unstacked batch multi-index for all members.
        """
        if (
            len(
                no_batch := [
                    m for m, t in self.members.items() if AxisId("batch") not in t.dims
                ]
            )
            == len(self.members)
            and errors != "ignore"
        ):
            error_msg = f"No member has a batch dimension: {no_batch}"
            if errors == "raise":
                raise ValueError(error_msg)
            elif errors == "warn":
                logger.warning(error_msg)
            else:
                assert_never(errors)

        members = {
            m: t
            if AxisId("batch") not in t.dims
            else t.unstack_batch_multi_index(
                errors=errors,
                unstacked_dim_order=unstacked_dim_order[m]
                if unstacked_dim_order is not None
                else None,
            )
            for m, t in self.members.items()
        }
        if (
            len(
                batch_lengths := {
                    t.sizes.get(AxisId("batch"))
                    for t in members.values()
                    if AxisId("batch") in t.dims
                }
            )
            > 1
        ):
            raise ValueError(
                f"Different batch lengths after unstacking: {batch_lengths}"
            )

        stat: Stat = {  # TODO: update stat calculations to maintain any multi-index batch index.
            k: v.unstack_batch_multi_index(errors="ignore", unstacked_dim_order=None)
            if isinstance(v, Tensor)
            else float(v)
            for k, v in self.stat.items()
        }
        return self.__class__(
            members=members,
            stat=stat,
            id=self.id,
        )


# @dataclass(frozen=True)
# class SampleBlock(SampleBlockBase[Block]):
#     """A block of a dataset sample"""

#     blocks: dict[MemberId, Block]
#     """Individual tensor blocks comprising this sample block"""

#     stat: Stat
#     """computed statistics"""

#     @property
#     def members(self) -> PerMember[Tensor]:
#         """the sample block's tensors"""
#         return {m: b.data for m, b in self.blocks.items()}

#     @classmethod
#     def from_meta(
#         cls, meta: SampleBlockMeta, data: PerMember[Tensor], *, stat: Stat
#     ) -> SampleBlock:
#         return cls(
#             sample_shape={
#                 m: {
#                     a: data[m].tagged_shape[a] if s == -1 else s
#                     for a, s in member_shape.items()
#                 }
#                 for m, member_shape in meta.sample_shape.items()
#             },
#             sample_id=meta.sample_id,
#             blocks={
#                 m: Block.from_meta(b, data=data[m]) for m, b in meta.blocks.items()
#             },
#             stat=stat,
#             block_loc=meta.block_loc,
#             blocks_loc_shape=meta.blocks_loc_shape,
#         )

#     def get_meta(self) -> SampleBlockMeta:
#         return SampleBlockMeta(
#             sample_id=self.sample_id,
#             blocks={m: b.get_meta() for m, b in self.blocks.items()},
#             sample_shape=self.sample_shape,
#             block_loc=self.block_loc,
#             blocks_loc_shape=self.blocks_loc_shape,
#         )

#     def as_sample(self) -> Sample:
#         """Convert this sample block to a `Sample` with the shape of this block.

#         Note:
#             If you want to convert one or more sample block to a sample with the shape of the original, whole sample,
#             use `Sample.from_blocks()` instead.
#         """
#         return Sample(
#             members=dict(self.members),
#             stat=dict(self.stat),
#             id=self.sample_id,
#             unstacked_dim_order=self.unstacked_dim_order,
#         )

#     @property
#     def unstacked_dim_order(self) -> PerMember[Sequence[AxisId] | None]:
#         return {m: b.unstacked_dim_order for m, b in self.blocks.items()}

#     @property
#     def batch_multi_index(self) -> MultiIndex | None:
#         """Return the batch multi-index of the sample block, if it has one.

#         Returns:
#             The batch multi-index of the sample block, or `None` if the sample block does not have a multi-index batch dimension.
#         """
#         for tensor in self.members.values():
#             idx = tensor.batch_multi_index
#             if idx is not None:
#                 return idx

#         return None


# @dataclass(frozen=True)
# class SampleBlockWithOrigin(SampleBlock):
#     """A `SampleBlock` with a reference (`origin`) to the whole `Sample`"""

#     origin: Sample
#     """the sample this sample block was taken from"""


# def sample_block_generator(
#     blocks: Iterable[PerMember[BlockMeta] | SampleBlockMeta],
#     *,
#     origin: Sample,
#     pad_mode: PadMode | PerMember[PadMode],
# ) -> Iterable[SampleBlockWithOrigin]:
#     for member_blocks in blocks:
#         if isinstance(member_blocks, SampleBlockMeta):
#             member_blocks = member_blocks.blocks

#         cons = _ConsolidatedMemberBlocks(member_blocks)
#         yield SampleBlockWithOrigin(
#             blocks={
#                 m: Block.from_sample_member(
#                     origin.members[m],
#                     block=member_blocks[m],
#                     pad_mode=pad_mode.get(m, "symmetric")
#                     if isinstance(pad_mode, collections.abc.Mapping)
#                     else pad_mode,
#                 )
#                 for m in origin.members
#             },
#             sample_shape=origin.shape,
#             origin=origin,
#             stat=origin.stat,
#             sample_id=origin.id,
#             block_loc=cons.block_index,
#             blocks_loc_shape=cons.blocks_in_sample,
#         )
