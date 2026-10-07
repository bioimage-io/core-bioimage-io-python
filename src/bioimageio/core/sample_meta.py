# from __future__ import annotations

# from collections.abc import Sequence
# from dataclasses import dataclass
# from math import ceil, floor
# from types import MappingProxyType
# from typing import (
#     Any,
#     Callable,
#     Generic,
#     TypeVar,
# )

# import numpy as np
# import pydantic
# from numpy.typing import NDArray
# from typing_extensions import Self, assert_never

# from bioimageio.core._utils.multi_index import get_unstacked_dim_order
# from bioimageio.core._utils.type_guards import is_mapping

# from ._common_annotations import PerMemberAnno
# from .axis import AxisId, PerAxis
# from .block_meta import (
#     BlockMeta,
#     LinearAxisTransform,
# )
# from .common import (
#     BlockLoc,
#     BlockLocShape,
#     Halo,
#     MemberId,
#     PerMember,
#     SampleId,
#     SliceInfo,
# )

# BlockT = TypeVar("BlockT", bound=BlockMeta)


# @pydantic.dataclasses.dataclass(frozen=True)
# class SampleBlockBase(Generic[BlockT]):
#     """base class for `SampleBlockMeta` and `SampleBlock`"""

#     sample_shape: PerMemberAnno[PerAxis[int]]
#     """the sample shape this block represents a part of"""

#     sample_id: SampleId
#     """identifier for the sample within its dataset"""

#     blocks: PerMemberAnno[BlockT]
#     """Individual tensor blocks comprising this sample block"""

#     block_loc: BlockLoc
#     """the location of the block within the sample"""

#     blocks_loc_shape: BlockLocShape
#     """the sample shape in number of blocks"""

#     @property
#     def shape(self) -> PerMember[PerAxis[int]]:
#         return MappingProxyType({mid: b.shape for mid, b in self.blocks.items()})

#     @property
#     def inner_shape(self) -> PerMember[PerAxis[int]]:
#         return MappingProxyType({mid: b.inner_shape for mid, b in self.blocks.items()})

#     @property
#     def unstacked_sample_batch_shape(self) -> PerAxis[int] | None:
#         """The shape of the unstacked batch axis, if the sample has a multi-index batch axis"""
#         for b in self.blocks.values():
#             if b.unstacked_sample_batch_shape is not None:
#                 return b.unstacked_sample_batch_shape

#         return None

#     @property
#     def unstacked_dim_order(self) -> PerMemberAnno[Sequence[AxisId] | None] | None:
#         """The dim order of the unstacked tensors, if the sample has a multi-index batch axis"""
#         for b in self.blocks.values():
#             if b.unstacked_dim_order is not None:
#                 return MappingProxyType(
#                     {mid: bb.unstacked_dim_order for mid, bb in self.blocks.items()}
#                 )

#         return None

#     @property
#     def unstacked_sample_shape(self) -> PerMemberAnno[PerAxis[int]]:
#         """combined (stacked) sample shape with the unstacked sample batch shape in order of unstacked_dim_order (if available)"""
#         return MappingProxyType(
#             {mid: b.unstacked_sample_shape for mid, b in self.blocks.items()}
#         )


# @dataclass
# class LinearSampleAxisTransform(LinearAxisTransform):
#     member: MemberId


# @pydantic.dataclasses.dataclass(frozen=True)
# class SampleBlockMeta(SampleBlockBase[BlockMeta]):
#     """Meta data of a dataset sample block"""

#     # @classmethod
#     # def from_blocks(cls, meta: PerMember[BlockMeta]) -> Self:
#     #     sample_ids = {
#     #         b.sample_id for b in meta.values() if b.sample_id is not None
#     #     } or {"unknown"}
#     #     block_loc_shapes = {b.block_loc_shape for b in meta.values()}
#     #     if len(block_loc_shapes) != 1:
#     #         raise ValueError(
#     #             f"Mismatch for block_loc_shape (sample shape in blocks) across members: {block_loc_shapes}"
#     #         )
#     #     block_locs = {b.block_loc for b in meta.values()}
#     #     if len(block_locs) != 1:
#     #         raise ValueError(
#     #             f"Mismatch for block_loc (block location within sample) across members: {block_locs}"
#     #         )
#     #     return cls(
#     #         sample_shape={m: b.sample_shape for m, b in meta.items()},
#     #         sample_id=next(iter(sample_ids)),
#     #         blocks=meta,
#     #         block_loc=next(iter(block_locs)),
#     #         blocks_loc_shape=next(iter(block_loc_shapes)),
#     #     )

#     def get_transformed(
#         self, new_axes: PerMember[PerAxis[LinearSampleAxisTransform | int]]
#     ) -> Self:
#         sample_shape = {
#             m: {
#                 a: (
#                     trf
#                     if isinstance(trf, int)
#                     else trf.compute(self.sample_shape[trf.member][trf.axis])
#                 )
#                 for a, trf in new_axes[m].items()
#             }
#             for m in new_axes
#         }

#         def get_member_halo(m: MemberId, round: Callable[[float], int]):
#             return {
#                 a: (
#                     Halo(0, 0)
#                     if isinstance(trf, int)
#                     or trf.axis not in self.blocks[trf.member].halo
#                     else Halo(
#                         round(self.blocks[trf.member].halo[trf.axis].left * trf.scale),
#                         round(self.blocks[trf.member].halo[trf.axis].right * trf.scale),
#                     )
#                 )
#                 for a, trf in new_axes[m].items()
#             }

#         halo: dict[MemberId, dict[AxisId, Halo]] = {}
#         for m in new_axes:
#             halo[m] = get_member_halo(m, floor)
#             if halo[m] != get_member_halo(m, ceil):
#                 raise ValueError(
#                     f"failed to unambiguously scale halo {halo[m]} with {new_axes[m]}"
#                     + f" for {m}."
#                 )

#         inner_slice = {
#             m: {
#                 a: (
#                     SliceInfo(0, trf)
#                     if isinstance(trf, int)
#                     else SliceInfo(
#                         trf.compute(
#                             self.blocks[trf.member].inner_slice[trf.axis].start
#                         ),
#                         trf.compute(self.blocks[trf.member].inner_slice[trf.axis].stop),
#                     )
#                 )
#                 for a, trf in new_axes[m].items()
#             }
#             for m in new_axes
#         }
#         unstacked_batch_slice = None
#         for b in self.blocks.values():
#             if b.unstacked_batch_slice is None:
#                 continue
#             elif unstacked_batch_slice is None:
#                 unstacked_batch_slice = b.unstacked_batch_slice
#             else:
#                 assert unstacked_batch_slice == b.unstacked_batch_slice, (
#                     "Incompatible unstacked batch slices across sample block members"
#                 )

#         unstacked_dim_order: dict[MemberId, list[AxisId] | None] = {}
#         for m, trfs in new_axes.items():
#             for trf in trfs.values():
#                 if isinstance(trf, LinearSampleAxisTransform):
#                     unstacked_in_order = self.blocks[trf.member].unstacked_dim_order
#                     stacked_dims = list(
#                         self.blocks[trf.member].unstacked_sample_batch_shape or ()
#                     )
#                     unstacked_dim_order[m] = get_unstacked_dim_order(
#                         list(trfs),
#                         unstacked_reference_order=unstacked_in_order,
#                         stacked_dims=stacked_dims,
#                     )
#                     break
#                 elif isinstance(trf, int):
#                     pass
#                 else:
#                     assert_never(trf)
#             else:
#                 unstacked_dim_order[m] = None

#         return self.__class__(
#             blocks={
#                 m: BlockMeta(
#                     sample_shape=sample_shape[m],
#                     sample_id=self.sample_id,
#                     unstacked_sample_batch_shape=self.unstacked_sample_batch_shape,
#                     unstacked_dim_order=unstacked_dim_order[m],
#                     inner_slice=inner_slice[m],
#                     unstacked_batch_slice=unstacked_batch_slice,
#                     halo=halo[m],
#                     block_loc=self.block_loc,
#                     block_loc_shape=self.blocks_loc_shape,
#                 )
#                 for m in new_axes
#             },
#             sample_shape=sample_shape,
#             sample_id=self.sample_id,
#             block_loc=self.block_loc,
#             blocks_loc_shape=self.blocks_loc_shape,
#         )


# class _ConsolidatedMemberBlocks:
#     def __init__(self, blocks: PerMember[BlockMeta]):
#         super().__init__()
#         block_indices = {b.block_loc for b in blocks.values()}
#         assert len(block_indices) == 1
#         self.block_index = block_indices.pop()
#         blocks_in_samples = {b.block_loc_shape for b in blocks.values()}
#         assert len(blocks_in_samples) == 1
#         self.blocks_in_sample = blocks_in_samples.pop()


# def sample_block_meta_generator(
#     blocks: NDArray[Any],  # NDArray[PerMember[BlockMeta]],
#     *,
#     sample_id: SampleId,
# ):
#     sample_shape: dict[MemberId, PerAxis[int]] = {}
#     # out = np.empty(blocks.shape, dtype=object).flatten()
#     out: list[SampleBlockMeta] = []

#     # member_blocks: PerMember[BlockMeta]
#     for member_blocks in blocks.flat:  # pyright: ignore[reportUnknownVariableType]
#         assert is_mapping(member_blocks, MemberId, BlockMeta)
#         cons = _ConsolidatedMemberBlocks(member_blocks)
#         unstacked_batch_shape = None
#         for m, b in member_blocks.items():
#             if b.unstacked_sample_batch_shape is not None:
#                 if unstacked_batch_shape is None:
#                     unstacked_batch_shape = b.unstacked_sample_batch_shape
#                 else:
#                     if unstacked_batch_shape != b.unstacked_sample_batch_shape:
#                         raise ValueError(
#                             f"Unstacked batch shape mismatch for member {m}: "
#                             + f"{unstacked_batch_shape} != {b.unstacked_sample_batch_shape}"
#                         )

#             if m not in sample_shape:
#                 sample_shape[m] = b.sample_shape
#             else:
#                 assert b.sample_shape == sample_shape[m]

#         out.append(
#             SampleBlockMeta(
#                 blocks=dict(member_blocks),
#                 sample_shape=sample_shape,
#                 sample_id=sample_id,
#                 block_loc=cons.block_index,
#                 blocks_loc_shape=cons.blocks_in_sample,
#             )
#         )

#     return np.asarray(out, dtype=object).reshape(blocks.shape)
