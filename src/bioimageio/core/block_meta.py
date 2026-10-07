# from __future__ import annotations

# import itertools
# from collections.abc import Collection, Iterator
# from dataclasses import dataclass
# from functools import cached_property
# from math import floor
# from types import MappingProxyType
# from typing import (
#     Annotated,
#     Callable,
#     Iterable,
#     Iterator,
# )

# import numpy as np
# import pydantic
# from loguru import logger
# from numpy.typing import NDArray
# from typing_extensions import Self

# from ._axis_annotations import PerAxisAnno
# from .axis import BATCH_AXIS_ID, AxisId, PerAxis
# from .common import (
#     BlockLoc,
#     BlockLocShape,
#     Halo,
#     HaloLike,
#     PadWidth,
#     PerMember,
#     SliceInfo,
#     SliceInfoWithOverlap,
# )


# @dataclass
# class LinearAxisTransform:
#     axis: AxisId
#     scale: float
#     offset: int

#     def compute(self, s: int, round: Callable[[float], int] = floor) -> int:
#         return round(s * self.scale) + self.offset


# @pydantic.dataclasses.dataclass(frozen=True)
# class BlockMeta:
#     """Block meta data of a sample member (a tensor in a sample)

#     Figure for illustration:
#     The first 2d block (dashed) of a sample member (**bold**).
#     The inner slice (thin) is expanded by a halo in both dimensions on both sides.
#     The outer slice reaches from the sample member origin (0, 0) to the right halo point.

#     ```
#     first block (at the sample origin)
#     ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
#     ╷ halo(left)                         ╷
#     ╷         padding outside the sample ╷
#     ╷  (0, 0)┏━━━━━━━━━━━━━━━━━┯━━━━━━━━━┯━━━➔
#     ╷        ┃                 │         ╷  sample member
#     ╷        ┃      inner      │  outer  ╷
#     ╷        ┃      region     │  region ╷
#     ╷        ┃      /slice     │  /slice ╷
#     ╷        ┃                 │         ╷
#     ╷        ┣─────────────────┘         ╷
#     ╷        ┃   outer region/slice      ╷
#     ╷        ┃               halo(right) ╷
#     └ ─ ─ ─ ─┃─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
#              ⬇
#     ```

#     Note:
#     - Inner and outer slices are specified in sample member coordinates.
#     - The outer_slice of a block at the sample edge may overlap by more than the
#         halo with the neighboring block (the inner slices will not overlap though).

#     """

#     sample_shape: PerAxisAnno[int]
#     """the axis sizes of the whole (unblocked) sample"""

#     inner_slice: PerAxisAnno[SliceInfo | SliceInfoWithOverlap]
#     """inner region (without halo) wrt the sample"""

#     halo: PerAxisAnno[Halo]
#     """halo enlarging the inner region to the block's sizes"""

#     block_loc: BlockLoc
#     """the block index within the sample"""

#     block_loc_shape: BlockLocShape
#     """The sample shape in numbers of blocks"""

#     @cached_property
#     def shape(self) -> PerAxis[int]:
#         """axis lengths of the block"""
#         return MappingProxyType(
#             {
#                 a: s.stop - s.start + (sum(self.halo[a]) if a in self.halo else 0)
#                 for a, s in self.inner_slice.items()
#             }
#         )

#     # @property
#     # def unstacked_sample_shape(self) -> PerAxis[int]:
#     #     """combined (stacked) sample shape with the unstacked sample batch shape in order of unstacked_dim_order (if available)"""
#     #     return get_unstacked_shape(
#     #         self.sample_shape,
#     #         unstacked_dim_order=self.unstacked_dim_order,
#     #         unstacked_batch_shape=self.unstacked_sample_batch_shape,
#     #     )

#     @cached_property
#     def padding(self) -> PerAxis[PadWidth]:
#         """padding to realize the halo at the sample edge
#         where we cannot simply enlarge the inner slice"""
#         return MappingProxyType(
#             {
#                 a: p
#                 for a in self.inner_slice
#                 if (
#                     p := PadWidth(
#                         (
#                             self.halo[a].left
#                             - (self.inner_slice[a].start - self.outer_slice[a].start)
#                             if a in self.halo
#                             else 0
#                         ),
#                         (
#                             self.halo[a].right
#                             - (self.outer_slice[a].stop - self.inner_slice[a].stop)
#                             if a in self.halo
#                             else 0
#                         ),
#                     )
#                 )
#                 != PadWidth(0, 0)
#             }
#         )

#     @cached_property
#     def outer_slice(self) -> PerAxis[SliceInfo]:
#         """slice of the outer block (without padding) wrt the sample"""
#         return MappingProxyType(
#             {
#                 a: SliceInfo(
#                     max(
#                         0,
#                         min(
#                             self.inner_slice[a].start
#                             - (self.halo[a].left if a in self.halo else 0),
#                             self.sample_shape[a]
#                             - self.inner_shape[a]
#                             - (self.halo[a].left if a in self.halo else 0),
#                         ),
#                     ),
#                     min(
#                         self.sample_shape[a],
#                         self.inner_slice[a].stop
#                         + (self.halo[a].right if a in self.halo else 0),
#                     ),
#                 )
#                 for a in self.inner_slice
#             }
#         )

#     @cached_property
#     def inner_shape(self) -> PerAxis[int]:
#         """axis lengths of the inner region (without halo)"""
#         return MappingProxyType(
#             {a: s.stop - s.start for a, s in self.inner_slice.items()}
#         )

#     @cached_property
#     def local_slice(self) -> PerAxis[SliceInfo]:
#         """inner slice wrt the block, **not** the sample"""
#         return MappingProxyType(
#             {
#                 a: (
#                     SliceInfo(
#                         h.left,
#                         h.left + self.inner_shape[a],
#                     )
#                     if (h := self.halo.get(a)) is not None
#                     else SliceInfo(0, self.inner_shape[a])
#                 )
#                 for a in self.inner_slice
#             }
#         )

#     @property
#     def dims(self) -> Collection[AxisId]:
#         return set(self.inner_shape)

#     @property
#     def tagged_shape(self) -> PerAxis[int]:
#         """alias for shape"""
#         return self.shape

#     @property
#     def inner_slice_wo_overlap(self) -> PerAxis[SliceInfo]:
#         """subslice of the inner slice, such that all `inner_slice_wo_overlap` can be
#         stiched together trivially to form the original sample.

#         This can also be used to calculate statistics
#         without overrepresenting block edge regions."""
#         return {
#             a: SliceInfo(s.start_wo_overlap, s.stop_wo_overlap)
#             if isinstance(s, SliceInfoWithOverlap)
#             else SliceInfo(s.start, s.stop)
#             for a, s in self.inner_slice.items()
#         }

#     def __post_init__(self):
#         assert all(a in self.sample_shape for a in self.inner_slice), (
#             "block has axes not present in sample"
#         )

#         assert all(a in self.inner_slice for a in self.halo), (
#             "halo has axes not present in block"
#         )

#         if any(s > self.sample_shape[a] for a, s in self.shape.items()):
#             logger.warning(
#                 "block {} larger than sample {}", self.shape, self.sample_shape
#             )

#     def get_transformed(self, new_axes: PerAxis[LinearAxisTransform | int]) -> Self:
#         assert BATCH_AXIS_ID not in new_axes, "cannot transform the batch axis"

#         sample_shape = {
#             a: (
#                 trf
#                 if isinstance(trf, int)
#                 else trf.compute(self.sample_shape[trf.axis])
#             )
#             for a, trf in new_axes.items()
#         }
#         assert sample_shape.get(BATCH_AXIS_ID) == self.sample_shape.get(
#             BATCH_AXIS_ID
#         ), "block transform must not change the batch axis size"

#         inner_slice = {
#             a: (
#                 SliceInfo(0, trf)
#                 if isinstance(trf, int)
#                 else SliceInfo(
#                     trf.compute(self.inner_slice[trf.axis].start),
#                     trf.compute(self.inner_slice[trf.axis].stop),
#                 )
#             )
#             for a, trf in new_axes.items()
#         }
#         assert inner_slice.get(BATCH_AXIS_ID) == self.inner_slice.get(BATCH_AXIS_ID), (
#             "block transform must not change the batch axis slice"
#         )

#         return self.__class__(
#             sample_shape=sample_shape,
#             # sample_id=self.sample_id,
#             inner_slice=inner_slice,
#             halo={
#                 a: (
#                     Halo(0, 0)
#                     if isinstance(trf, int)
#                     else Halo(self.halo[trf.axis].left, self.halo[trf.axis].right)
#                 )
#                 for a, trf in new_axes.items()
#             },
#             block_loc=self.block_loc,
#             block_loc_shape=self.block_loc_shape,
#         )


# BlockArray = Annotated[NDArray[np.object_], "NDArray[Block]"]


# def split_shape_into_blocks(
#     shape: PerAxis[int],
#     *,
#     block_shape: PerAxis[int],
#     halo: PerAxis[HaloLike],
#     # sample_id: SampleId,
# ) -> tuple[BlockLocShape, BlockArray]:
#     """
#     Args:
#         item_size: size of a single element in bytes, used to find shard and chunk sizes
#     """
#     if unknown_axes := [
#         a for a in block_shape if a not in shape and block_shape[a] != 1
#     ]:
#         raise ValueError(
#             f"unknown axes in block_shape: {unknown_axes} for shape {shape}"
#         )

#     if large_block_dims := [
#         (a, b, shape_size)
#         for a in block_shape
#         if (shape_size := shape.get(a, 1)) < (b := block_shape[a])
#     ]:
#         logger.warning(
#             "Block shape dimensions {} larger than shape dimensions {} are limiting the shape size",
#             {a: b for a, b, _s in large_block_dims},
#             {a: s for a, _b, s in large_block_dims},
#         )

#     block_shape = {a: min(s, block_shape.get(a, s)) for a, s in shape.items()}
#     if unknown_halo_dims := [a for a in halo if a not in shape]:
#         raise ValueError(f"unknown dims in halo: {unknown_halo_dims} for shape {shape}")

#     # fill in default halo (0) and block axis length (from tensor shape)
#     halo = {a: Halo.create(halo.get(a, 0)) for a in shape}

#     inner_1d_slices: dict[AxisId, list[SliceInfo | SliceInfoWithOverlap]] = {}
#     inner_sizes = {a: block_shape[a] - sum(halo[a]) for a in shape}
#     for a, shape_size in shape.items():
#         inner_size = inner_sizes[a]
#         inner_1d_slices[a] = [
#             SliceInfo(p, min(p + inner_size, shape_size))
#             if p <= shape_size - inner_size
#             else SliceInfoWithOverlap(
#                 shape_size - inner_size,
#                 (stop := min(p + inner_size, shape_size)),
#                 shape_size - inner_size,
#                 stop,
#             )
#             for p in range(0, shape_size, inner_size)
#         ]

#     # n_blocks = prod(map(len, inner_1d_slices.values()))

#     #     return n_blocks, _block_meta_generator(
#     #         shape,
#     #         unstacked_batch_shape,
#     #         blocks_in_sample=n_blocks,
#     #         unstacked_dim_order=unstacked_dim_order,
#     #         inner_1d_slices=inner_1d_slices,
#     #         halo=halo,
#     #     )

#     # def _block_meta_generator(
#     #     sample_shape: PerAxis[int],
#     #     *,
#     #     unstacked_dim_order: Sequence[AxisId] | None,
#     #     blocks_in_sample: int,
#     #     inner_1d_slices: dict[AxisId, list[SliceInfo | SliceInfoWithOverlap]],
#     #     halo: PerAxis[HaloLike],
#     #     sample_id: SampleId,
#     # ) -> BlockMetaArray:
#     assert all(a in shape for a in halo)

#     halo = {a: Halo.create(halo.get(a, 0)) for a in inner_1d_slices}
#     # if unstacked_batch_shape is None:
#     #     idx: MultiIndex | None = None
#     # else:
#     #     idx = MultiIndex.from_product(  # pyright: ignore[reportUnknownVariableType]
#     #         [range(unstacked_batch_shape[a]) for a in unstacked_batch_shape],
#     #         names=list(unstacked_batch_shape),
#     #     )
#     #     assert isinstance(idx, MultiIndex)

#     out: list[BlockMeta] = []
#     out_shape = tuple(len(slices) for slices in inner_1d_slices.values())
#     for i, nd_tile in enumerate(itertools.product(*inner_1d_slices.values())):
#         inner_1d_slices[BATCH_AXIS_ID]
#         inner_slice = dict(zip(inner_1d_slices, nd_tile))
#         # if idx is None:
#         #     unstacked_batch_slice = None
#         # else:
#         #     batch_slice = inner_slice[BATCH_AXIS_ID]
#         #     unstacked_batch_slice: dict[AxisId, SliceInfo] | None = {}
#         #     for a, c in zip(idx.names, idx.codes):
#         #         a = BATCH_AXIS_ID if a == "original_batch" else AxisId(a)
#         #         a_start = int(c[batch_slice.start])
#         #         a_stop = int(c[batch_slice.stop - 1]) + 1
#         #         unstacked_batch_slice[a] = SliceInfo(a_start, a_stop)
#         out.append(
#             BlockMeta(
#                 sample_shape=shape,
#                 sample_id=sample_id,
#                 # unstacked_sample_batch_shape=unstacked_sample_batch_shape,
#                 # unstacked_dim_order=unstacked_dim_order,
#                 inner_slice=inner_slice,
#                 # unstacked_batch_slice=unstacked_batch_slice,
#                 halo=halo,
#                 block_loc=i,  # TODO: update
#                 block_loc_shape=blocks_in_sample,
#             )
#         )
#     return np.asarray(out, dtype=object).reshape(out_shape)


# def split_multiple_shapes_into_blocks(
#     shapes: PerMember[PerAxis[int]],
#     block_shapes: PerMember[PerAxis[int]],
#     *,
#     halo: PerMember[PerAxis[HaloLike]],
#     broadcast: bool = False,
#     # sample_id: SampleId,
# ) -> tuple[BlockLocShape, PerMember[BlockArray]]:
#     """Split multiple shapes (e.g. sample member shapes) into aligned blocks.

#     Args:
#         shapes: the shapes of the sample members to split into blocks
#         block_shapes: the block shape per sample member (depending on `broaadcast` the block shapes must result in the same number of blocks for all sample members)
#         unstacked_batch_shape: the shape of the unstacked batch
#         unstacked_dim_order: the order of the dimensions of the unstacked tensor dimensions (unstacked batch dimensions plus the other tensor dimensions)
#         halo: the halo around each block per sample member
#         broadcast: whether to broadcast block shapes of members resulting in fewer blocks to match other sample members.
#     """

#     if unknown_blocks := [t for t in block_shapes if t not in shapes]:
#         raise ValueError(
#             f"block shape specified for unknown tensors: {unknown_blocks}."
#         )

#     if not block_shapes:
#         block_shapes = shapes

#     if not broadcast and (
#         missing_blocks := [t for t in shapes if t not in block_shapes]
#     ):
#         raise ValueError(
#             f"no block shape specified for {missing_blocks}."
#             + " Set `broadcast` to True if these tensors should be repeated"
#             + " as a whole for each block."
#         )

#     if extra_halo := [t for t in halo if t not in block_shapes]:
#         raise ValueError(
#             f"`halo` specified for tensors without block shape: {extra_halo}."
#         )

#     blocks: PerMember[BlockArray] = {}
#     block_locs: PerMember[BlockLocShape] = {}
#     for t in block_shapes:
#         block_locs[t], blocks[t] = split_shape_into_blocks(
#             shape=shapes[t],
#             # sample_id=sample_id,
#             # unstacked_batch_shape=unstacked_batch_shape,
#             # unstacked_dim_order=unstacked_dim_order[t]
#             # if unstacked_dim_order is not None
#             # else None,
#             block_shape=block_shapes[t],
#             halo=halo.get(t, {}),
#         )
#         assert all(b > 0 for b in block_locs[t]), block_locs

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
#     # def foo(*bs: BlockMeta) -> SampleBlockMeta:
#     #     return SampleBlockMeta.from_blocks(dict(zip(blocks.keys(), bs)))

#     # foo_op = np.frompyfunc(foo, len(blocks), 1)
#     # out = foo_op(*blocks.values())
#     # assert out.shape == next(iter(blocks.values())).shape
#     # return n, out


# # def _consolidate_blocks(blocks: PerMember[BlockMetaArray]) -> PerMember[BlockMetaArray]:
# #     sample_ids = {b.sample_id for b in blocks.values() if b.sample_id is not None}
# #     if len(sample_ids) > 1:
# #         raise ValueError(f"Mismatch for sample_id across members: {sample_ids}")
# #     block_loc_shapes = {b.block_loc_shape for b in blocks.values()}
# #     if len(block_loc_shapes) != 1:
# #         raise ValueError(
# #             f"Mismatch for block_loc_shape (sample shape in blocks) across members: {block_loc_shapes}"
# #         )
# #     block_locs = {b.block_loc for b in blocks.values()}
# #     if len(block_locs) != 1:
# #         raise ValueError(
# #             f"Mismatch for block_loc (block location within sample) across members: {block_locs}"
# #         )
# #     return blocks


# @dataclass
# class Chunk:
#     blocks: Iterable[PerMember[BlockMeta]]
#     n_blocks: int

#     def __iter__(self) -> Iterator[PerMember[BlockMeta]]:
#         yield from self.blocks


# @dataclass
# class Shard:
#     chunks: Iterable[Chunk]
#     n_chunks: int

#     def __iter__(self) -> Iterator[Chunk]:
#         yield from self.chunks


# def split_multiple_shapes_into_shards(
#     shapes: PerMember[PerAxis[int]],
#     *,
#     block_shapes: PerMember[PerAxis[int]],
#     halo: PerMember[PerAxis[HaloLike]],
#     broadcast: bool = False,
#     item_size: int | None,
#     chunk_shapes: PerMember[PerAxis[int]] | None = None,
#     shard_shapes: PerMember[PerAxis[int]] | None = None,
# ) -> tuple[BlockLocShape, Iterable[Shard]]:
#     """Split sample member shapes into aligned blocks.

#     Shards/chunks will be aligned across all sample members, which might lead to sup-optimal chunking/sharding for some sample members, but simplifies keeping the blocks aligned across all sample members.

#     Args:
#         shapes: the shapes of the sample members to split into blocks
#         block_shapes: the block shape per sample member (depending on `broaadcast` the block shapes must result in the same number of blocks for all sample members)
#         halo: the halo around each block per sample member
#         broadcast: whether to broadcast block shapes of members resulting in fewer blocks to match other sample members.
#         item_size: the size in bytes of the largest data type of all members. This is used to determine shard and chunk sizes. May be None if `chunk_shapes` and `shard_shapes` are given.
#         chunk_shapes: the chunk shape for at least one sample member; must be a multiple of the respective block shape. May be None if `item_size` is given.
#         shard_shapes: the shard shape for at leasst one sample member; must be a multiple of the respective chunk shape. May be None if `item_size` is given.


#     Returns:
#         A tuple of the total number of blocks and an iterable of shards.

#     """

#     raise NotImplementedError("sharding is not yet implemented")
