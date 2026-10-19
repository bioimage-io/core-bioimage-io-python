from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from math import ceil, floor, prod
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    cast,
)

import dask.array as da
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import assert_never

from bioimageio.core._utils.type_hints import NestedList, cast_and_assert
from bioimageio.spec._internal.utils import SLOTS

from ._axis_annotations import PerAxisAnno
from ._settings import settings
from .axis import TIME_AXIS_ID, AxisId, AxisSize, PerAxis
from .common import (
    BlockLoc,
    Halo,
    HaloLike,
    MemberId,
    PadMode,
    PadWidth,
    PerMember,
    SliceInfo,
    SliceInfoWithOverlap,
)
from .tensor import Array, Tensor


@dataclass
class LinearAxisTransform:
    member: MemberId
    axis: AxisId
    scale: float
    offset: int

    def compute(self, s: int, round: Callable[[float], int] = floor) -> int:
        return round(s * self.scale) + self.offset


@dataclass(frozen=True, **SLOTS)
class BlockMeta:
    """A block of data within a sample

    Figure for illustration:
    The first 2d block (dashed) of a sample member (**bold**).
    The inner slice (thin) is expanded by a halo in both dimensions on both sides.
    The outer slice reaches from the sample member origin (0, 0) to the right halo point.

    ```
    first block (at the sample origin)
    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
    ╷ halo(left)                         ╷
    ╷         padding outside the sample ╷
    ╷  (0, 0)┏━━━━━━━━━━━━━━━━━┯━━━━━━━━━┯━━━➔
    ╷        ┃                 │         ╷  sample member
    ╷        ┃      inner      │  outer  ╷
    ╷        ┃      region     │  region ╷
    ╷        ┃      /slice     │  /slice ╷
    ╷        ┃                 │         ╷
    ╷        ┣─────────────────┘         ╷
    ╷        ┃   outer region/slice      ╷
    ╷        ┃               halo(right) ╷
    └ ─ ─ ─ ─┃─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
             ⬇
    ```

    Note:
    - Inner and outer slices are specified in sample member coordinates.
    - The outer_slice of a block at the sample edge may overlap by more than the
        halo with the neighboring block (the inner slices will not overlap though).

    """

    parent: PerAxis[int] | Tensor
    """the full tensor this block is a part of, or its shape if the full tensor is not available"""

    inner_slice: PerAxisAnno[SliceInfo | SliceInfoWithOverlap]
    """inner region (without halo) wrt the full tensor"""

    halo: PerAxisAnno[Halo]
    """halo enlarging the inner region to the block's sizes"""

    # block_loc: BlockLoc
    # """the block index within the full tensor"""

    # block_loc_shape: BlockLocShape  # TODO: rename block_loc_shape
    # """The full tensor shape in numbers of blocks"""

    @property
    def full_shape(self) -> PerAxis[int]:
        """shape of the full tensor this block is a part of"""
        if isinstance(self.parent, Tensor):
            return self.parent.tagged_shape
        else:
            return self.parent

    @property
    def shape(self) -> PerAxis[int]:
        """shape of the block (including halo)"""
        return MappingProxyType(
            {
                a: s.stop - s.start + (sum(self.halo[a]) if a in self.halo else 0)
                for a, s in self.inner_slice.items()
            }
        )

    @property
    def padding(self) -> PerAxis[PadWidth]:
        """padding to realize the halo at the sample edge
        where we cannot simply enlarge the inner slice"""
        return MappingProxyType(
            {
                a: p
                for a in self.inner_slice
                if (
                    p := PadWidth(
                        (
                            self.halo[a].left
                            - (self.inner_slice[a].start - self.outer_slice[a].start)
                            if a in self.halo
                            else 0
                        ),
                        (
                            self.halo[a].right
                            - (self.outer_slice[a].stop - self.inner_slice[a].stop)
                            if a in self.halo
                            else 0
                        ),
                    )
                )
                != PadWidth(0, 0)
            }
        )

    @property
    def outer_slice(self) -> PerAxis[SliceInfo]:
        """slice of the outer block (including halo, excluding padding) wrt the sample"""
        return MappingProxyType(
            {
                a: SliceInfo(
                    max(
                        0,
                        self.inner_slice[a].start
                        - (self.halo[a].left if a in self.halo else 0),
                    ),
                    min(
                        self.full_shape[a],
                        self.inner_slice[a].stop
                        + (self.halo[a].right if a in self.halo else 0),
                    ),
                )
                for a in self.inner_slice
            }
        )

    @property
    def inner_shape(self) -> PerAxis[int]:
        """axis lengths of the inner region (without halo)"""
        return MappingProxyType(
            {a: s.stop - s.start for a, s in self.inner_slice.items()}
        )

    @property
    def local_slice(self) -> PerAxis[SliceInfo]:
        """inner slice wrt the block, **not** the sample"""
        return MappingProxyType(
            {
                a: (
                    SliceInfo(
                        h.left,
                        h.left + self.inner_shape[a],
                    )
                    if (h := self.halo.get(a)) is not None
                    else SliceInfo(0, self.inner_shape[a])
                )
                for a in self.inner_slice
            }
        )

    @property
    def dims(self) -> tuple[AxisId, ...]:
        return tuple(self.inner_shape)

    @property
    def tagged_shape(self) -> PerAxis[int]:
        """alias for shape"""
        return self.shape

    @property
    def inner_slice_wo_overlap(self) -> PerAxis[SliceInfo]:
        """subslice of the inner slice, such that all `inner_slice_wo_overlap` can be
        stiched together trivially to form the original sample.

        This can also be used to calculate statistics
        without overrepresenting block edge regions."""
        return {
            a: SliceInfo(s.start_wo_overlap, s.stop_wo_overlap)
            if isinstance(s, SliceInfoWithOverlap)
            else SliceInfo(s.start, s.stop)
            for a, s in self.inner_slice.items()
        }

    @property
    def inner_shape_wo_overlap(self) -> PerAxis[int]:
        """axis lengths of the inner region (without halo and without overlap)"""
        return MappingProxyType(
            {a: s.stop - s.start for a, s in self.inner_slice_wo_overlap.items()}
        )

    def __post_init__(self):
        assert all(a in self.full_shape for a in self.inner_slice), (
            "block has axes not present in its full tensor shape"
        )

        assert all(a in self.inner_slice for a in self.halo), (
            "halo has axes not present in block"
        )

        if any(
            s > self.full_shape[a] + sum(self.padding.get(a, PadWidth(0, 0)))
            for a, s in self.shape.items()
        ):
            logger.warning(
                "block {} larger than its full tensor shape {}",
                self.shape,
                self.full_shape,
            )

    # def get_transformed(self, new_axes: PerAxis[LinearAxisTransform | int]) -> Self:
    #     assert BATCH_AXIS_ID not in new_axes, "cannot transform the batch axis"

    #     full_shape = {
    #         a: (trf if isinstance(trf, int) else trf.compute(self.full_shape[trf.axis]))
    #         for a, trf in new_axes.items()
    #     }
    #     assert full_shape.get(BATCH_AXIS_ID) == self.full_shape.get(BATCH_AXIS_ID), (
    #         "block transform must not change the batch axis size"
    #     )

    #     inner_slice = {
    #         a: (
    #             SliceInfo(0, trf)
    #             if isinstance(trf, int)
    #             else SliceInfo(
    #                 trf.compute(self.inner_slice[trf.axis].start),
    #                 trf.compute(self.inner_slice[trf.axis].stop),
    #             )
    #         )
    #         for a, trf in new_axes.items()
    #     }
    #     assert inner_slice.get(BATCH_AXIS_ID) == self.inner_slice.get(BATCH_AXIS_ID), (
    #         "block transform must not change the batch axis slice"
    #     )

    #     return self.__class__(
    #         parent=full_shape,
    #         inner_slice=inner_slice,
    #         halo={
    #             a: (
    #                 Halo(0, 0)
    #                 if isinstance(trf, int)
    #                 else Halo(self.halo[trf.axis].left, self.halo[trf.axis].right)
    #             )
    #             for a, trf in new_axes.items()
    #         },
    #     )

    # def get_transformed(
    #     self, transformed_tensor: Tensor, new_axes: PerAxis[LinearAxisTransform | int]
    # ) -> Self:
    #     assert BATCH_AXIS_ID not in new_axes, "cannot transform the batch axis"

    #     full_shape = {
    #         a: (trf if isinstance(trf, int) else trf.compute(self.full_shape[trf.axis]))
    #         for a, trf in new_axes.items()
    #     }
    #     assert full_shape.get(BATCH_AXIS_ID) == self.full_shape.get(BATCH_AXIS_ID), (
    #         "block transform must not change the batch axis size"
    #     )

    #     inner_slice = {
    #         a: (
    #             SliceInfo(0, trf)
    #             if isinstance(trf, int)
    #             else SliceInfo(
    #                 trf.compute(self.inner_slice[trf.axis].start),
    #                 trf.compute(self.inner_slice[trf.axis].stop),
    #             )
    #         )
    #         for a, trf in new_axes.items()
    #     }
    #     assert inner_slice.get(BATCH_AXIS_ID) == self.inner_slice.get(BATCH_AXIS_ID), (
    #         "block transform must not change the batch axis slice"
    #     )

    #     return self.__class__(
    #         full_tensor=transformed_tensor,
    #         inner_slice=inner_slice,
    #         halo={
    #             a: (
    #                 Halo(0, 0)
    #                 if isinstance(trf, int)
    #                 else Halo(self.halo[trf.axis].left, self.halo[trf.axis].right)
    #             )
    #             for a, trf in new_axes.items()
    #         },
    #         block_loc=self.block_loc,
    #         block_loc_shape=self.block_loc_shape,
    #     )


@dataclass(frozen=True, **SLOTS)
class Block(BlockMeta):
    """A block of data within a sample

    Figure for illustration:
    The first 2d block (dashed) of a sample member (**bold**).
    The inner slice (thin) is expanded by a halo in both dimensions on both sides.
    The outer slice reaches from the sample member origin (0, 0) to the right halo point.

    ```
    first block (at the sample origin)
    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
    ╷ halo(left)                         ╷
    ╷         padding outside the sample ╷
    ╷  (0, 0)┏━━━━━━━━━━━━━━━━━┯━━━━━━━━━┯━━━➔
    ╷        ┃                 │         ╷  sample member
    ╷        ┃      inner      │  outer  ╷
    ╷        ┃      region     │  region ╷
    ╷        ┃      /slice     │  /slice ╷
    ╷        ┃                 │         ╷
    ╷        ┣─────────────────┘         ╷
    ╷        ┃   outer region/slice      ╷
    ╷        ┃               halo(right) ╷
    └ ─ ─ ─ ─┃─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
             ⬇
    ```

    Note:
    - Inner and outer slices are specified in sample member coordinates.
    - The outer_slice of a block at the sample edge may overlap by more than the
        halo with the neighboring block (the inner slices will not overlap though).

    """

    parent: Tensor
    """the full tensor this block is a part of"""

    pad_mode: PadMode | Literal["none"]
    """The padding mode to use when the halo extends beyond the tensor boundary.

    A ValueError is raised when the halo extends beyond the tensor boundary and `pad_mode` is "none"."""

    def __post_init__(self):
        assert not self.padding or self.pad_mode != "none", (
            "block requires padding but `pad_mode` is 'none'"
        )

    @property
    def outer_tensor(self) -> Tensor:
        """the outer tensor of the block (including halo and possibly padding)"""
        outer_tensor = self.parent[self.outer_slice]
        if self.padding:
            if self.pad_mode == "none":
                raise ValueError("block requires padding but `pad_mode` is 'none'")
            outer_tensor = outer_tensor.pad(self.padding, mode=self.pad_mode)

        return outer_tensor


BlockArray = Annotated[NDArray[np.object_], "NDArray[Block]"]
# BlockMetaArray = Annotated[NDArray[np.object_], "NDArray[BlockMeta]"]


def split_tensor_into_blocks(
    tensor: Tensor,
    *,
    block_shape: PerAxis[int | AxisSize],
    halo: PerAxis[HaloLike],
    pad_mode: PadMode | Literal["none"],
) -> BlockArray:
    """
    Args:
        item_size: size of a single element in bytes, used to find shard and chunk sizes
    """
    if unknown_dims := [
        a for a in block_shape if a not in tensor.dims and block_shape[a] != 1
    ]:
        raise ValueError(
            f"unknown dims in block_shape: {unknown_dims} for tensor {tensor.dims}"
        )

    if unknown_halo_dims := [a for a in halo if a not in tensor.dims]:
        raise ValueError(
            f"unknown dims in halo: {unknown_halo_dims} for tensor {tensor.dims}"
        )

    # fill in default halo (0)
    halo = {d: Halo.create(halo.get(d, 0)) for d in tensor.dims}

    main_block_shape = {
        d: (
            bs_limit := min(
                s + (s_halo := 0 if pad_mode == "none" else sum(halo[d])),
                bs.max or s + s_halo,
            )
        )
        - ((bs_limit - bs.min) % (bs.step or 1))
        if isinstance((bs := block_shape.get(d, s)), AxisSize)
        else bs
        for d, s in tensor.tagged_shape.items()
    }

    if large_block_dims := {
        a
        for a, bs in block_shape.items()
        if tensor.tagged_shape.get(a, 1)
        < (bs.max or bs.min if isinstance(bs, AxisSize) else bs)
    }:
        raise ValueError(
            "Block shape {} larger than tensor shape {} along {}",
            block_shape,
            tensor.tagged_shape,
            large_block_dims,
        )

    inner_1d_slices: dict[AxisId, list[SliceInfo | SliceInfoWithOverlap]] = {}
    inner_sizes = {d: main_block_shape[d] - sum(halo[d]) for d in tensor.dims}
    for d in tensor.dims:
        inner_first_start = halo[d].left if pad_mode == "none" else 0
        inner_last_stop = tensor.tagged_shape[d] - (
            halo[d].right if pad_mode == "none" else 0
        )
        inner_size = inner_sizes[d]
        outer_size = block_shape.get(d, tensor.tagged_shape[d] + sum(halo[d]))
        if isinstance(outer_size, int):
            outer_size = AxisSize(min=outer_size)

        outer_min = outer_size.min
        step = outer_size.step or 1
        assert inner_size % step == 0

        inner_1d_slices[d] = []
        for start in range(inner_first_start, inner_last_stop, inner_size):
            if start + inner_size <= inner_last_stop:
                inner_1d_slices[d].append(SliceInfo(start, start + inner_size))
            else:
                inner_remainder = inner_last_stop - start
                outer_remainder = inner_remainder + sum(halo[d])
                if (
                    outer_remainder >= outer_min
                    and (outer_remainder - outer_min) % step == 0
                ):
                    # last block has valid shape
                    inner_1d_slices[d].append(SliceInfo(start, inner_last_stop))
                else:
                    # last block needs to overlap to have a valid shape
                    minimum_outer_remainder = outer_min + (
                        step
                        * max(
                            0,
                            ceil((outer_remainder - outer_min) / step),
                        )
                    )
                    valid_inner_start = inner_last_stop - (
                        minimum_outer_remainder - sum(halo[d])
                    )
                    inner_1d_slices[d].append(
                        SliceInfoWithOverlap(
                            valid_inner_start, inner_last_stop, start, inner_last_stop
                        )
                    )

    out_shape = tuple(len(slices) for slices in inner_1d_slices.values())
    out: BlockArray = np.empty(out_shape, dtype=object)
    for idx in itertools.product(*(range(s) for s in out_shape)):
        block_loc: BlockLoc = idx
        inner_slice = {d: inner_1d_slices[d][i] for d, i in zip(tensor.dims, block_loc)}
        out[idx] = Block(
            parent=tensor, inner_slice=inner_slice, halo=halo, pad_mode=pad_mode
        )

    return out


def split_multiple_tensors_into_blocks(
    tensors: PerMember[Tensor],
    block_shape: PerMember[PerAxis[int | AxisSize]],
    *,
    halo: PerMember[PerAxis[HaloLike]],
    broadcast: bool = False,
    pad_mode: PadMode | PerMember[PadMode] | Literal["none"],
) -> dict[MemberId, BlockArray]:
    """Split multiple tensors (e.g. sample members) into aligned blocks.

    Args:
        tensors: the tensors to split into blocks
        block_shape: the block shape per sample member (depending on `broaadcast` the block shapes must result in the same number of blocks for all tensors).
            Omitted axes are treated as having a block size equal to the tensor axis size.
            To allow for variable block sizes along a given axis, the block size can be specified as an `AxisSize` with a `min` `max` and `step` value.
            Any such parameterized axis may result in blocks overlapping along that axis, if the tensor shape does not divide by the maximum block size with a valid remainder.
        halo: the halo around each block per sample member, omitted axes are treated as having a halo of 0.
        broadcast: whether to allow broadcasting the resulting tensor blocks to each other, e.g. if one tensor does not have a dimension along which another is blocked up.
        pad_mode: the padding mode (optionally per sample member) to use when the halo extends beyond the tensor boundary.
    Raises:
        ValueError: If the tensor shape cannot be divided into blocks according to `block_shape`.
    """

    if unknown_blocks := [m for m in block_shape if m not in tensors]:
        raise ValueError(
            f"block shape specified for unknown tensors: {unknown_blocks}."
        )

    if not block_shape:
        block_shape = {m: t.tagged_shape for m, t in tensors.items()}

    if not broadcast and (
        missing_blocks := [m for m in tensors if m not in block_shape]
    ):
        raise ValueError(
            f"no block shape specified for {missing_blocks}."
            + " Set `broadcast` to True if these tensors should be repeated"
            + " as a whole for each block."
        )

    if extra_halo := [m for m in halo if m not in block_shape]:
        raise ValueError(
            f"`halo` specified for tensors without block shape: {extra_halo}."
        )

    blocks: PerMember[BlockArray] = {}
    for m in block_shape:
        blocks[m] = split_tensor_into_blocks(
            tensor=tensors[m],
            block_shape=block_shape[m],
            halo=halo.get(m, {}),
            pad_mode=pad_mode.get(m, "symmetric")
            if isinstance(pad_mode, Mapping)
            else pad_mode,
        )

    assert len(blocks) > 0, blocks
    unique_block_ndims = {b.shape for b in blocks.values()}

    if len(unique_block_ndims) != 1:
        if not broadcast:
            raise ValueError(
                f"Mismatch for block layout across members: { {t: b.shape for t, b in blocks.items()} }. Set `broadcast` to True if this is expected."
            )

        blocks = dict(zip(blocks.keys(), np.broadcast_arrays(*blocks.values())))
        block_locs = {m: b.shape for m, b in blocks.items()}
        unique_block_locs = set(block_locs.values())
        assert len(unique_block_locs) == 1, (
            f"Mismatch for total number of blocks (after braodcasting?!): {block_locs}"
        )

    return blocks


def apply_blockwise(
    func: Callable[
        [PerMember[Tensor]],
        PerMember[Tensor | None],  # TODO: return non-optional outputs only?
    ],
    inputs: PerMember[Tensor],
    *,
    block_shape: PerMember[PerAxis[int | AxisSize]],
    pad_mode: PadMode | PerMember[PadMode] | Literal["none"],
    input_halo: PerMember[PerAxis[Halo]],
    output_halo: PerMember[PerAxis[Halo]],
    get_output_shape: Callable[[PerMember[PerAxis[int]]], PerMember[PerAxis[int]]],
    chunk_shape: PerMember[PerAxis[int]] | int = settings.processing_chunk_min_bytes,
    blockwise_processing_mode: Literal["parallel", "sequential", "auto"] = "auto",
) -> dict[MemberId, Tensor]:
    """Apply `func` to `inputs` blockwise.


    Args:
        func: the function to apply to each block
        inputs: the input tensors to process
        block_shape: the shape of the blocks to be processed by func. May be expressed using `AxisSize` objects to allow for variable block shapes.
        pad_mode: the padding mode (optionally per input member) to use when the halo extends beyond the tensor boundary
            if "none" and `output_halo` is not empty, the output will be smaller by `output_halo`
        input_halo: the halo around each input block (e.g. to compenstate for `output_halo` such that cropped output blocks can be stitched together)
            Within the sample halo is realized as overlap between neighboring blocks, outside the sample halo is realized as padding.
        output_halo: the halo to be cropped from each output block
        get_output_shape: a function to compute the output shape based on the input shape.
            May return values <1 for output axes without halo, which means the output will be concatenated along those axes.
            (Only used if the input shape does not evenly divide by the block shape to correct overlapping outputs.)
        chunk_shape: the shape of the chunks to process.
            Chunks are larger than blocks and help to reduce the number of tasks in the dask graph for 'parallel' processing.
            Must be larger than `block_shape`.
            If an integer is given, it is interpreted as the minimum number of bytes of any input chunk tensor.
        blockwise_processing_mode: (WIP) processing mode wihin a (parallelly processed) dask chunk.
            Options are
            - 'parallel': nest blockwise tasks inside the chunk task)
            - 'sequential': (not yet implemented) full chunk is a single task
            - 'auto': automatically choose between 'parallel' and 'sequential' based on the number of blocks per chunk and the total number of chunks to avoid large task graphs.
    """
    if blockwise_processing_mode == "auto":
        blockwise_processing_mode = "parallel"  # TODO: implement auto mode
        logger.info("using blockwise_processing_mode='parallel' for ")
    elif blockwise_processing_mode == "sequential":
        raise NotImplementedError(
            "blockwise_processing_mode='sequential' is not yet implemented"
        )

    max_block_shape: dict[MemberId, dict[AxisId, int]] = {}
    for m, b in block_shape.items():
        max_block_shape[m] = {}
        for d, bs in b.items():
            if isinstance(bs, AxisSize):
                max_tensor_size = inputs[m].tagged_shape.get(d, 1) + sum(
                    input_halo.get(m, {}).get(d, Halo(0, 0))
                )
                effective_bs = max(
                    bs.min,
                    max_tensor_size if bs.max is None else min(bs.max, max_tensor_size),
                )
                if (
                    not isinstance(chunk_shape, int)
                    and m in chunk_shape
                    and d in chunk_shape[m]
                ):
                    effective_bs = min(effective_bs, chunk_shape[m][d])
            else:
                effective_bs = bs

            max_block_shape[m][d] = effective_bs

    if isinstance(chunk_shape, int):
        chunk_shape = get_chunk_shape(
            inputs, block_shape=max_block_shape, chunk_min_bytes=chunk_shape
        )

    if any(
        max_block_shape[m][d] > chunk_shape[m][d]
        for m in chunk_shape
        for d in chunk_shape[m]
        if m in max_block_shape and d in max_block_shape[m]
    ):
        raise ValueError(
            f"block_shape {block_shape} must not be larger than chunk_shape {chunk_shape}"
        )

    process_chunk = partial(
        _apply_blockwise_impl,
        func,
        block_shape=block_shape,
        pad_mode="none",
        input_halo=input_halo,
        output_halo=output_halo,
        processing_mode=blockwise_processing_mode,
        get_output_shape=get_output_shape,
    )

    return _apply_blockwise_impl(
        process_chunk,
        inputs,
        block_shape=chunk_shape,
        pad_mode=pad_mode,
        input_halo=input_halo,
        output_halo=output_halo,
        processing_mode="parallel",
        get_output_shape=get_output_shape,
    )


def _apply_blockwise_impl(
    func: Callable[[PerMember[Tensor]], PerMember[Tensor | None]],
    inputs: PerMember[Tensor],
    *,
    block_shape: PerMember[PerAxis[int | AxisSize]],
    pad_mode: PadMode | PerMember[PadMode] | Literal["none"],
    input_halo: PerMember[PerAxis[HaloLike]],
    output_halo: PerMember[PerAxis[HaloLike]],
    get_output_shape: Callable[[PerMember[PerAxis[int]]], PerMember[PerAxis[int]]],
    processing_mode: Literal["parallel", "sequential"] = "parallel",
) -> dict[MemberId, Tensor]:
    """
    Apply a function to a mapping (dict) of tensors, processing them in blocks.

    Args:
        func: the function to apply to each block
        inputs: the input tensors to process
        block_shape: the shape of the blocks to be processed by func. May be expressed using `AxisSize` objects to allow for variable block shapes.
        pad_mode: the padding mode (optionally per input member) to use when the halo extends beyond the tensor boundary
            if "none" and `output_halo` is not empty, the output will be smaller by the `output_halo`
        input_halo: the halo around each input block (e.g. to compenstate for `output_halo` such that cropped output blocks can be stitched together)
            Within the sample halo is realized as overlap between neighboring blocks, outside the sample halo is realized as padding.
        output_halo: the halo to be cropped from each output block
        get_output_shape: a function to compute the output shape based on the input shape
            (only used if the input shape does not evenly divide by the block shape to correct overlapping outputs)
        processing_mode: the mode of blockwise processing ('parallel' (via dask graph) or 'sequential')
    """
    input_blocks = split_multiple_tensors_into_blocks(
        inputs,
        block_shape=block_shape,
        halo=input_halo,
        broadcast=True,
        pad_mode=pad_mode,
    )

    return apply_to_block_array(
        func,
        input_blocks,
        output_halo=output_halo,
        keep_output_halo_at_sample_edge=pad_mode == "none",
        processing_mode=processing_mode,
        get_output_shape=get_output_shape,
    )


def apply_to_block_array(
    func: Callable[[PerMember[Tensor]], PerMember[Tensor | None]],
    input_blocks: PerMember[BlockArray],
    *,
    output_halo: PerMember[PerAxis[HaloLike]],
    keep_output_halo_at_sample_edge: bool,
    processing_mode: Literal["parallel", "sequential"] = "parallel",
    get_output_shape: Callable[[PerMember[PerAxis[int]]], PerMember[PerAxis[int]]],
) -> dict[MemberId, Tensor]:

    nested_outputs = apply_to_blocks(
        func,
        input_blocks,
        output_halo=output_halo,
        get_output_shape=get_output_shape,
        keep_output_halo_at_sample_edge=keep_output_halo_at_sample_edge,
    )

    def select_arrays(
        nested: NestedList[dict[MemberId, Tensor]], member: MemberId
    ) -> NestedList[NDArray[Any] | da.Array]:
        if isinstance(nested, list):
            return [select_arrays(item, member) for item in nested]
        else:
            return nested[member].data.data

    first_block = nested_outputs
    while isinstance(first_block, list):
        first_block = first_block[0]

    if processing_mode == "parallel":
        block_impl = cast(Callable[[NestedList[Array]], da.Array], da.block)
    elif processing_mode == "sequential":
        block_impl = cast(Callable[[NestedList[Array]], NDArray[Any]], np.block)
    else:
        assert_never(processing_mode)

    return {
        output_id: Tensor(
            block_impl(
                select_arrays(nested_outputs, output_id),
            ),
            dims=first_block[output_id].dims,
        )
        for output_id in first_block
    }


def apply_to_blocks(
    func: Callable[[PerMember[Tensor]], PerMember[Tensor | None]],
    blocks: PerMember[BlockArray],
    *,
    output_halo: PerMember[PerAxis[HaloLike]],
    get_output_shape: Callable[[PerMember[PerAxis[int]]], PerMember[PerAxis[int]]],
    keep_output_halo_at_sample_edge: bool,
) -> NestedList[dict[MemberId, Tensor]]:
    if not all(b.size > 0 for b in blocks.values()):
        raise ValueError(
            "Sample is not split into blocks. Please provide `block_shapes` or split the sample first."
        )

    block_ndims = {m: b.shape for m, b in blocks.items()}
    if len(set(block_ndims.values())) != 1:
        raise ValueError(f"Input block grids do not match: {block_ndims}")

    block_grid_shape = next(iter(block_ndims.values()))
    first_block = cast(Sequence[Block], next(iter(blocks.values())).flat)[0]
    block_grid_dims = first_block.dims

    output_halo = {
        m: {a: Halo.create(h) for a, h in h_per_a.items()}
        for m, h_per_a in output_halo.items()
    }

    def func_blocks(
        axis: int = 0, loc: tuple[int, ...] = ()
    ) -> NestedList[dict[MemberId, Tensor]]:
        if axis < len(block_grid_shape):
            grid_size = block_grid_shape[axis]
            return [func_blocks(axis + 1, (*loc, i)) for i in range(grid_size)]

        left_input_overlap = {
            a
            for b in blocks.values()
            for a, s in cast_and_assert(Block, b[loc]).inner_slice.items()
            if isinstance(s, SliceInfoWithOverlap) and s.start_wo_overlap != s.start
        }
        right_input_overlap = {
            a
            for b in blocks.values()
            for a, s in cast_and_assert(Block, b[loc]).inner_slice.items()
            if isinstance(s, SliceInfoWithOverlap) and s.stop_wo_overlap != s.stop
        }
        if right_input_overlap:
            # we currently do not use overlap to the right...
            raise NotImplementedError(
                "Output block overlap to the right is not implemented."
            )

        valid_input_shape = {
            m: b[loc].inner_shape_wo_overlap for m, b in blocks.items()
        }
        valid_output_shape = get_output_shape(valid_input_shape)
        input_members = {m: b[loc].outer_tensor for m, b in blocks.items()}
        ret: dict[MemberId, Tensor] = {}
        for m, t in func(input_members).items():
            if t is None:
                continue

            # crop halo to keep inner output
            keep_halo_here: dict[AxisId, Halo] = {}
            if (crop_halo_here := output_halo.get(m)) is not None:
                crop_halo_here = dict(crop_halo_here)
                if keep_output_halo_at_sample_edge:
                    assert len(block_grid_dims) == len(block_grid_shape) == len(loc)
                    for a, grid_size, loc_idx in zip(
                        block_grid_dims, block_grid_shape, loc
                    ):
                        if a not in crop_halo_here:
                            continue

                        if loc_idx == 0 and loc_idx == grid_size - 1:
                            # single block is first and last block
                            keep_halo_here[a] = crop_halo_here[a]
                            crop_halo_here[a] = Halo(0, 0)
                        elif loc_idx == 0:
                            # first block
                            keep_halo_here[a] = Halo(crop_halo_here[a].left, 0)
                            crop_halo_here[a] = Halo(0, crop_halo_here[a].right)
                        elif loc_idx == grid_size - 1:
                            # last block
                            keep_halo_here[a] = Halo(0, crop_halo_here[a].right)
                            crop_halo_here[a] = Halo(crop_halo_here[a].left, 0)

                t = t.crop_halo(crop_halo_here)

            overlap_halo = {
                a: Halo(overlap, 0)
                for a, vs in valid_output_shape[m].items()
                if (
                    overlap := t.tagged_shape[a]
                    - sum(
                        keep_halo_here.get(a, Halo(0, 0))
                    )  # consider the edge halo we explicitly keep to be valid, so we do not error consider it overlap
                    - vs
                )
                != 0
            }
            assert set(overlap_halo) == left_input_overlap, (
                f"need to crop overlap {overlap_halo} but expected overlap in {left_input_overlap}"
            )
            t = t.crop_halo(overlap_halo)

            ret[m] = t

        return ret

    return func_blocks()


def get_chunk_shape(
    tensors: PerMember[Tensor],
    block_shape: PerMember[PerAxis[int]],
    chunk_min_bytes: int = settings.processing_chunk_min_bytes,
) -> PerMember[PerAxis[int]]:
    if unknown_members := [m for m in block_shape if m not in tensors]:
        raise ValueError(
            f"Block shape specified for unknown tensors: {unknown_members}."
        )

    # default to 1 for missing block dimensions, and map -1 to tensor shape
    block_shape = {
        m: {
            d: t.tagged_shape[d]
            if (b_s := block_shape.get(m, {}).get(d, 1)) == -1
            else b_s
            for d in t.dims
        }
        for m, t in tensors.items()
    }
    dim_sizes = {d: d_s for b_s in block_shape.values() for d, d_s in b_s.items()}

    if not any(
        a in tensors[m].tagged_shape for m, b_s in block_shape.items() for a in b_s
    ):
        raise ValueError(
            f"Block shape {block_shape} has dimensions not present in sample shape { {m: tensors[m].tagged_shape for m in tensors} }"
        )

    chunk_shape = deepcopy(block_shape)
    item_sizes = {m: np.dtype(t.dtype).itemsize for m, t in tensors.items()}

    def get_max_chunk_size():
        return max(prod(chunk_shape[m].values()) * item_sizes[m] for m in chunk_shape)

    def increment_chunk_dims(incr_dims: Sequence[AxisId]):
        incr_dims = list(incr_dims)
        while incr_dims:
            for a in incr_dims:
                if get_max_chunk_size() >= chunk_min_bytes:
                    return
                for m in tensors:
                    chunk_shape[m][a] = min(
                        tensors[m].tagged_shape[a],
                        chunk_shape[m][a] + block_shape[m][a],
                    )

                    if all(
                        chunk_shape[m][a] == tensors[m].tagged_shape[a]
                        for m in chunk_shape
                        if a in chunk_shape[m]
                    ):
                        incr_dims.remove(a)
                        break

    # increment chunk dimensions in order of non-singleton spatial, time, other, then any singleton dim
    non_singleton_spatial_dims = [
        d for d in (AxisId("x"), AxisId("y"), AxisId("z")) if dim_sizes.get(d, 0) > 1
    ]
    increment_chunk_dims(non_singleton_spatial_dims)

    non_singleton_time_dims = [d for d in (TIME_AXIS_ID,) if dim_sizes.get(d, 0) > 1]
    increment_chunk_dims(non_singleton_time_dims)

    other_non_singleton_dims = [d for d, s in dim_sizes.items() if s > 1]
    increment_chunk_dims(other_non_singleton_dims[::-1])

    singleton_dims = [d for d, s in dim_sizes.items() if s <= 1]
    increment_chunk_dims(singleton_dims[::-1])

    logger.info(
        "Calculated chunk shape {} ({} {}) for block shape {} and sample shape {}",
        chunk_shape,
        c
        if (c := get_max_chunk_size()) < 1024
        else c // 1024
        if c < 1024**2
        else c // 1024**2
        if c < 1024**3
        else c // 1024**3,
        "B" if c < 1024 else "KiB" if c < 1024**2 else "MiB" if c < 1024**3 else "GiB",
        block_shape,
        {m: dict(t.tagged_shape) for m, t in tensors.items()},
    )
    return chunk_shape
