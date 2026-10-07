from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, cast

import dask.array as da

from bioimageio.core._utils.type_hints import NestedList

from ..axis import PerAxis
from ..common import HaloLike, MemberId, PadMode, PerMember
from ..tensor import Tensor


def chunk_tensors(
    tensors: PerMember[Tensor],
    chunk_shapes: PerMember[PerAxis[int]],
    *,
    halo: PerMember[PerAxis[HaloLike]] | None = None,
    pad_mode: PadMode | PerMember[PadMode] | Literal["none"] = "none",
) -> dict[MemberId, Tensor]:
    if unknown_chunks := [m for m in chunk_shapes if m not in tensors]:
        raise ValueError(
            f"chunk sizes specified for unknown tensors: {unknown_chunks}."
        )
    return {
        m: t.chunk(
            chunk_shapes[m],
            halo=halo.get(m) if halo is not None else None,
            pad_mode=pad_mode
            if isinstance(pad_mode, str)
            else pad_mode.get(m, "none")
            if isinstance(pad_mode, Mapping)
            else pad_mode,
        )
        for m, t in tensors.items()
    }


def chunkwise_to_nested_list_prechunked(
    tensors: PerMember[Tensor], func: Callable[[PerMember[Tensor]], PerMember[Tensor]]
) -> NestedList[PerMember[Tensor]]:
    """Apply a function to each chunk of the chunked tensors and return a nested list of processed chunks.

    See [chunk_tensors][] for chunking the tensors beforehand.

    Args:
        func: The function to apply to each sample chunk.
    """

    if not all(t.data.data.numblocks for t in tensors.values()):
        raise ValueError(
            "Sample is not chunked and no `chunk_shapes` provided. Please provide `chunk_shapes` or chunk the sample first."
        )

    chunk_ndims = cast(
        PerMember[tuple[int, ...]],
        {m: t.data.data.numblocks for m, t in tensors.items()},
    )
    if len(set(chunk_ndims.values())) != 1:
        raise ValueError(f"Input chunk grids do not match: {chunk_ndims}")

    chunk_grid_shape = next(iter(chunk_ndims.values()))

    def func_chunks(
        axis: int = 0, loc: tuple[int, ...] = ()
    ) -> NestedList[PerMember[Tensor]]:
        if axis < len(chunk_grid_shape):
            grid_size = chunk_grid_shape[axis]
            return [func_chunks(axis + 1, (*loc, i)) for i in range(grid_size)]

        return func(
            {
                m: Tensor(t.data.data.blocks[loc], dims=t.dims)
                for m, t in tensors.items()
            }
        )

    return func_chunks()


# def blockwise_to_nested_list(
#     tensors: PerMember[Tensor], func: Callable[[PerMember[Tensor]], PerMember[Tensor]], block_shapes: PerMember[PerAxis[int]], halo: PerMember[PerAxis[HaloLike]], pad_mode: PadMode | PerMember[PadMode]
# ) -> NestedList[PerMember[Tensor]]:
#     """Apply a function to each block and return a nested list of processed blocks.
#     Each block will have the same shape, the last two blocks along each dimension may overlap by more than the halo.

#     See [chunk_tensors][] for chunking the tensors beforehand.

#     Args:
#         func: The function to apply to each sample chunk.
#     """

#     if not all(t.data.data.numblocks for t in tensors.values()):
#         raise ValueError(
#             "Sample is not chunked and no `chunk_shapes` provided. Please provide `chunk_shapes` or chunk the sample first."
#         )

#     chunk_ndims = cast(
#         PerMember[tuple[int, ...]],
#         {m: t.data.data.numblocks for m, t in tensors.items()},
#     )
#     if len(set(chunk_ndims.values())) != 1:
#         raise ValueError(f"Input chunk grids do not match: {chunk_ndims}")

#     chunk_grid_shape = next(iter(chunk_ndims.values()))

#     def func_chunks(
#         axis: int = 0, loc: tuple[int, ...] = ()
#     ) -> NestedList[PerMember[Tensor]]:
#         if axis < len(chunk_grid_shape):
#             grid_size = chunk_grid_shape[axis]
#             return [func_chunks(axis + 1, (*loc, i)) for i in range(grid_size)]

#         return func(
#             {
#                 m: Tensor(t.data.data.blocks[loc], dims=t.dims)
#                 for m, t in tensors.items()
#             }
#         )

#     return func_chunks()


def chunkwise(
    tensors: PerMember[Tensor], func: Callable[[PerMember[Tensor]], PerMember[Tensor]]
) -> dict[MemberId, Tensor]:
    """Apply a function to each chunk of the chunked tensors.

    See [chunk_tensors][] for chunking the tensors beforehand.

    Args:
        func: The function to apply to each sample chunk.
    """
    nested_outputs = chunkwise_to_nested_list_prechunked(tensors, func)

    def select_output(
        nested: NestedList[PerMember[Tensor]], output_id: MemberId
    ) -> NestedList[da.Array]:
        if isinstance(nested, list):
            return [select_output(item, output_id) for item in nested]
        else:
            a = nested[output_id].data.data
            assert isinstance(a, da.Array)
            return a

    first_block = nested_outputs
    while isinstance(first_block, list):
        first_block = first_block[0]

    return {
        output_id: Tensor(
            da.block(
                select_output(nested_outputs, output_id),
            ),
            dims=first_block[output_id].dims,
        )
        for output_id in first_block
    }
