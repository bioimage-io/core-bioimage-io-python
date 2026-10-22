from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Callable,
    Generic,
    TypeVar,
    cast,
)

import dask.array as da
import numpy as np

from bioimageio.spec.model import v0_5

from ._utils.type_hints import NestedList
from .axis import AxisSize, PerAxis
from .block import Block, split_multiple_tensors_into_blocks
from .common import HaloLike, MemberId, PadMode, PerMember
from .sample import Sample
from .tensor import Array, Tensor

SerializedSampleBlockType = TypeVar("SerializedSampleBlockType")


class SampleSerializer(ABC, Generic[SerializedSampleBlockType]):
    @classmethod
    def serialize_sample(
        cls,
        sample: Sample,
    ) -> NestedList[SerializedSampleBlockType]:
        """Serialize a sample as a single block."""

        return cls.serialize_sample_block(sample)

    @classmethod
    def deserialize_sample(
        cls,
        serialized: NestedList[SerializedSampleBlockType],
    ) -> Sample:
        """Assemble a nested block grid; blocks must already be cropped to avoid overlap."""

        if not isinstance(serialized, list):
            return cls.deserialize_sample_block(serialized)

        leaves: list[Sample] = []

        def deserialize(
            grid: NestedList[SerializedSampleBlockType],
        ) -> NestedList[Sample]:
            if isinstance(grid, list):
                if not grid:
                    raise ValueError("Serialized sample contains an empty block grid")
                return [
                    deserialize(item)
                    for item in cast(list[NestedList[SerializedSampleBlockType]], grid)
                ]

            sample = cls.deserialize_sample_block(grid)
            leaves.append(sample)
            return sample

        samples = deserialize(cast(NestedList[SerializedSampleBlockType], serialized))
        first = leaves[0]
        for sample in leaves[1:]:
            if sample.id != first.id:
                raise ValueError("Sample blocks have different sample IDs")
            if sample.members.keys() != first.members.keys():
                raise ValueError("Sample blocks have different members")
            if any(sample.members[m].dims != t.dims for m, t in first.members.items()):
                raise ValueError("Sample blocks have different member dimensions")

        def select_arrays(
            grid: NestedList[Sample], member: MemberId
        ) -> NestedList[Array]:
            if isinstance(grid, list):
                return [select_arrays(item, member) for item in grid]
            return grid.members[member].data.data

        members: dict[MemberId, Tensor] = {}
        for member, tensor in first.members.items():
            block_impl = cast(
                Callable[[NestedList[Array]], Array],
                da.block
                if any(
                    isinstance(s.members[member].data.data, da.Array) for s in leaves
                )
                else np.block,
            )
            members[member] = Tensor(
                block_impl(select_arrays(samples, member)), tensor.dims
            )
            members[member].set_metadata(tensor.get_metadata())

        return Sample(id=first.id, members=members, stat=first.stat)

    @classmethod
    def serialize_sample_blockwise(
        cls,
        sample: Sample,
        *,
        model: v0_5.ModelDescr,
        blocksize_parameter: int,
        batch_size: int = 1,
    ) -> NestedList[SerializedSampleBlockType]:
        """Serialize a non-overlapping grid with model-derived maximum block sizes.

        Edge blocks may be smaller so deserialization reconstructs the original sample.
        """

        block_shape = model.get_tensor_sizes(
            blocksize_parameter, batch_size, sample.shape
        ).inputs
        return cls.serialize_sample_with_fixed_blocking(
            sample,
            block_shapes={
                m: {a: AxisSize(min=1, max=size) for a, size in shape.items()}
                for m, shape in block_shape.items()
            },
            halo={},
        )

    @classmethod
    def serialize_sample_with_fixed_blocking(
        cls,
        sample: Sample,
        *,
        block_shapes: PerMember[PerAxis[int | AxisSize]],
        halo: PerMember[PerAxis[HaloLike]],
        pad_mode: PadMode | PerMember[PadMode] = "symmetric",
    ) -> NestedList[SerializedSampleBlockType]:
        """Serialize an aligned block grid, retaining each block's halo and overlap."""

        blocks = split_multiple_tensors_into_blocks(
            sample.members,
            block_shape=block_shapes,
            halo=halo,
            pad_mode=pad_mode,
        )
        grid_shape = next(iter(blocks.values())).shape
        if any(size == 0 for size in grid_shape):
            raise ValueError("Sample has no blocks")

        def serialize_grid(
            loc: tuple[int, ...] = (),
        ) -> NestedList[SerializedSampleBlockType]:
            if len(loc) < len(grid_shape):
                return [serialize_grid((*loc, i)) for i in range(grid_shape[len(loc)])]
            return cls.serialize_sample_block(
                Sample(
                    id=sample.id,
                    stat=sample.stat,
                    members={
                        m: cast(Block, b[loc]).outer_tensor for m, b in blocks.items()
                    },
                )
            )

        return serialize_grid()

    @staticmethod
    @abstractmethod
    def serialize_sample_block(
        sample_block: Sample,
    ) -> SerializedSampleBlockType: ...

    @staticmethod
    @abstractmethod
    def deserialize_sample_block(serialized: SerializedSampleBlockType) -> Sample:
        """Deserialize one block into a sample."""
