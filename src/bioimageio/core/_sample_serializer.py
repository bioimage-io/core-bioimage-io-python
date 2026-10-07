from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import (
    Generic,
    TypeVar,
)

from bioimageio.spec.model import v0_5

from ._utils.type_hints import NestedList
from .axis import PerAxis
from .common import HaloLike, PadMode, PerMember
from .sample import Sample

SerializedSampleBlockType = TypeVar("SerializedSampleBlockType")


class SampleSerializer(ABC, Generic[SerializedSampleBlockType]):
    @classmethod
    def serialize_sample(
        cls,
        sample: Sample,
    ) -> NestedList[SerializedSampleBlockType]:
        """Serialize a sample (chunkwise for chunked samples)"""

        # for block in sample.blockwise
        return cls.serialize_sample_block(block)

    @classmethod
    def deserialize_sample(
        cls,
        serialized: Iterable[SerializedSampleBlockType],
    ) -> Sample:
        return Sample.from_blocks((cls.deserialize_sample_block(s) for s in serialized))

    def serialize_sample_blockwise(
        self,
        sample: Sample,
        *,
        model: v0_5.ModelDescr,
        blocksize_parameter: int,
        batch_size: int = 1,
    ) -> Iterable[SerializedSampleBlockType]:
        """Split a sample into blocks according to the model's input specifications and `blocksize_parameter` and serialize each block."""

        block_shape = model.get_tensor_sizes(
            blocksize_parameter, batch_size, sample.shape
        ).inputs
        sample = sample.chunk(block_shape)

    @classmethod
    def serialize_sample_with_fixed_blocking(
        cls,
        sample: Sample,
        *,
        block_shapes: PerMember[PerAxis[int]],
        halo: PerMember[PerAxis[HaloLike]],
        pad_mode: PadMode | PerMember[PadMode] = "symmetric",
    ) -> Iterable[SerializedSampleBlockType]:

        _n_blocks, blocks = sample.split_into_blocks(
            block_shapes=block_shapes,
            halo=halo,
            pad_mode=pad_mode,
        )
        assert _n_blocks > 0, "sample has no blocks"
        for block in blocks:
            yield cls.serialize_sample_block(block)

    @staticmethod
    @abstractmethod
    def serialize_sample_block(
        sample_block: SampleBlock,
    ) -> SerializedSampleBlockType: ...

    @staticmethod
    @abstractmethod
    def deserialize_sample_block(serialized: SerializedSampleBlockType) -> SampleBlock:
        """Deserialize a sample block into a new sample or merge it into `output_sample` if provided."""
