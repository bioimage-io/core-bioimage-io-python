from abc import ABC, abstractmethod
from typing import (
    Generic,
    Iterable,
    Tuple,
    TypeVar,
    Union,
)

from bioimageio.spec.model import v0_5

from ..axis import PerAxis
from ..common import PadMode, PerMember
from ..digest_spec import split_sample_into_blocks_for_model
from ..sample import Sample, SampleBlock

SerializedSampleBlockType = TypeVar("SerializedSampleBlockType")


class SampleSerializer(ABC, Generic[SerializedSampleBlockType]):
    @classmethod
    def serialize_sample(
        cls,
        sample: Sample,
    ) -> Tuple[SerializedSampleBlockType]:
        """Serialize a sample as a single block"""
        return (cls.serialize_sample_block(sample.as_single_block()),)

    @classmethod
    def deserialize_sample(
        cls, serialized: Iterable[SerializedSampleBlockType]
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

        for block in split_sample_into_blocks_for_model(
            sample,
            model=model,
            blocksize_parameter=blocksize_parameter,
            batch_size=batch_size,
        ):
            yield self.serialize_sample_block(block)

    @classmethod
    def serialize_sample_with_fixed_blocking(
        cls,
        sample: Sample,
        *,
        block_shapes: PerMember[PerAxis[int]],
        halo: PerMember[PerAxis[int]],
        pad_mode: Union[PadMode, PerMember[PadMode]] = "symmetric",
    ) -> Iterable[SerializedSampleBlockType]:

        _n_blocks, input_blocks = sample.split_into_blocks(
            block_shapes=block_shapes,
            halo=halo,
            pad_mode=pad_mode,
        )
        for block in input_blocks:
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
