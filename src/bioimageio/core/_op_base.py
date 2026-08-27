from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from typing import Generic, Union

from typing_extensions import TypeVar, assert_never

from ._restore_batch_multi_index import restore_batch_multi_index
from .axis import BATCH_AXIS_ID, PerAxis
from .block import Block
from .common import MemberId
from .sample import Sample, SampleBlock, SampleBlockWithOrigin
from .stat_measures import (
    Measure,
    Stat,
)
from .tensor import Tensor

SampleT = TypeVar("SampleT", bound=Union[Sample, SampleBlock, SampleBlockWithOrigin])


@dataclass
class Operator(ABC, Generic[SampleT]):
    """Base class for all operators."""

    @abstractmethod
    def __call__(self, sample: SampleT) -> None: ...

    @property
    @abstractmethod
    def required_measures(self) -> Collection[Measure]: ...


@dataclass
class SamplewiseOperator(Operator[Sample]):
    """Base class for operators that can only be applied to whole samples."""


@dataclass
class BlockwiseOperator(Operator[Union[Sample, SampleBlock]]):
    """Base class for operators that can be applied to whole sample or blockwise."""


@dataclass
class SimpleOperator(BlockwiseOperator):
    """Convenience base class for blockwise operators with a single input and single output."""

    input: MemberId
    output: MemberId

    @abstractmethod
    def get_output_shape(self, input_shape: PerAxis[int]) -> PerAxis[int]: ...

    def __call__(self, sample: Sample | SampleBlock) -> None:
        if self.input not in sample.members:
            return  # TODO: raise?

        input_tensor = sample.members[self.input]
        output_tensor = self._apply(input_tensor, sample.stat)
        output_tensor = restore_batch_multi_index(
            {self.input: input_tensor}, {self.output: output_tensor}
        )[self.output]
        assert output_tensor is not None
        if self.output in sample.members:
            assert (
                sample.members[self.output].tagged_shape == output_tensor.tagged_shape
            )

        if isinstance(sample, Sample):
            sample.members[self.output] = output_tensor
        elif isinstance(sample, SampleBlock):
            b = sample.blocks[self.input]
            sample_shape_batch_unstacked = {
                a: s
                for a, s in sample.sample_shape_batch_unstacked[self.input].items()
                if a not in sample.sample_shape[self.input]
            }
            sample_shape = self.get_output_shape(sample.sample_shape[self.input])
            for a, s in sample_shape.items():
                if a == BATCH_AXIS_ID:
                    continue
                elif a in sample_shape_batch_unstacked:
                    raise ValueError(
                        f"Output axis {a} already exists in sample_shape_batch_unstacked"
                    )

                sample_shape_batch_unstacked[a] = s

            sample.blocks[self.output] = Block(
                sample_shape=sample_shape,
                sample_shape_batch_unstacked=sample_shape_batch_unstacked,
                data=output_tensor,
                inner_slice=b.inner_slice,
                inner_slice_batch_unstacked=b.inner_slice_batch_unstacked,
                halo=b.halo,
                block_index=b.block_index,
                blocks_in_sample=b.blocks_in_sample,
            )
        else:
            assert_never(sample)

    @abstractmethod
    def _apply(self, x: Tensor, stat: Stat) -> Tensor: ...
