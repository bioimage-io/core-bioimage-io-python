from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Generic, Union

from typing_extensions import TypeVar, assert_never

from .axis import PerAxis
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
class Operator(Generic[SampleT], ABC):
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

    def __call__(self, sample: Union[Sample, SampleBlock]) -> None:
        if self.input not in sample.members:
            return

        input_tensor = sample.members[self.input]
        output_tensor = self._apply(input_tensor, sample.stat)

        if self.output in sample.members:
            assert (
                sample.members[self.output].tagged_shape == output_tensor.tagged_shape
            )

        if isinstance(sample, Sample):
            sample.members[self.output] = output_tensor
        elif isinstance(sample, SampleBlock):
            b = sample.blocks[self.input]
            sample.blocks[self.output] = Block(
                sample_shape=self.get_output_shape(sample.shape[self.input]),
                data=output_tensor,
                inner_slice=b.inner_slice,
                halo=b.halo,
                block_index=b.block_index,
                blocks_in_sample=b.blocks_in_sample,
            )
        else:
            assert_never(sample)

    @abstractmethod
    def _apply(self, x: Tensor, stat: Stat) -> Tensor: ...
