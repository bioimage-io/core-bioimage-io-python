from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Union

from typing_extensions import assert_never

from .axis import PerAxis
from .block import Block
from .common import MemberId
from .sample import Sample, SampleBlock, SampleBlockWithOrigin
from .stat_measures import (
    Measure,
    Stat,
)
from .tensor import Tensor


@dataclass
class Operator(ABC):
    @abstractmethod
    def __call__(self, sample: Union[Sample, SampleBlockWithOrigin]) -> None: ...

    @property
    @abstractmethod
    def required_measures(self) -> Collection[Measure]: ...


@dataclass
class BlockedOperator(Operator, ABC):
    @abstractmethod
    def __call__(
        self, sample: Union[Sample, SampleBlock, SampleBlockWithOrigin]
    ) -> None: ...

    @property
    @abstractmethod
    def required_measures(self) -> Collection[Measure]: ...


@dataclass
class SimpleOperator(BlockedOperator, ABC):
    input: MemberId
    output: MemberId

    @property
    def required_measures(self) -> Collection[Measure]:
        return set()

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
