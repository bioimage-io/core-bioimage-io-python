from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar

from .axis import PerAxis
from .common import MemberId
from .sample import Sample
from .stat_measures import (
    Measure,
    Stat,
)
from .tensor import Tensor

SampleT = TypeVar("SampleT", bound=Sample)


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
class BlockwiseOperator(Operator[Sample]):  # Union[Sample, SampleBlock]
    """Base class for operators that can be applied to whole sample or blockwise."""


@dataclass
class SimpleOperator(BlockwiseOperator):
    """Convenience base class for blockwise operators with a single input and single output."""

    input: MemberId
    output: MemberId

    @abstractmethod
    def get_output_shape(self, input_shape: PerAxis[int]) -> PerAxis[int]: ...

    def __call__(self, sample: Sample) -> None:  # | SampleBlock
        if self.input not in sample.members:
            raise ValueError(
                f"Input {self.input} not found in sample members: {list(sample.members)}"
            )

        input_tensor = sample.members[self.input]
        output_tensor = self._apply(input_tensor, sample.stat)
        if (multi_index := input_tensor.batch_multi_index) is not None:
            output_tensor = output_tensor.assign_batch_multi_index(multi_index)

        assert self.output not in sample.members or (
            sample.members[self.output].tagged_shape == output_tensor.tagged_shape
        ), "overwriting existing output member with different shape"

        sample.members[self.output] = output_tensor

    @abstractmethod
    def _apply(self, x: Tensor, stat: Stat) -> Tensor: ...
