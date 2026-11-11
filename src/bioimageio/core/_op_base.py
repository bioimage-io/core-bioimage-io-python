from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import Self, TypeVar

from .axis import PerAxis
from .common import MemberId

if TYPE_CHECKING:
    from .sample import Sample
    from .stat_measures import Measure, Stat
    from .tensor import Tensor

SampleT = TypeVar("SampleT", bound="Sample")


@dataclass(frozen=True, kw_only=True)
class Operator(ABC, Generic[SampleT]):
    """Base class for all operators."""

    preceding_ops: tuple[Operator[Any], ...] = ()

    @abstractmethod
    def __call__(self, sample: SampleT) -> None: ...

    @property
    def output_member_ids(self) -> Collection[MemberId]:
        return ()

    def get_required_measures(
        self, preceding_ops: Sequence[Operator[Any]] = ()
    ) -> Collection[Measure]:
        return ()

    def bind(self, preceding_ops: Sequence[Operator[Any]]) -> Self:
        bound = copy.copy(self)
        object.__setattr__(bound, "preceding_ops", tuple(preceding_ops))
        return bound

    def get_descr(self) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} has no processing description"
        )


@dataclass(frozen=True, kw_only=True)
class SamplewiseOperator(Operator["Sample"]):
    """Base class for operators that can only be applied to whole samples."""


@dataclass(frozen=True, kw_only=True)
class BlockwiseOperator(Operator["Sample"]):  # Union[Sample, SampleBlock]
    """Base class for operators that can be applied to whole sample or blockwise."""


@dataclass(frozen=True)
class SimpleOperator(BlockwiseOperator):
    """Convenience base class for blockwise operators with a single input and single output."""

    input: MemberId
    output: MemberId

    @property
    def output_member_ids(self) -> Collection[MemberId]:
        return (self.output,)

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
