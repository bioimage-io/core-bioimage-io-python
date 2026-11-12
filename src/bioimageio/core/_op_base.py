from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from torch import TYPE_CHECKING

from .axis import PerAxis
from .common import MemberId
from .tensor import Tensor

if TYPE_CHECKING:
    from .sample import Sample
    from .stat_measures import Measure, Stat


@dataclass(frozen=True, kw_only=True)
class Operator(ABC):
    """Base class for all operators.

    Implementations must record themselves in ``sample.applied_ops`` after success.
    """

    @abstractmethod
    def __call__(self, sample: Sample) -> None: ...

    @property
    def output_member_ids(self) -> Collection[MemberId]:
        return ()

    @property
    def affects_statistics(self) -> bool:
        """Whether this operator may change statistics of its output tensors."""
        return True

    def get_required_measures(self) -> Collection[Measure]:
        return ()

    def get_descr(self) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} has no processing description"
        )


def get_relevant_preceding_ops(
    member_id: MemberId, applied_ops: Sequence[Operator]
) -> tuple[Operator, ...]:
    return tuple(
        op
        for op in applied_ops
        if op.affects_statistics and member_id in op.output_member_ids
    )


@dataclass(frozen=True, kw_only=True)
class SamplewiseOperator(Operator):
    """Base class for operators that can only be applied to whole samples."""


@dataclass(frozen=True, kw_only=True)
class BlockwiseOperator(Operator):
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
        sample.applied_ops += (self,)

    @abstractmethod
    def _apply(self, x: Tensor, stat: Stat) -> Tensor: ...
