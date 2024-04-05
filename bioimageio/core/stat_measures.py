from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple, TypeVar, Union

from .axis import AxisId
from .common import MemberId, PerMember
from .tensor import Tensor

MeasureValue = Union[float, Tensor]


# using Sample Protocol really only to avoid circular imports
class SampleLike(Protocol):
    @property
    def members(self) -> PerMember[Tensor]: ...


@dataclass(frozen=True)
class MeasureBase:
    member_id: MemberId


@dataclass(frozen=True)
class SampleMeasureBase(MeasureBase, ABC):
    @abstractmethod
    def compute(self, sample: SampleLike) -> MeasureValue:
        """compute the measure"""
        ...


@dataclass(frozen=True)
class DatasetMeasureBase(MeasureBase, ABC):
    pass


@dataclass(frozen=True)
class _Mean:
    axes: Optional[Tuple[AxisId, ...]] = None
    """`axes` to reduce"""


@dataclass(frozen=True)
class SampleMean(_Mean, SampleMeasureBase):
    """The mean value of a single tensor"""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.mean(dim=self.axes)

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetMean(_Mean, DatasetMeasureBase):
    """The mean value across multiple samples"""

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") in self.axes


@dataclass(frozen=True)
class _Std:
    axes: Optional[Tuple[AxisId, ...]] = None
    """`axes` to reduce"""


@dataclass(frozen=True)
class SampleStd(_Std, SampleMeasureBase):
    """The standard deviation of a single tensor"""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.std(dim=self.axes)

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetStd(_Std, DatasetMeasureBase):
    """The standard deviation across multiple samples"""

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") in self.axes


@dataclass(frozen=True)
class _Var:
    axes: Optional[Tuple[AxisId, ...]] = None
    """`axes` to reduce"""


@dataclass(frozen=True)
class SampleVar(_Var, SampleMeasureBase):
    """The variance of a single tensor"""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.var(dim=self.axes)

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetVar(_Var, DatasetMeasureBase):
    """The variance across multiple samples"""

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") in self.axes


@dataclass(frozen=True)
class _Percentile:
    q: float
    axes: Optional[Tuple[AxisId, ...]] = None
    """`axes` to reduce"""

    def __post_init__(self):
        assert self.q >= 0.0
        assert self.q <= 1.0


@dataclass(frozen=True)
class SamplePercentile(_Percentile, SampleMeasureBase):
    """The `n`th percentile of a single tensor"""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.quantile(self.q, dim=self.axes)

    def __post_init__(self):
        super().__post_init__()
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetPercentile(_Percentile, DatasetMeasureBase):
    """The `n`th percentile across multiple samples"""

    def __post_init__(self):
        super().__post_init__()
        assert self.axes is None or AxisId("batch") in self.axes


SampleMeasure = Union[SampleMean, SampleStd, SampleVar, SamplePercentile]
DatasetMeasure = Union[DatasetMean, DatasetStd, DatasetVar, DatasetPercentile]
Measure = Union[SampleMeasure, DatasetMeasure]
Stat = Dict[Measure, MeasureValue]

MeanMeasure = Union[SampleMean, DatasetMean]
StdMeasure = Union[SampleStd, DatasetStd]
VarMeasure = Union[SampleVar, DatasetVar]
PercentileMeasure = Union[SamplePercentile, DatasetPercentile]
MeanMeasureT = TypeVar("MeanMeasureT", bound=MeanMeasure)
StdMeasureT = TypeVar("StdMeasureT", bound=StdMeasure)
VarMeasureT = TypeVar("VarMeasureT", bound=VarMeasure)
PercentileMeasureT = TypeVar("PercentileMeasureT", bound=PercentileMeasure)
