from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, TypeVar, Union

import xarray as xr

from bioimageio.core.common import Sample
from bioimageio.spec.model.v0_5 import AxisId, TensorId

MeasureValue = Union[float, xr.DataArray]


@dataclass(frozen=True)
class MeasureBase(ABC):
    tensor_id: TensorId


@dataclass(frozen=True)
class SampleMeasureBase(MeasureBase, ABC):
    @abstractmethod
    def compute(self, sample: Sample) -> MeasureValue:
        """compute the measure"""
        ...


@dataclass(frozen=True)
class DatasetMeasureBase(MeasureBase, ABC):
    pass


@dataclass(frozen=True)
class _Mean(MeasureBase):
    axes: Optional[Tuple[AxisId, ...]] = None


@dataclass(frozen=True)
class SampleMean(_Mean, SampleMeasureBase):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample[self.tensor_id].mean(dim=self.axes)


@dataclass(frozen=True)
class DatasetMean(_Mean, DatasetMeasureBase):
    pass


@dataclass(frozen=True)
class _Std(MeasureBase):
    axes: Optional[Tuple[AxisId, ...]] = None


@dataclass(frozen=True)
class SampleStd(_Std, SampleMeasureBase):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample[self.tensor_id].std(dim=self.axes)


@dataclass(frozen=True)
class DatasetStd(_Std, DatasetMeasureBase):
    pass


@dataclass(frozen=True)
class _Var(MeasureBase):
    axes: Optional[Tuple[AxisId, ...]] = None


@dataclass(frozen=True)
class SampleVar(_Var, SampleMeasureBase):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample[self.tensor_id].var(dim=self.axes)


@dataclass(frozen=True)
class DatasetVar(_Var, DatasetMeasureBase):
    pass


@dataclass(frozen=True)
class _Percentile(MeasureBase):
    n: float
    axes: Optional[Tuple[AxisId, ...]] = None

    def __post_init__(self):
        assert self.n >= 0
        assert self.n <= 100


@dataclass(frozen=True)
class SamplePercentile(_Percentile, SampleMeasureBase):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample[self.tensor_id].tensor.quantile(self.n / 100.0, dim=self.axes)


@dataclass(frozen=True)
class DatasetPercentile(_Percentile, DatasetMeasureBase):
    pass


SampleMeasure = Union[SampleMean, SampleStd, SampleVar, SamplePercentile]
DatasetMeasure = Union[DatasetMean, DatasetStd, DatasetVar, DatasetPercentile]
Measure = Union[SampleMeasure, DatasetMeasure]

# MeasureVar = TypeVar("MeasureVar", bound=MeasureBase)
# SampleMeasureVar = TypeVar("SampleMeasureVar", bound=SampleMeasureBase)
# DatasetMeasureVar = TypeVar("DatasetMeasureVar", bound=DatasetMeasureBase)
# ModeVar = TypeVar("ModeVar", bound=Literal["per_sample", "per_dataset"])
