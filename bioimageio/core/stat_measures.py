from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, TypeVar, Union

import xarray as xr

from bioimageio.core.common import AxisId, Sample, TensorId

MeasureValue = Union[float, xr.DataArray]


@dataclass(frozen=True)
class MeasureBase:
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
class _Mean:
    axes: Optional[Tuple[AxisId, ...]] = None


@dataclass(frozen=True)
class SampleMean(_Mean, SampleMeasureBase):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample.data[self.tensor_id].mean(dim=self.axes)

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetMean(_Mean, DatasetMeasureBase):
    def __post_init__(self):
        assert self.axes is None or AxisId("batch") in self.axes


@dataclass(frozen=True)
class _Std:
    axes: Optional[Tuple[AxisId, ...]] = None


@dataclass(frozen=True)
class SampleStd(_Std, SampleMeasureBase):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample.data[self.tensor_id].std(dim=self.axes)

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetStd(_Std, DatasetMeasureBase):
    def __post_init__(self):
        assert self.axes is None or AxisId("batch") in self.axes


@dataclass(frozen=True)
class _Var:
    axes: Optional[Tuple[AxisId, ...]] = None


@dataclass(frozen=True)
class SampleVar(_Var, SampleMeasureBase):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample.data[self.tensor_id].var(dim=self.axes)

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetVar(_Var, DatasetMeasureBase):
    def __post_init__(self):
        assert self.axes is None or AxisId("batch") in self.axes


@dataclass(frozen=True)
class _Percentile:
    n: float
    axes: Optional[Tuple[AxisId, ...]] = None

    def __post_init__(self):
        assert self.n >= 0
        assert self.n <= 100


@dataclass(frozen=True)
class SamplePercentile(_Percentile, SampleMeasureBase):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample.data[self.tensor_id].quantile(self.n / 100.0, dim=self.axes)

    def __post_init__(self):
        super().__post_init__()
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetPercentile(_Percentile, DatasetMeasureBase):
    def __post_init__(self):
        super().__post_init__()
        assert self.axes is None or AxisId("batch") in self.axes


SampleMeasure = Union[SampleMean, SampleStd, SampleVar, SamplePercentile]
DatasetMeasure = Union[DatasetMean, DatasetStd, DatasetVar, DatasetPercentile]
Measure = Union[SampleMeasure, DatasetMeasure]

MeanMeasure = Union[SampleMean, DatasetMean]
StdMeasure = Union[SampleStd, DatasetStd]
VarMeasure = Union[SampleVar, DatasetVar]
PercentileMeasure = Union[SamplePercentile, DatasetPercentile]
MeanMeasureT = TypeVar("MeanMeasureT", bound=MeanMeasure)
StdMeasureT = TypeVar("StdMeasureT", bound=StdMeasure)
VarMeasureT = TypeVar("VarMeasureT", bound=VarMeasure)
PercentileMeasureT = TypeVar("PercentileMeasureT", bound=PercentileMeasure)
