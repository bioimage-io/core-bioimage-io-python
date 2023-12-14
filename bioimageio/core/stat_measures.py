from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

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
class SampleMean(SampleMeasureBase, _Mean):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample.data[self.tensor_id].mean(dim=self.axes)

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetMean(DatasetMeasureBase, _Mean):
    def __post_init__(self):
        assert self.axes is None or AxisId("batch") in self.axes


@dataclass(frozen=True)
class _Std:
    axes: Optional[Tuple[AxisId, ...]] = None


@dataclass(frozen=True)
class SampleStd(SampleMeasureBase, _Std):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample.data[self.tensor_id].std(dim=self.axes)

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetStd(DatasetMeasureBase, _Std):
    def __post_init__(self):
        assert self.axes is None or AxisId("batch") in self.axes


@dataclass(frozen=True)
class _Var:
    axes: Optional[Tuple[AxisId, ...]] = None


@dataclass(frozen=True)
class SampleVar(SampleMeasureBase, _Var):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample.data[self.tensor_id].var(dim=self.axes)

    def __post_init__(self):
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetVar(DatasetMeasureBase, _Var):
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
class SamplePercentile(SampleMeasureBase, _Percentile):
    def compute(self, sample: Sample) -> MeasureValue:
        return sample.data[self.tensor_id].quantile(self.n / 100.0, dim=self.axes)

    def __post_init__(self):
        super().__post_init__()
        assert self.axes is None or AxisId("batch") not in self.axes


@dataclass(frozen=True)
class DatasetPercentile(DatasetMeasureBase, _Percentile):
    def __post_init__(self):
        super().__post_init__()
        assert self.axes is None or AxisId("batch") in self.axes


SampleMeasure = Union[SampleMean, SampleStd, SampleVar, SamplePercentile]
DatasetMeasure = Union[DatasetMean, DatasetStd, DatasetVar, DatasetPercentile]
Measure = Union[SampleMeasure, DatasetMeasure]
