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
    """`axes` to reduce"""


@dataclass(frozen=True)
class SampleMean(_Mean, SampleMeasureBase):
    """The mean value of a single tensor"""

    def compute(self, sample: Sample) -> MeasureValue:
        tensor = sample.data[self.tensor_id]
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

    def compute(self, sample: Sample) -> MeasureValue:
        tensor = sample.data[self.tensor_id]
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

    def compute(self, sample: Sample) -> MeasureValue:
        tensor = sample.data[self.tensor_id]
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
    n: float
    axes: Optional[Tuple[AxisId, ...]] = None
    """`axes` to reduce"""

    def __post_init__(self):
        assert self.n >= 0
        assert self.n <= 100


@dataclass(frozen=True)
class SamplePercentile(_Percentile, SampleMeasureBase):
    """The `n`th percentile of a single tensor"""

    def compute(self, sample: Sample) -> MeasureValue:
        tensor = sample.data[self.tensor_id]
        return tensor.quantile(self.n / 100.0, dim=self.axes)

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

MeanMeasure = Union[SampleMean, DatasetMean]
StdMeasure = Union[SampleStd, DatasetStd]
VarMeasure = Union[SampleVar, DatasetVar]
PercentileMeasure = Union[SamplePercentile, DatasetPercentile]
MeanMeasureT = TypeVar("MeanMeasureT", bound=MeanMeasure)
StdMeasureT = TypeVar("StdMeasureT", bound=StdMeasure)
VarMeasureT = TypeVar("VarMeasureT", bound=VarMeasure)
PercentileMeasureT = TypeVar("PercentileMeasureT", bound=PercentileMeasure)
