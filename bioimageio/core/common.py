from typing import Any, Dict, Generic, List, Literal, NamedTuple, TypeVar, Union

import numpy as np
import xarray as xr
from attr import dataclass
from typing_extensions import Final

from bioimageio.core.stat_measures import Measure
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import TensorId

TensorId = v0_5.TensorId
AxisId = v0_5.AxisId

Sample = Dict[TensorId, xr.DataArray]

ProcessingDescrBase = Union[v0_4.ProcessingDescrBase, v0_5.ProcessingDescrBase]
ProcessingKwargs = Union[v0_4.ProcessingKwargs, v0_5.ProcessingKwargs]

PER_SAMPLE = "per_sample"
PER_DATASET = "per_dataset"


MeasureVar = TypeVar("MeasureVar", bound=Measure)
ModeVar = TypeVar("ModeVar", Literal["per_sample"], Literal["per_dataset"])


@dataclass(frozen=True)
class RequiredMeasure(Generic[MeasureVar, ModeVar]):
    measure: MeasureVar
    tensor_id: TensorId
    mode: ModeVar


@dataclass(frozen=True)
class SampleMeasure(RequiredMeasure[MeasureVar, Literal["per_sample"]]):
    pass


@dataclass(frozen=True)
class DatasetMeasure(RequiredMeasure[MeasureVar, Literal["per_dataset"]]):
    pass


MeasureValue = xr.DataArray
