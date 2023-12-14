from dataclasses import field
from typing import Dict, Union

import xarray as xr
from attr import dataclass

from bioimageio.core.stat_measures import Measure, MeasureValue
from bioimageio.spec.model import v0_4, v0_5

TensorId = v0_5.TensorId
AxisId = v0_5.AxisId

Tensor = xr.DataArray

Data = Dict[TensorId, Tensor]
Stat = Dict[Measure, MeasureValue]


@dataclass
class Sample:
    data: Data = field(default_factory=dict)
    stat: Stat = field(default_factory=dict)


ProcessingDescrBase = Union[v0_4.ProcessingDescrBase, v0_5.ProcessingDescrBase]
ProcessingKwargs = Union[v0_4.ProcessingKwargs, v0_5.ProcessingKwargs]
