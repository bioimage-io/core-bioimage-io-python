from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict

import xarray as xr

from bioimageio.spec.model import v0_5

if TYPE_CHECKING:
    from bioimageio.core.stat_measures import Measure, MeasureValue

TensorId = v0_5.TensorId
AxisId = v0_5.AxisId

Tensor = xr.DataArray

Data = Dict[TensorId, Tensor]
Stat = Dict["Measure", "MeasureValue"]


@dataclass
class Sample:
    """A (dataset) sample"""

    data: Data = field(default_factory=dict)
    """the samples tensors"""

    stat: Stat = field(default_factory=dict)
    """sample and dataset statistics"""
