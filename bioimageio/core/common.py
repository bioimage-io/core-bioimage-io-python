from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Literal, Protocol

import xarray as xr

from bioimageio.spec.model import v0_5

if TYPE_CHECKING:
    from bioimageio.core.stat_measures import Measure, MeasureValue

TensorId = v0_5.TensorId
AxisId = v0_5.AxisId


@dataclass
class Axis:
    id: AxisId
    type: Literal["batch", "channel", "index", "space", "time"]


class AxisLike(Protocol):
    id: str
    type: Literal["batch", "channel", "index", "space", "time"]


BatchSize = int
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


Dataset = Iterable[Sample]
