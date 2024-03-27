from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Tuple,
    Union,
)

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


class LeftRight(NamedTuple):
    left: int
    right: int


class SliceInfo(NamedTuple):
    start: int
    stop: int
