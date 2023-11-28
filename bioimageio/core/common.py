from typing import Any, Dict, Generic, List, Literal, NamedTuple, TypeVar, Union

import numpy as np
import xarray as xr
from attr import dataclass
from typing_extensions import Final

from bioimageio.core.stat_measures import MeasureBase
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import TensorId

TensorId = v0_5.TensorId
AxisId = v0_5.AxisId

Sample = Dict[TensorId, xr.DataArray]

ProcessingDescrBase = Union[v0_4.ProcessingDescrBase, v0_5.ProcessingDescrBase]
ProcessingKwargs = Union[v0_4.ProcessingKwargs, v0_5.ProcessingKwargs]

PER_SAMPLE = "per_sample"
PER_DATASET = "per_dataset"
