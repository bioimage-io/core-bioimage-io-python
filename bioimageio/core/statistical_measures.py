from dataclasses import dataclass
from typing import Optional, Tuple

import xarray as xr


@dataclass(frozen=True)
class Measure:
    def compute(self, tensor: xr.DataArray):
        raise NotImplementedError(self.__class__.__name__)


@dataclass(frozen=True)
class Mean(Measure):
    axes: Optional[Tuple[str]] = None

    def compute(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor.mean(dim=self.axes)


@dataclass(frozen=True)
class Std(Measure):
    axes: Optional[Tuple[str]] = None

    def compute(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor.std(dim=self.axes)


@dataclass(frozen=True)
class Percentile(Measure):
    n: float
    axes: Optional[Tuple[str]] = None

    def __post_init__(self):
        assert self.n >= 0
        assert self.n <= 100

    def compute(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor.quantile(self.n / 100.0, dim=self.axes)
