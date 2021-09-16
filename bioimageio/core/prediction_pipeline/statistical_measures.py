from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Measure:
    # value: Any  # not specified here to keep clean arg order, e.g. Percentile(50) == Percentile(n=50)
    def __post_init__(self):
        assert hasattr(self, "value")  # monkey patch for abstract dataclass


@dataclass(frozen=True)
class Mean(Measure):
    value: Optional[float] = None


@dataclass(frozen=True)
class Std(Measure):
    value: Optional[float] = None


@dataclass(frozen=True)
class Percentile(Measure):
    n: float
    value: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.n >= 0
        assert self.n <= 100
