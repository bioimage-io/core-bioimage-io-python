from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Measure:
    pass


@dataclass(frozen=True)
class Mean(Measure):
    axes: Optional[Tuple[str]] = None


@dataclass(frozen=True)
class Std(Measure):
    axes: Optional[Tuple[str]] = None


@dataclass(frozen=True)
class Percentile(Measure):
    n: float
    axes: Optional[Tuple[str]] = None

    def __post_init__(self):
        assert self.n >= 0
        assert self.n <= 100
