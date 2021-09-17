from dataclasses import dataclass


@dataclass(frozen=True)
class Measure:
    pass


@dataclass(frozen=True)
class Mean(Measure):
    pass


@dataclass(frozen=True)
class Std(Measure):
    pass


@dataclass(frozen=True)
class Percentile(Measure):
    n: float

    def __post_init__(self):
        assert self.n >= 0
        assert self.n <= 100
