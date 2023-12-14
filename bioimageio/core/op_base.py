from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection

from bioimageio.core.common import Sample
from bioimageio.core.stat_measures import Measure


@dataclass
class Operator(ABC):
    @abstractmethod
    def __call__(self, sample: Sample) -> None:
        ...

    @property
    @abstractmethod
    def required_measures(self) -> Collection[Measure]:
        ...
