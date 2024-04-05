from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection

from .sample import SampleBlock
from .stat_measures import Measure


@dataclass
class Operator(ABC):
    @abstractmethod
    def __call__(self, sample_block: SampleBlock) -> None: ...

    @property
    @abstractmethod
    def required_measures(self) -> Collection[Measure]: ...
