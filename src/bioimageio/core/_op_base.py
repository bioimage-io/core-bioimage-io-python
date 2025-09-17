from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Union

from .sample import Sample, SampleBlock, SampleBlockWithOrigin
from .stat_measures import Measure


@dataclass
class Operator(ABC):
    @abstractmethod
    def __call__(self, sample: Union[Sample, SampleBlockWithOrigin]) -> None: ...

    @property
    @abstractmethod
    def required_measures(self) -> Collection[Measure]: ...


@dataclass
class BlockedOperator(Operator, ABC):
    @abstractmethod
    def __call__(
        self, sample: Union[Sample, SampleBlock, SampleBlockWithOrigin]
    ) -> None: ...

    @property
    @abstractmethod
    def required_measures(self) -> Collection[Measure]: ...
