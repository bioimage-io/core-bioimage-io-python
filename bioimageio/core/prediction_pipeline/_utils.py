from __future__ import annotations

import collections.abc
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, NamedTuple, Set, Union

import xarray as xr
from bioimageio.spec.model.v0_5 import TensorId

from bioimageio.core.statistical_measures import Measure, MeasureValue

FixedMode = Literal["fixed"]
SampleMode = Literal["per_sample"]
DatasetMode = Literal["per_dataset"]
Mode = Literal[FixedMode, SampleMode, DatasetMode]

FIXED: FixedMode = "fixed"
PER_SAMPLE: SampleMode = "per_sample"
PER_DATASET: DatasetMode = "per_dataset"
MODES: Set[Mode] = {FIXED, PER_SAMPLE, PER_DATASET}


Sample = Dict[TensorId, xr.DataArray]


class RequiredMeasure(NamedTuple):
    measure: Measure
    tensor_id: TensorId
    mode: Mode

    # def __repr__(self) -> str:
    #     return f"{self.measure} of {self.tensor_id} ({self.mode})"


# RequiredMeasures = List[ReqMeasure]
# @dataclass
# class RequiredMeasures(collections.abc.Iterator[ReqMeasureEntry]):
#     per_sample: Dict[TensorId, Set[Measure]] = field(default_factory=dict)
#     per_dataset: Dict[TensorId, Set[Measure]] = field(default_factory=dict)

#     def update(self, *others: RequiredMeasures):
#         for other in others:
#             for t, ms in other.per_sample.items():
#                 self.per_sample.setdefault(t, set()).update(ms)

#             for t, ms in other.per_dataset.items():
#                 self.per_dataset.setdefault(t, set()).update(ms)

#     def __iter__(self) -> Iterator[ReqMeasureEntry]:
#         for t, ms in self.per_sample.items():
#             for m in ms:
#                 yield ReqMeasureEntry("per_sample", t, m)

#         for t, ms in self.per_dataset.items():
#             for m in ms:
#                 yield ReqMeasureEntry("per_dataset", t, m)


# class ComputedMeasure(NamedTuple):
#     measure: Measure
#     tensor_id: TensorId
#     mode: Mode
#     value: MeasureValue
#     def __repr__(self) -> str:
#         return f"{self.measure} of {self.tensor_id} ({self.mode}) is {self.value}"


# @dataclass
# class ComputedMeasures(collections.abc.Container[CompMeasureEntry]):
#     per_sample: Dict[TensorId, Dict[Measure, MeasureValue]] = field(default_factory=dict)
#     per_dataset: Dict[TensorId, Dict[Measure, MeasureValue]] = field(default_factory=dict)

#     def update(self, other: ComputedMeasures) -> None:
#         for t, ms in other.per_sample.items():
#             self.per_sample.setdefault(t, {}).update(ms)

#         for t, ms in other.per_dataset.items():
#             self.per_dataset.setdefault(t, {}).update(ms)

#     def __iter__(self) -> Iterator[CompMeasureEntry]:
#         for t, ms in self.per_sample.items():
#             for m, v in ms.items():
#                 yield CompMeasureEntry("per_sample", t, m, v)

#         for t, ms in self.per_dataset.items():
#             for m, v in ms.items():
#                 yield CompMeasureEntry("per_dataset", t, m, v)

#     def __contains__(self, __x: Any) -> bool:
#         if isinstance(__x, CompMeasureEntry):

#         elif isinstance(__x, ReqMeasureEntry):

#         else:
#             return super().__contains__(__x)