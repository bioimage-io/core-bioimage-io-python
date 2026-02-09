from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    Discriminator,
    PlainSerializer,
)
from typing_extensions import Annotated

from .axis import AxisId
from .common import MemberId, PerMember, QuantileMethod
from .tensor import Tensor


def tensor_custom_before_validator(data: Union[Tensor, Mapping[str, Any]]):
    if isinstance(data, Tensor):
        return data

    # custom before validation logic
    return Tensor(np.asarray(data["data"]), dims=data["dims"])


def tensor_custom_serializer(t: Tensor) -> Dict[str, Any]:
    # custome serialization logic
    return {"data": t.data.data.tolist(), "dims": list(map(str, t.dims))}


MeasureValue = Union[
    float,
    Annotated[
        Tensor,
        BeforeValidator(tensor_custom_before_validator),
        PlainSerializer(tensor_custom_serializer),
    ],
]


# using Sample Protocol really only to avoid circular imports
class SampleLike(Protocol):
    @property
    def members(self) -> PerMember[Tensor]: ...


class MeasureBase(BaseModel, frozen=True):
    member_id: MemberId


class SampleMeasureBase(MeasureBase, ABC, frozen=True):
    scope: Literal["sample"] = "sample"

    @abstractmethod
    def compute(self, sample: SampleLike) -> MeasureValue:
        """compute the measure"""
        ...


class DatasetMeasureBase(MeasureBase, ABC, frozen=True):
    scope: Literal["dataset"] = "dataset"


class _Mean(BaseModel, frozen=True):
    name: Literal["mean"] = "mean"
    axes: Optional[Tuple[AxisId, ...]] = None
    """`axes` to reduce"""


class SampleMean(_Mean, SampleMeasureBase, frozen=True):
    """The mean value of a single tensor"""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.mean(dim=self.axes)

    def model_post_init(self, __context: Any):
        assert self.axes is None or AxisId("batch") not in self.axes


class DatasetMean(_Mean, DatasetMeasureBase, frozen=True):
    """The mean value across multiple samples"""

    def model_post_init(self, __context: Any):
        assert self.axes is None or AxisId("batch") in self.axes


class _Std(BaseModel, frozen=True):
    name: Literal["std"] = "std"
    axes: Optional[Tuple[AxisId, ...]] = None
    """`axes` to reduce"""


class SampleStd(_Std, SampleMeasureBase, frozen=True):
    """The standard deviation of a single tensor"""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.std(dim=self.axes)

    def model_post_init(self, __context: Any):
        assert self.axes is None or AxisId("batch") not in self.axes


class DatasetStd(_Std, DatasetMeasureBase, frozen=True):
    """The standard deviation across multiple samples"""

    def model_post_init(self, __context: Any):
        assert self.axes is None or AxisId("batch") in self.axes


class _Var(BaseModel, frozen=True):
    name: Literal["var"] = "var"
    axes: Optional[Tuple[AxisId, ...]] = None
    """`axes` to reduce"""


class SampleVar(_Var, SampleMeasureBase, frozen=True):
    """The variance of a single tensor"""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.var(dim=self.axes)

    def model_post_init(self, __context: Any):
        assert self.axes is None or AxisId("batch") not in self.axes


class DatasetVar(_Var, DatasetMeasureBase, frozen=True):
    """The variance across multiple samples"""

    def model_post_init(self, __context: Any):  # TODO: turn into @model_validator
        assert self.axes is None or AxisId("batch") in self.axes


class _Quantile(BaseModel, frozen=True):
    name: Literal["quantile"] = "quantile"
    q: float
    axes: Optional[Tuple[AxisId, ...]] = None
    """`axes` to reduce"""

    def model_post_init(self, __context: Any):
        assert self.q >= 0.0
        assert self.q <= 1.0


class SampleQuantile(_Quantile, SampleMeasureBase, frozen=True):
    """The `q`th quantile of a single tensor"""

    method: QuantileMethod = "linear"
    """Method to use when the desired quantile lies between two data points.
    See https://numpy.org/devdocs/reference/generated/numpy.quantile.html#numpy-quantile for details."""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.quantile(self.q, dim=self.axes, method=self.method)

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        assert self.axes is None or AxisId("batch") not in self.axes


class DatasetQuantile(_Quantile, DatasetMeasureBase, frozen=True):
    """The `q`th quantile across multiple samples"""

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        assert self.axes is None or AxisId("batch") in self.axes


SampleMeasure = Annotated[
    Union[SampleMean, SampleStd, SampleVar, SampleQuantile], Discriminator("name")
]
DatasetMeasure = Annotated[
    Union[DatasetMean, DatasetStd, DatasetVar, DatasetQuantile], Discriminator("name")
]
Measure = Annotated[Union[SampleMeasure, DatasetMeasure], Discriminator("scope")]
Stat = Dict[Measure, MeasureValue]

MeanMeasure = Union[SampleMean, DatasetMean]
StdMeasure = Union[SampleStd, DatasetStd]
VarMeasure = Union[SampleVar, DatasetVar]
PercentileMeasure = Union[SampleQuantile, DatasetQuantile]
MeanMeasureT = TypeVar("MeanMeasureT", bound=MeanMeasure)
StdMeasureT = TypeVar("StdMeasureT", bound=StdMeasure)
VarMeasureT = TypeVar("VarMeasureT", bound=VarMeasure)
PercentileMeasureT = TypeVar("PercentileMeasureT", bound=PercentileMeasure)
