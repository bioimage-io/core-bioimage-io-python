from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    cast,
)

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    PlainSerializer,
)

from ._op_base import Operator
from .axis import AxisId
from .common import MemberId, PerMember, QuantileMethod
from .tensor import Tensor


def tensor_custom_before_validator(data: Tensor | Mapping[str, Any]):
    if isinstance(data, Tensor):
        return data

    # custom before validation logic
    return Tensor(np.asarray(data["data"]), dims=data["dims"])


def tensor_custom_serializer(t: Tensor) -> dict[str, Any]:
    # Serialization explicitly evaluates lazy statistics.
    return {"data": t.data.to_numpy().tolist(), "dims": list(map(str, t.dims))}


MeasureValue: TypeAlias = (
    float
    | Annotated[
        Tensor,
        BeforeValidator(tensor_custom_before_validator),
        PlainSerializer(tensor_custom_serializer),
    ]
)


def _validate_applied_ops(value: Any) -> tuple[Operator, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("applied_ops must be a sequence")

    values = cast(list[Any] | tuple[Any, ...], value)
    if all(isinstance(op, Operator) for op in values):
        return cast(tuple[Operator, ...], tuple(values))

    from .proc_ops import deserialize_processing

    return tuple(deserialize_processing(cast(Mapping[str, Any], op)) for op in values)


def _serialize_applied_ops(ops: tuple[Operator, ...]) -> list[dict[str, Any]]:
    from .proc_ops import serialize_processing

    return [serialize_processing(op) for op in ops]


AppliedOps: TypeAlias = Annotated[
    tuple[Operator, ...],
    BeforeValidator(_validate_applied_ops),
    PlainSerializer(_serialize_applied_ops),
]


# using Sample Protocol really only to avoid circular imports
class SampleLike(Protocol):
    @property
    def members(self) -> PerMember[Tensor]: ...


class MeasureBase(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    member_id: MemberId
    preceding_ops: AppliedOps = ()


class SampleMeasureBase(MeasureBase, ABC, frozen=True):
    scope: Literal["sample"] = "sample"

    @abstractmethod
    def compute(self, sample: SampleLike) -> MeasureValue:
        """Compute eagerly for NumPy inputs and lazily for Dask inputs."""
        ...


class DatasetMeasureBase(MeasureBase, ABC, frozen=True):
    scope: Literal["dataset"] = "dataset"


class _Mean(BaseModel, frozen=True):
    name: Literal["mean"] = "mean"
    axes: tuple[AxisId, ...] | None = None
    """`axes` to reduce"""


class SampleMean(_Mean, SampleMeasureBase, frozen=True):
    """The mean value of a single tensor"""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.mean(dim=self.axes)

    def model_post_init(self, __context: Any, /):
        assert self.axes is None or AxisId("batch") not in self.axes


class DatasetMean(_Mean, DatasetMeasureBase, frozen=True):
    """The mean value across multiple samples"""

    def model_post_init(self, __context: Any, /):
        assert self.axes is None or AxisId("batch") in self.axes


class _Std(BaseModel, frozen=True):
    name: Literal["std"] = "std"
    axes: tuple[AxisId, ...] | None = None
    """`axes` to reduce"""


class SampleStd(_Std, SampleMeasureBase, frozen=True):
    """The standard deviation of a single tensor"""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.std(dim=self.axes)

    def model_post_init(self, __context: Any, /):
        assert self.axes is None or AxisId("batch") not in self.axes


class DatasetStd(_Std, DatasetMeasureBase, frozen=True):
    """The standard deviation across multiple samples"""

    def model_post_init(self, __context: Any, /):
        assert self.axes is None or AxisId("batch") in self.axes


class _Var(BaseModel, frozen=True):
    name: Literal["var"] = "var"
    axes: tuple[AxisId, ...] | None = None
    """`axes` to reduce"""


class SampleVar(_Var, SampleMeasureBase, frozen=True):
    """The variance of a single tensor"""

    def compute(self, sample: SampleLike) -> MeasureValue:
        tensor = sample.members[self.member_id]
        return tensor.var(dim=self.axes)

    def model_post_init(self, __context: Any, /):
        assert self.axes is None or AxisId("batch") not in self.axes


class DatasetVar(_Var, DatasetMeasureBase, frozen=True):
    """The variance across multiple samples"""

    def model_post_init(self, __context: Any, /):  # TODO: turn into @model_validator
        assert self.axes is None or AxisId("batch") in self.axes


class _Quantile(BaseModel, frozen=True):
    name: Literal["quantile"] = "quantile"
    q: float
    axes: tuple[AxisId, ...] | None = None
    """`axes` to reduce"""

    def model_post_init(self, __context: Any, /):
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

    def model_post_init(self, __context: Any, /):
        super().model_post_init(__context)
        assert self.axes is None or AxisId("batch") not in self.axes


class DatasetQuantile(_Quantile, DatasetMeasureBase, frozen=True):
    """The `q`th quantile across multiple samples"""

    def model_post_init(self, __context: Any, /):
        super().model_post_init(__context)
        assert self.axes is None or AxisId("batch") in self.axes


SampleMeasure: TypeAlias = Annotated[
    SampleMean | SampleStd | SampleVar | SampleQuantile, Discriminator("name")
]
DatasetMeasure: TypeAlias = Annotated[
    DatasetMean | DatasetStd | DatasetVar | DatasetQuantile, Discriminator("name")
]
Measure: TypeAlias = Annotated[SampleMeasure | DatasetMeasure, Discriminator("scope")]
Stat: TypeAlias = dict[Measure, MeasureValue]

MeanMeasure: TypeAlias = SampleMean | DatasetMean
StdMeasure: TypeAlias = SampleStd | DatasetStd
VarMeasure: TypeAlias = SampleVar | DatasetVar
PercentileMeasure: TypeAlias = SampleQuantile | DatasetQuantile
