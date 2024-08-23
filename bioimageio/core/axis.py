from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, TypeVar, Union

from typing_extensions import assert_never

from bioimageio.spec.model import v0_5


def _get_axis_type(a: Literal["b", "t", "i", "c", "x", "y", "z"]):
    if a == "b":
        return "batch"
    elif a == "t":
        return "time"
    elif a == "i":
        return "index"
    elif a == "c":
        return "channel"
    elif a in ("x", "y", "z"):
        return "space"
    else:
        return "index"  # return most unspecific axis


S = TypeVar("S", bound=str)


AxisId = v0_5.AxisId

T = TypeVar("T")
PerAxis = Mapping[AxisId, T]

BatchSize = int

AxisLetter = Literal["b", "i", "t", "c", "z", "y", "x"]
AxisLike = Union[AxisId, AxisLetter, v0_5.AnyAxis, "Axis"]


@dataclass
class Axis:
    id: AxisId
    type: Literal["batch", "channel", "index", "space", "time"]

    @classmethod
    def create(cls, axis: AxisLike) -> Axis:
        if isinstance(axis, cls):
            return axis
        elif isinstance(axis, Axis):
            return Axis(id=axis.id, type=axis.type)
        elif isinstance(axis, str):
            return Axis(id=AxisId(axis), type=_get_axis_type(axis))
        elif isinstance(axis, v0_5.AxisBase):
            return Axis(id=AxisId(axis.id), type=axis.type)
        else:
            assert_never(axis)


@dataclass
class AxisInfo(Axis):
    maybe_singleton: bool  # TODO: replace 'maybe_singleton' with size min/max for better axis guessing

    @classmethod
    def create(cls, axis: AxisLike, maybe_singleton: Optional[bool] = None) -> AxisInfo:
        if isinstance(axis, AxisInfo):
            return axis

        axis_base = super().create(axis)
        if maybe_singleton is None:
            if isinstance(axis, (Axis, str)):
                maybe_singleton = True
            else:
                if axis.size is None:
                    maybe_singleton = True
                elif isinstance(axis.size, int):
                    maybe_singleton = axis.size == 1
                elif isinstance(axis.size, v0_5.SizeReference):
                    maybe_singleton = (
                        True  # TODO: check if singleton is ok for a `SizeReference`
                    )
                elif isinstance(
                    axis.size, (v0_5.ParameterizedSize, v0_5.DataDependentSize)
                ):
                    try:
                        maybe_size_one = axis.size.validate_size(
                            1
                        )  # TODO: refactor validate_size() to have boolean func here
                    except ValueError:
                        maybe_singleton = False
                    else:
                        maybe_singleton = maybe_size_one == 1
                else:
                    assert_never(axis.size)

        return AxisInfo(
            id=axis_base.id, type=axis_base.type, maybe_singleton=maybe_singleton
        )
