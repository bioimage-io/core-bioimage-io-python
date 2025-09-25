from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, TypeVar, Union

from bioimageio.spec.model import v0_5
from typing_extensions import Protocol, assert_never, runtime_checkable


def _guess_axis_type(a: str):
    if a in ("b", "batch"):
        return "batch"
    elif a in ("t", "time"):
        return "time"
    elif a in ("i", "index"):
        return "index"
    elif a in ("c", "channel"):
        return "channel"
    elif a in ("x", "y", "z"):
        return "space"
    else:
        raise ValueError(
            f"Failed to infer axis type for axis id '{a}'."
            + " Consider using one of: '"
            + "', '".join(
                ["b", "batch", "t", "time", "i", "index", "c", "channel", "x", "y", "z"]
            )
            + "'. Or creating an `Axis` object instead."
        )


S = TypeVar("S", bound=str)


AxisId = v0_5.AxisId
"""An axis identifier, e.g. 'batch', 'channel', 'z', 'y', 'x'"""

T = TypeVar("T")
PerAxis = Mapping[AxisId, T]

BatchSize = int

AxisLetter = Literal["b", "i", "t", "c", "z", "y", "x"]
_AxisLikePlain = Union[str, AxisId, AxisLetter]


@runtime_checkable
class AxisDescrLike(Protocol):
    id: _AxisLikePlain
    type: Literal["batch", "channel", "index", "space", "time"]


AxisLike = Union[_AxisLikePlain, "Axis", v0_5.AnyAxis, AxisDescrLike]


@dataclass
class Axis:
    id: AxisId
    type: Literal["batch", "channel", "index", "space", "time"]

    def __post_init__(self):
        if self.type == "batch":
            self.id = AxisId("batch")
        elif self.type == "channel":
            self.id = AxisId("channel")

    @classmethod
    def create(cls, axis: AxisLike) -> Axis:
        if isinstance(axis, cls):
            return axis

        if isinstance(axis, (AxisId, str)):
            axis_id = axis
            axis_type = _guess_axis_type(str(axis))
        else:
            if hasattr(axis, "type"):
                axis_type = axis.type
            else:
                axis_type = _guess_axis_type(str(axis))

            if hasattr(axis, "id"):
                axis_id = axis.id
            else:
                axis_id = axis

        return Axis(id=AxisId(axis_id), type=axis_type)


@dataclass
class AxisInfo(Axis):
    maybe_singleton: bool  # TODO: replace 'maybe_singleton' with size min/max for better axis guessing

    @classmethod
    def create(cls, axis: AxisLike, maybe_singleton: Optional[bool] = None) -> AxisInfo:
        if isinstance(axis, AxisInfo):
            return axis

        axis_base = super().create(axis)
        if maybe_singleton is None:
            if not isinstance(axis, v0_5.AxisBase):
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
