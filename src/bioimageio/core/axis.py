from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import (
    Literal,
    TypeVar,
    Union,
)

from typing_extensions import Protocol, TypeAlias, assert_never, runtime_checkable

from bioimageio.spec.model import v0_5


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
            + "b', 'batch', 't', 'time', 'i', 'index', 'c', 'channel', 'x', 'y', 'z"
            + "'. Or creating an `Axis` object instead."
        )


S = TypeVar("S", bound=str)


AxisId: TypeAlias = v0_5.AxisId
"""An axis identifier, e.g. 'batch', 'channel', 'z', 'y', 'x'"""

_T = TypeVar("_T")
PerAxis = Mapping[AxisId, _T]


BatchSize = int

AxisLetter = Literal["b", "i", "t", "c", "z", "y", "x"]
_AxisLikePlain = Union[str, AxisId, AxisLetter]


@runtime_checkable
class AxisDescrLike(Protocol):
    id: _AxisLikePlain
    type: Literal["batch", "channel", "index", "space", "time"]


AxisLike = Union[_AxisLikePlain, AxisDescrLike, v0_5.AnyAxis, "Axis", "AxisInfo"]


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
class AxisSize:
    min: int
    max: int | None = None
    step: int | None = None


@dataclass
class AxisInfo(Axis):
    size: AxisSize

    @classmethod
    def create(
        cls, axis: AxisLike, size: int | AxisSize | None = None
    ) -> AxisInfo:
        if isinstance(axis, AxisInfo):
            return axis

        axis_base = super().create(axis)
        if size is None:
            if not isinstance(axis, v0_5.AxisBase):
                size = AxisSize(min=1)
            else:
                if axis.size is None:
                    size = AxisSize(min=1)
                elif isinstance(axis.size, int):
                    size = AxisSize(
                        min=axis.size,
                        max=None
                        if isinstance(axis, (v0_5.TimeAxisBase, v0_5.SpaceAxisBase))
                        or (
                            not isinstance(axis, v0_5.IndexOutputAxis)
                            and axis.concatenable
                        )
                        else axis.size,
                    )
                elif isinstance(axis.size, v0_5.SizeReference):
                    size = AxisSize(min=axis.size.offset + 1)
                elif isinstance(axis.size, v0_5.ParameterizedSize):
                    size = AxisSize(min=axis.size.min, step=axis.size.step)
                elif isinstance(axis.size, v0_5.DataDependentSize):
                    size = AxisSize(min=axis.size.min, max=axis.size.max)
                else:
                    assert_never(axis.size)
        elif isinstance(size, int):
            size = AxisSize(min=size, max=size)

        return AxisInfo(id=axis_base.id, type=axis_base.type, size=size)
