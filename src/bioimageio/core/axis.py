from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Literal,
    TypeVar,
    Union,
)

from loguru import logger
from typing_extensions import Protocol, TypeAlias, assert_never, runtime_checkable

from bioimageio.spec.model import v0_5


def _guess_axis_type(a: str):
    a = a.lower()
    if a in ("b", "batch"):
        return "batch"
    elif a in ("t", "time"):
        return "time"
    elif a in ("s", "sample", "i", "index"):
        return "index"
    elif a in ("c", "channel"):
        return "channel"
    elif a in ("x", "y", "z") or a.startswith("space"):
        return "space"
    else:
        logger.warning(
            "Failed to infer axis type for axis id '{}'."
            + " Consider using one of:"
            + " 'b', 'batch', 't', 'time', 's', 'sample', 'i',"
            + " 'index', 'c', 'channel', 'x', 'y', 'z',"
            + " 'space*'. Or create an `Axis` object with an explicity type instead.",
            a,
        )
        return "unknown"


S = TypeVar("S", bound=str)


AxisId: TypeAlias = v0_5.AxisId
"""An axis identifier, e.g. 'batch', 'channel', 'z', 'y', 'x'"""

BATCH_AXIS_ID = v0_5.BATCH_AXIS_ID
CHANNEL_AXIS_ID = v0_5.CHANNEL_AXIS_ID
TIME_AXIS_ID = v0_5.DEFAULT_TIME_AXIS_ID

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


def axis_id_from_axis_like(a: AxisLike) -> AxisId:
    if isinstance(
        a,
        (Axis, v0_5.AxisBase),
    ):
        return a.id
    elif isinstance(a, AxisDescrLike):
        a = a.id

    if isinstance(a, AxisId):
        return a
    elif isinstance(a, str):
        return AxisId(a)
    else:
        assert_never(a)


@dataclass
class Axis:
    id: AxisId
    type: Literal["batch", "channel", "index", "space", "time", "unknown"]

    def __post_init__(self):
        if self.type == "batch" and self.id != BATCH_AXIS_ID:
            raise ValueError(
                f"Axis with type 'batch' must have id '{BATCH_AXIS_ID}', but got '{self.id}'"
            )
        if self.type == "channel" and self.id != CHANNEL_AXIS_ID:
            raise ValueError(
                f"Axis with type 'channel' must have id '{CHANNEL_AXIS_ID}', but got '{self.id}'"
            )

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
    def create(cls, axis: AxisLike, size: int | AxisSize | None = None) -> AxisInfo:
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


def single_letter_dims_if_possible(
    dims: Sequence[AxisId],
) -> tuple[str, ...]:
    """Return a tuple of single-letter dimension names if possible, otherwise return the original dimension names."""
    single_letter_dims: list[str] = []

    for d in dims:
        d = str(d).lower()
        if d in ("batch", "b"):
            single_letter_dims.append("b")
        elif d in ("time", "t"):
            single_letter_dims.append("t")
        elif d in ("index", "i"):
            single_letter_dims.append("i")
        elif d in ("channel", "c"):
            single_letter_dims.append("c")
        elif d in "zyx":
            single_letter_dims.append(d)
        else:
            # Return original dims if any dim cannot be converted to a single letter
            return tuple(dims)

    if len(set(single_letter_dims)) != len(single_letter_dims):
        # Return original dims if any single letter dim is not unique
        return tuple(dims)

    return tuple(single_letter_dims)
