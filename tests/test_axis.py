from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from bioimageio.core.axis import (
    Axis,
    AxisId,
    AxisInfo,
    AxisLike,
    AxisSize,
    axis_id_from_axis_like,
    single_letter_dims_if_possible,
)
from bioimageio.spec.model import v0_5


@pytest.mark.parametrize(
    "name,expected",
    [
        ("batch", "batch"),
        ("TIME", "time"),
        ("sample", "index"),
        ("index", "index"),
        ("channel", "channel"),
        ("x", "space"),
        ("space_custom", "space"),
        ("custom", "unknown"),
    ],
)
def test_axis_type_inference(name: str, expected: str):
    axis = Axis.create(name)
    assert axis.id == name.lower()
    assert axis.type == expected
    assert Axis.create(axis) is axis


@pytest.mark.parametrize("axis_type", ["batch", "channel"])
def test_axis_requires_canonical_id(axis_type: Literal["batch", "channel"]):
    with pytest.raises(ValueError, match=f"type '{axis_type}' must have id"):
        _ = Axis(AxisId("wrong"), axis_type)


@dataclass
class AxisDescription:
    id: str = "custom"
    type: Literal["space"] = "space"


@pytest.mark.parametrize(
    "axis,expected",
    [
        ("x", "x"),
        (AxisId("y"), "y"),
        (Axis(AxisId("z"), "space"), "z"),
        (v0_5.SpaceInputAxis(id=AxisId("x"), size=8), "x"),
        (AxisDescription(), "custom"),
    ],
)
def test_axis_like_conversion(axis: AxisLike, expected: str):
    assert axis_id_from_axis_like(axis) == expected
    converted = Axis.create(axis)
    assert converted.id == expected
    assert converted.type == "space"


@pytest.mark.parametrize(
    "minimum,maximum,step,message",
    [
        (-1, None, None, "non-negative"),
        (4, 3, None, "greater than or equal to min"),
        (1, None, 0, "positive"),
        (1, None, -1, "positive"),
    ],
)
def test_invalid_axis_size(
    minimum: int, maximum: int | None, step: int | None, message: str
):
    with pytest.raises(ValueError, match=message):
        _ = AxisSize(minimum, maximum, step)


def test_axis_size_allows_zero_and_equal_bounds():
    assert AxisSize(0, 0, 1).min == 0
    assert AxisSize(4, 4).max == 4


@pytest.mark.parametrize(
    "axis,expected",
    [
        ("x", AxisSize(1)),
        (v0_5.BatchAxis(), AxisSize(1)),
        (v0_5.BatchAxis(size=1), AxisSize(1)),
        (v0_5.ChannelAxis(channel_names=["red", "green"]), AxisSize(2, 2)),
        (v0_5.SpaceInputAxis(size=8), AxisSize(8)),
        (v0_5.TimeInputAxis(size=8), AxisSize(8)),
        (v0_5.IndexInputAxis(size=8), AxisSize(8)),
        (v0_5.IndexInputAxis(size=8, concatenable=False), AxisSize(8, 8)),
        (v0_5.IndexOutputAxis(size=8), AxisSize(8, 8)),
        (
            v0_5.SpaceInputAxis(size=v0_5.ParameterizedSize(min=8, step=4)),
            AxisSize(8, step=4),
        ),
        (
            v0_5.IndexOutputAxis(size=v0_5.DataDependentSize(min=2, max=10)),
            AxisSize(2, 10),
        ),
        (
            v0_5.SpaceOutputAxis(
                size=v0_5.SizeReference(
                    tensor_id=v0_5.TensorId("input"), axis_id=AxisId("x"), offset=2
                )
            ),
            AxisSize(3),
        ),
    ],
)
def test_axis_info_infers_size(axis: AxisLike, expected: AxisSize):
    info = AxisInfo.create(axis)
    assert info.size == expected
    assert info.id == axis_id_from_axis_like(axis)
    assert AxisInfo.create(info) is info


@pytest.mark.parametrize("size", [5, AxisSize(4, 16, 4)])
def test_explicit_axis_size_overrides_description(size: int | AxisSize):
    info = AxisInfo.create(v0_5.SpaceInputAxis(size=8), size=size)
    assert info.size == (AxisSize(5, 5) if isinstance(size, int) else size)


@pytest.mark.parametrize(
    "dims,expected",
    [
        ((), ()),
        (("batch", "time", "index", "channel", "z", "y", "x"), tuple("bticzyx")),
        (("b", "t", "i", "c", "z", "y", "x"), tuple("bticzyx")),
        (("Z", "Y", "X"), ("z", "y", "x")),
        (("batch", "custom", "x"), ("batch", "custom", "x")),
        (("batch", "b", "x"), ("batch", "batch", "x")),
        (("batch", "batch", "x"), ("batch", "batch", "x")),
    ],
)
def test_single_letter_dims(dims: tuple[str, ...], expected: tuple[str, ...]):
    assert single_letter_dims_if_possible(tuple(map(AxisId, dims))) == expected
