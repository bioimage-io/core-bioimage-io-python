from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId, SliceInfo
from bioimageio.core.proc_ops import Clip
from bioimageio.core.sample import Sample
from bioimageio.core.tensor import Tensor


@pytest.fixture
def sample() -> Sample:
    raw, mask = MemberId("raw"), MemberId("mask")
    result = Sample(
        id="image",
        members={
            raw: Tensor(np.arange(12).reshape(3, 4), dims=("y", "x")),
            mask: Tensor(np.arange(12).reshape(3, 4) % 2, dims=("y", "x")),
        },
        stat={},
    )
    Clip(raw, raw, min=0.0, max=10.0)(result)
    return result


@pytest.mark.parametrize("index", [slice(1, 3), SliceInfo(1, 3), 1])
def test_slice_all_members_by_axis(sample: Sample, index: slice | SliceInfo | int):
    result = sample[{AxisId("x"): index}]
    assert result.id == sample.id
    assert result.stat is sample.stat
    assert result.applied_ops == sample.applied_ops
    assert set(result.members) == set(sample.members)
    for member, tensor in sample.members.items():
        expected_index = slice(1, 3) if isinstance(index, SliceInfo) else index
        np.testing.assert_array_equal(
            result.members[member].to_numpy(), tensor.to_numpy()[:, expected_index]
        )
        assert tensor.shape == (3, 4)


def test_slice_selected_member(sample: Sample):
    raw = MemberId("raw")
    result = sample[{raw: {AxisId("y"): slice(1, None)}}]
    assert set(result.members) == {raw}
    np.testing.assert_array_equal(
        result.members[raw].to_numpy(), sample.members[raw].to_numpy()[1:]
    )


def test_boolean_mask_applies_to_all_members(sample: Sample):
    mask = Tensor(np.array([True, False, True]), dims=("y",))
    result = sample[mask]
    for member, tensor in sample.members.items():
        np.testing.assert_array_equal(
            result.members[member].to_numpy(), tensor.to_numpy()[[0, 2]]
        )


def test_empty_selection(sample: Sample):
    assert sample[{}].members == {}


@pytest.mark.parametrize(
    "key,error,message",
    [
        ({"x": slice(None)}, TypeError, "Only keys of type MemberId or AxisId"),
        ({AxisId("x"): "invalid"}, TypeError, "Per-axis keys"),
        ({MemberId("raw"): "invalid"}, TypeError, "Per-member keys"),
        ({MemberId("raw"): {"x": slice(None)}}, TypeError, "keys of type AxisId"),
        (
            {MemberId("raw"): {AxisId("x"): "invalid"}},
            TypeError,
            "Per-axis mappings only support value types",
        ),
        (
            {MemberId("raw"): slice(None), AxisId("x"): slice(None)},
            ValueError,
            "Mixing both MemberId and AxisId",
        ),
    ],
)
def test_slice_rejects_invalid_keys(
    sample: Sample, key: Any, error: type[Exception], message: str
):
    with pytest.raises(error, match=message):
        _ = sample[key]
