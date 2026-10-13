from __future__ import annotations

import json
import tempfile
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId, SampleId
from bioimageio.core.remote_backends.gradio.serializer import (
    GradioSampleSerializer,
    SerializedSampleBlock,
)
from bioimageio.core.sample import Sample
from bioimageio.core.stat_measures import SampleMean
from bioimageio.core.tensor import Tensor


@pytest.fixture(autouse=True)
def local_temp_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        tempfile,
        "NamedTemporaryFile",
        partial(tempfile.NamedTemporaryFile, dir=tmp_path),
    )


def make_sample(sample_id: SampleId, lazy: bool = False) -> Sample:
    member = MemberId("raw")
    raw = Tensor(np.arange(24, dtype=np.int16).reshape(4, 6), dims=["y", "x"])
    value = Tensor(np.arange(6, dtype=np.float32), dims=["x"])
    if lazy:
        raw = raw.chunk({AxisId("y"): 2, AxisId("x"): 3})
        value = value.as_dask()
    return Sample(
        id=sample_id,
        members={
            member: raw,
            MemberId("mask"): Tensor(np.eye(6, 4, dtype=bool), dims=["x", "y"]),
        },
        stat={
            SampleMean(member_id=member): 11.5,
            SampleMean(member_id=member, axes=(AxisId("y"),)): value,
        },
    )


def assert_same_sample(actual: Sample, expected: Sample):
    assert actual.id == expected.id
    assert actual.members.keys() == expected.members.keys()
    for member, tensor in expected.members.items():
        restored = actual.members[member]
        assert restored.dims == tensor.dims
        assert restored.dtype == tensor.dtype
        np.testing.assert_array_equal(restored.to_numpy(), tensor.to_numpy())
    assert actual.stat.keys() == expected.stat.keys()
    for measure, value in expected.stat.items():
        restored_value = actual.stat[measure]
        if isinstance(value, Tensor):
            assert isinstance(restored_value, Tensor)
            assert restored_value.dims == value.dims
            np.testing.assert_array_equal(restored_value.to_numpy(), value.to_numpy())
        else:
            assert restored_value == value


@pytest.mark.parametrize("sample_id", [None, "sample", 7, ("sample", (7, None))])
@pytest.mark.parametrize("lazy", [False, True])
def test_json_round_trip(sample_id: SampleId, lazy: bool):
    sample = make_sample(sample_id, lazy)
    serialized = GradioSampleSerializer.serialize_sample(sample)
    assert isinstance(serialized, dict)
    payload = cast(SerializedSampleBlock, json.loads(json.dumps(serialized)))
    restored = GradioSampleSerializer.deserialize_sample(payload)
    assert_same_sample(restored, sample)


def test_file_paths():
    sample = make_sample("sample")
    payload = GradioSampleSerializer.serialize_sample_block(sample)
    data = payload["data"]
    assert isinstance(data, dict)
    for member, file in data.items():
        assert isinstance(file, dict)
        data[member] = file["path"]
    assert_same_sample(GradioSampleSerializer.deserialize_sample_block(payload), sample)


def test_nested_block_round_trip():
    sample = make_sample("sample")
    _ = sample.members.pop(MemberId("mask"))
    serialized = GradioSampleSerializer.serialize_sample_with_fixed_blocking(
        sample,
        block_shapes={MemberId("raw"): {AxisId("y"): 2, AxisId("x"): 3}},
        halo={},
    )
    assert isinstance(serialized, list)
    restored = GradioSampleSerializer.deserialize_sample(serialized)
    assert_same_sample(restored, sample)


def test_mismatched_members():
    payload = GradioSampleSerializer.serialize_sample_block(make_sample("sample"))
    dims = payload["dims"]
    assert isinstance(dims, dict)
    _ = dims.pop("raw")
    with pytest.raises(ValueError, match="different members"):
        _ = GradioSampleSerializer.deserialize_sample_block(payload)
