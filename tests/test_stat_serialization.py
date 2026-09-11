from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId
from bioimageio.core.io import load_stat, save_stat, serialize_stat
from bioimageio.core.stat_measures import DatasetMean, SampleQuantile, Stat
from bioimageio.core.tensor import Tensor

ArrayBackend = Literal["numpy", "dask"]
ARRAY_BACKENDS = pytest.mark.parametrize("backend", ["numpy", "dask"])


def _array(backend: ArrayBackend, data: NDArray[Any]) -> NDArray[Any] | da.Array:
    return (
        cast(da.Array, da.from_array(data, chunks="auto"))
        if backend == "dask"
        else data
    )


def _make_stat_with_tensor_value(backend: ArrayBackend) -> Stat:
    member_id = MemberId("test_tensor")
    mean_measure = DatasetMean(member_id=member_id, axes=(AxisId("batch"),))
    quantile_measure = SampleQuantile(member_id=member_id, q=0.5)

    tensor_value = Tensor(
        _array(backend, np.arange(6, dtype=np.float32).reshape(2, 3)),
        dims=(AxisId("x"), AxisId("y")),
    )
    float_value = 3.14

    return {mean_measure: tensor_value, quantile_measure: float_value}


@ARRAY_BACKENDS
def test_serialize_stat_with_tensor_value(backend: ArrayBackend):
    stat = _make_stat_with_tensor_value(backend)

    serialized = serialize_stat(stat)
    # should be plain JSON-compatible data, e.g. round-trippable through json
    json_str = json.dumps(serialized)
    assert json.loads(json_str) == serialized

    loaded = load_stat(serialized)
    assert loaded.keys() == stat.keys()
    for measure, expected_value in stat.items():
        actual_value = loaded[measure]
        if isinstance(expected_value, Tensor):
            assert isinstance(actual_value, Tensor)
            xr.testing.assert_allclose(expected_value.data, actual_value.data)
        else:
            assert actual_value == expected_value


@ARRAY_BACKENDS
def test_save_and_load_stat_with_tensor_value(backend: ArrayBackend, tmp_path: Path):
    stat = _make_stat_with_tensor_value(backend)
    path = tmp_path / "stat.json"

    save_stat(stat, path)
    loaded = load_stat(path)

    assert loaded.keys() == stat.keys()
    for measure, expected_value in stat.items():
        actual_value = loaded[measure]
        if isinstance(expected_value, Tensor):
            assert isinstance(actual_value, Tensor)
            xr.testing.assert_allclose(expected_value.data, actual_value.data)
        else:
            assert actual_value == expected_value
