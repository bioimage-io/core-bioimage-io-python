from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_equal  # pyright: ignore[reportUnknownVariableType]

from bioimageio.core import AxisId, Tensor


@pytest.mark.parametrize(
    "axes",
    [
        "yx",
        "xy",
        "cyx",
        "yxc",
        "bczyx",
        "xyz",
        "xyzc",
        "bzyxc",
        ("batch", "channel", "x", "y"),
    ],
)
def test_transpose_tensor_2d(axes: Sequence[str]):
    tensor = Tensor.from_numpy(np.random.rand(256, 256), dims=None)
    transposed = tensor.transpose([AxisId(a) for a in axes])
    assert transposed.ndim == len(axes)


@pytest.mark.parametrize(
    "axes",
    [
        "zyx",
        "cyzx",
        "yzixc",
        "bczyx",
        "xyz",
        "xyzc",
        "bzyxtc",
        ("batch", "channel", "x", "y", "z"),
    ],
)
def test_transpose_tensor_3d(axes: Sequence[str]):
    tensor = Tensor.from_numpy(np.random.rand(64, 64, 64), dims=None)
    transposed = tensor.transpose([AxisId(a) for a in axes])
    assert transposed.ndim == len(axes)


def test_crop_and_pad():
    tensor = Tensor.from_xarray(
        xr.DataArray(np.random.rand(10, 20), dims=("x", "y"), name="id")
    )
    padded = tensor.pad({AxisId("x"): 7, AxisId("y"): (3, 3)})
    cropped = padded.crop_to(tensor.sizes)
    assert_equal(tensor.data, cropped.data)


def test_some_magic_ops():
    tensor = Tensor.from_numpy(np.random.rand(256, 256), dims=None)
    assert tensor + 2 == 2 + tensor


def test_shape_attributes():
    tensor = Tensor.from_numpy(np.random.rand(1, 2, 25, 26), dims=None)
    assert tensor.shape_tuple == tensor.shape


def test_transpose_stack_extra_dims_to_batch_creates_multi_index():
    tensor = Tensor.from_numpy(
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
        dims=("batch", "z", "x"),
    )

    transposed = tensor.transpose(
        [AxisId("batch"), AxisId("x")],
        extra_dims="stack",
        missing_dims="raise",
    )

    assert transposed.dims == (AxisId("batch"), AxisId("x"))
    assert transposed.sizes[AxisId("batch")] == 6
    assert transposed.sizes[AxisId("x")] == 4
    batch_index = transposed.data.indexes[AxisId("batch")]  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(batch_index, pd.MultiIndex)
    assert tuple(str(level) for level in batch_index.names) == ("original_batch", "z")


def test_unstack_batch_multi_index_roundtrip_after_stack():
    tensor = Tensor.from_numpy(
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
        dims=("batch", "z", "x"),
    )

    stacked = tensor.transpose(
        [AxisId("batch"), AxisId("x")],
        extra_dims="stack",
        missing_dims="raise",
    )
    unstacked = stacked.unstack_batch_multi_index()
    restored = unstacked.transpose(
        [AxisId("batch"), AxisId("z"), AxisId("x")],
        extra_dims="raise",
        missing_dims="raise",
    )

    np.testing.assert_array_equal(restored.to_numpy(), tensor.to_numpy())


def test_unstack_batch_multi_index_error_modes():
    no_batch = Tensor.from_numpy(np.arange(6).reshape(2, 3), dims=("y", "x"))
    with pytest.raises(ValueError, match="without a 'batch' axis"):
        _ = no_batch.unstack_batch_multi_index()
    assert no_batch.unstack_batch_multi_index(errors="ignore") is no_batch

    non_multi_index_batch = Tensor.from_numpy(
        np.arange(6).reshape(2, 3), dims=("batch", "x")
    )
    with pytest.raises(ValueError, match="does not have a MultiIndex"):
        _ = non_multi_index_batch.unstack_batch_multi_index()
    assert (
        non_multi_index_batch.unstack_batch_multi_index(errors="ignore")
        is non_multi_index_batch
    )


def test_transpose_unstacks_missing_dims_from_batch():
    tensor = Tensor.from_numpy(
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
        dims=("batch", "z", "x"),
    )
    stacked = tensor.transpose(
        [AxisId("batch"), AxisId("x")],
        extra_dims="stack",
        missing_dims="raise",
    )

    unstacked = stacked.transpose(
        [AxisId("batch"), AxisId("z"), AxisId("x")],
        extra_dims="raise",
        missing_dims="unstack",
    )

    assert unstacked.dims == (AxisId("batch"), AxisId("z"), AxisId("x"))
    np.testing.assert_array_equal(unstacked.to_numpy(), tensor.to_numpy())


def test_transpose_unstack_or_expand_expands_without_multi_index_batch():
    tensor = Tensor.from_numpy(np.arange(2 * 4).reshape(2, 4), dims=("batch", "x"))

    transposed = tensor.transpose(
        [AxisId("batch"), AxisId("z"), AxisId("x")],
        extra_dims="raise",
        missing_dims="unstack_or_expand",
    )

    assert transposed.dims == (AxisId("batch"), AxisId("z"), AxisId("x"))
    assert transposed.sizes[AxisId("batch")] == 2
    assert transposed.sizes[AxisId("z")] == 1
    assert transposed.sizes[AxisId("x")] == 4
