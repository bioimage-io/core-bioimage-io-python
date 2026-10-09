from __future__ import annotations

from typing import Any, Literal, cast

import dask.array as da
import numpy as np
import pytest
from numpy.typing import NDArray

from bioimageio.core import AxisId
from bioimageio.core.tensor import MultiIndex, Tensor

UNSTACKED_SHAPE = {AxisId("batch"): 2, AxisId("z"): 3, AxisId("x"): 4}
TRANSPOSE_TO = (AxisId("batch"), AxisId("x"))
ARRAY_BACKENDS = pytest.mark.parametrize("backend", ["numpy", "dask"])
ArrayBackend = Literal["numpy", "dask"]


def _array(backend: ArrayBackend, data: NDArray[Any]) -> NDArray[Any] | da.Array:
    return (
        cast(da.Array, da.from_array(data, chunks="auto"))
        if backend == "dask"
        else data
    )


@ARRAY_BACKENDS
def test_stack_and_unstack_batch_multi_index(backend: ArrayBackend):
    data = np.arange(24).reshape(2, 3, 4)
    tensor = Tensor.from_array(_array(backend, data), dims=tuple(UNSTACKED_SHAPE))

    stacked, dim_order = tensor.transpose_and_return_unstacked_dim_order(
        TRANSPOSE_TO, extra_dims="stack"
    )

    assert stacked.dims == TRANSPOSE_TO
    assert stacked.shape == (6, 4)
    assert dim_order == list(UNSTACKED_SHAPE)
    assert isinstance(stacked.data.data, da.Array if backend == "dask" else np.ndarray)
    np.testing.assert_array_equal(stacked.to_numpy(), data.reshape(6, 4))
    index = stacked.batch_multi_index
    assert isinstance(index, MultiIndex)
    assert tuple(index.names) == ("original_batch", "z")
    assert list(index) == [(b, z) for b in range(2) for z in range(3)]

    for restored in (
        stacked.unstack_batch_multi_index(unstacked_dim_order=dim_order),
        stacked.transpose(tuple(UNSTACKED_SHAPE), missing_dims="unstack"),
    ):
        assert restored.dims == tensor.dims
        assert restored.batch_multi_index is None
        np.testing.assert_array_equal(restored.to_numpy(), data)


@ARRAY_BACKENDS
def test_assign_batch_multi_index_to_stacked_data(backend: ArrayBackend):
    data = np.arange(24, dtype=np.float32).reshape(6, 4)
    tensor = Tensor.from_array(_array(backend, data), dims=TRANSPOSE_TO)
    index = MultiIndex.from_product([range(2), range(3)], names=["original_batch", "z"])

    indexed = tensor.assign_batch_multi_index(index)

    assert tensor.batch_multi_index is None
    assert indexed.batch_multi_index is not None
    assert indexed.batch_multi_index.equals(index)
    assert indexed.dtype == "float32"
    np.testing.assert_array_equal(indexed.to_numpy(), data)
    restored = indexed.unstack_batch_multi_index()
    assert restored.dims == tuple(UNSTACKED_SHAPE)
    np.testing.assert_array_equal(restored.to_numpy(), data.reshape(2, 3, 4))


@ARRAY_BACKENDS
def test_stack_without_original_batch(backend: ArrayBackend):
    data = np.arange(12).reshape(3, 4)
    tensor = Tensor.from_array(_array(backend, data), dims=("z", "x"))

    stacked = tensor.transpose(TRANSPOSE_TO, extra_dims="stack")

    assert stacked.shape == (3, 4)
    index = stacked.batch_multi_index
    assert isinstance(index, MultiIndex)
    assert tuple(index.names) == ("original_batch", "z")
    assert list(index) == [(0, z) for z in range(3)]
    unstacked = stacked.unstack_batch_multi_index()
    assert unstacked.dims == tuple(UNSTACKED_SHAPE)
    assert unstacked.shape == (1, 3, 4)
    np.testing.assert_array_equal(unstacked.to_numpy(), data[np.newaxis])
    restored = stacked.transpose(tensor.dims, missing_dims="unstack")
    assert restored.dims == tensor.dims
    np.testing.assert_array_equal(restored.to_numpy(), data)


@ARRAY_BACKENDS
def test_multi_index_rejects_incompatible_axes(backend: ArrayBackend):
    tensor = Tensor.from_array(_array(backend, np.zeros((3, 4))), dims=("z", "x"))
    index = MultiIndex.from_product([range(3)], names=["z"])

    with pytest.raises(ValueError, match="without a 'batch' axis"):
        _ = tensor.assign_batch_multi_index(index)

    with pytest.raises(ValueError, match="Cannot stack extra dimensions to batch"):
        _ = tensor.transpose(("x",), extra_dims="stack")

    with pytest.raises(ValueError, match="does not have a MultiIndex"):
        _ = Tensor.from_array(
            _array(backend, np.zeros((6, 4))), dims=TRANSPOSE_TO
        ).transpose(tuple(UNSTACKED_SHAPE), missing_dims="unstack")
