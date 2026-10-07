from __future__ import annotations

from typing import Any, Literal, cast

import dask.array as da
import numpy as np
import pytest
from numpy.typing import NDArray

from bioimageio.core import AxisId
from bioimageio.core.tensor import MultiIndex

UNSTACKED_SHAPE = {AxisId("batch"): 2, AxisId("z"): 3, AxisId("x"): 4}
TRANSPOSE_TO = (AxisId("batch"), AxisId("x"))
ARRAY_BACKENDS = pytest.mark.parametrize(
    "backend",
    ["numpy", "dask"],
)
ArrayBackend = Literal["numpy", "dask"]


def _array(backend: ArrayBackend, data: NDArray[Any]) -> NDArray[Any] | da.Array:
    return (
        cast(da.Array, da.from_array(data, chunks="auto"))
        if backend == "dask"
        else data
    )


@ARRAY_BACKENDS
def test_create_tensor_with_multiindex_from_unstacked_data(backend: ArrayBackend):
    unstacked_data = _array(
        backend,
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
    )

    tensor = create_tensor_with_multiindex(
        UNSTACKED_SHAPE,
        TRANSPOSE_TO,
        unstacked_data=unstacked_data,
    )

    assert tensor.dims == TRANSPOSE_TO
    assert tensor.shape == (6, 4)
    np.testing.assert_array_equal(
        tensor.to_numpy(), np.asarray(unstacked_data).reshape(6, 4)
    )
    batch_index = tensor.data.indexes[AxisId("batch")]  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(batch_index, MultiIndex)
    assert tuple(str(level) for level in batch_index.names) == ("original_batch", "z")


def test_create_tensor_with_multiindex_fills_unstacked_tensor():
    tensor = create_tensor_with_multiindex(
        UNSTACKED_SHAPE,
        TRANSPOSE_TO,
        fill_value=7,
        dtype=np.dtype(np.int16),
    )

    assert tensor.dtype == "int16"
    np.testing.assert_array_equal(tensor.to_numpy(), np.full((6, 4), 7, dtype=np.int16))


@ARRAY_BACKENDS
def test_create_tensor_with_multiindex_from_stacked_data(backend: ArrayBackend):
    stacked_data = _array(
        backend,
        np.arange(6 * 4, dtype=np.float32).reshape(6, 4),
    )

    tensor = create_tensor_with_multiindex(
        UNSTACKED_SHAPE,
        TRANSPOSE_TO,
        stacked_data=stacked_data,
    )

    assert tensor.dtype == "float32"
    np.testing.assert_array_equal(tensor.to_numpy(), stacked_data)


@ARRAY_BACKENDS
def test_create_tensor_with_multiindex_rejects_incompatible_inputs(
    backend: ArrayBackend,
):
    with pytest.raises(ValueError, match="may not be provided together"):
        _ = create_tensor_with_multiindex(
            UNSTACKED_SHAPE,
            TRANSPOSE_TO,
            unstacked_data=_array(backend, np.zeros((2, 3, 4))),
            stacked_data=_array(backend, np.zeros((6, 4))),
        )

    with pytest.raises(ValueError, match="dtype must be specified"):
        _ = create_tensor_with_multiindex(UNSTACKED_SHAPE, TRANSPOSE_TO)

    with pytest.raises(ValueError, match="does not match"):
        _ = create_tensor_with_multiindex(
            UNSTACKED_SHAPE,
            TRANSPOSE_TO,
            stacked_data=_array(backend, np.zeros((2, 3, 4))),
        )
