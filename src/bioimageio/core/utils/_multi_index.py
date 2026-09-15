from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import assert_never

from ..axis import BATCH_AXIS_ID, AxisId, PerAxis
from ..common import PerMember
from ._type_guards import is_ndarray

if TYPE_CHECKING:
    from ..tensor import Tensor

MultiIndex = pd.MultiIndex


def create_tensor_with_multiindex(
    unstacked_shape: PerAxis[int],
    transpose_to: Sequence[AxisId],
    *,
    fill_value: float | None = None,
    dtype: np.dtype | None = None,
    unstacked_data: NDArray[Any] | da.Array | None = None,
    stacked_data: NDArray[Any] | da.Array | None = None,
) -> Tensor:
    """Create a tensor potentially with a MultiIndex in the batch dimension.

    The tensor is created with the given unstacked shape and then transposed to `transpose_to`.
    If `transpose_to` is missing some of the unstacked_shape dimensions a MultiIndex will be created.
    Otherwise the tensor will be created without a batch MultiIndex.

    Args:
        unstacked_shape: The shape of the tensor before stacking.
        transpose_to: The order of the axes after transposing.
        fill_value: The value to fill the tensor with. If None, the tensor will be empty (ignored if unstacked_data or stacked_data are not None).
        dtype: The data type of the tensor. If None, the data type will be inferred from unstacked_data or stacked_data (must be specified if neither unstacked_data nor stacked_data is provided).
        unstacked_data: The data to use for the unstacked tensor. If provided, fill_value will be ignored. May not be provided together with stacked_data.
        stacked_data: The data to use for the stacked tensor. If provided, fill_value will be ignored. May not be provided together with unstacked_data.

    Returns:
        A tensor (with a MultiIndex in the batch dimension).
    """
    from ..tensor import Tensor

    if unstacked_data is not None and stacked_data is not None:
        raise ValueError("unstacked_data and stacked_data may not be provided together")

    if unstacked_data is None:
        if stacked_data is not None:
            dtype = stacked_data.dtype  # pyright: ignore[reportUnknownVariableType]
        elif dtype is None:
            raise ValueError(
                "dtype must be specified if neither unstacked_data nor stacked_data is provided"
            )

        if fill_value is None or stacked_data is not None:
            unstacked_data = da.zeros(  # pyright: ignore[reportUnknownVariableType]
                shape=tuple(unstacked_shape.values()), dtype=dtype
            )
        else:
            unstacked_data = da.full(  # pyright: ignore[reportUnknownVariableType]
                shape=tuple(unstacked_shape.values()),
                fill_value=fill_value,
                dtype=dtype,
            )

        assert isinstance(unstacked_data, da.Array)
    elif dtype is not None:
        unstacked_data = unstacked_data.astype(dtype)  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(unstacked_data, (np.ndarray, da.Array))
    else:
        dtype = unstacked_data.dtype  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(dtype, np.dtype)

    if isinstance(unstacked_data, da.Array):
        tensor = Tensor.from_array(unstacked_data, dims=tuple(unstacked_shape.keys()))
    elif isinstance(unstacked_data, np.ndarray):
        assert is_ndarray(unstacked_data)
        tensor = Tensor.from_numpy(unstacked_data, dims=tuple(unstacked_shape.keys()))
    else:
        assert_never(unstacked_data)

    stacked_tensor = tensor.transpose(
        transpose_to, extra_dims="stack", missing_dims="expand"
    )
    if stacked_data is not None:
        if stacked_data.shape != stacked_tensor.data.shape:
            raise ValueError(
                f"stacked_data shape {stacked_data.shape} does not match the stacked tensor shape {stacked_tensor.data.shape}"
            )
        stacked_tensor.data.data = stacked_data

    return stacked_tensor


def restore_batch_multi_index(
    inputs: PerMember[Tensor | None], outputs: PerMember[Tensor | None]
) -> PerMember[Tensor | None]:
    """Restore the first batch multi-index found in the inputs to all outputs with batch dimension."""
    for tensor in inputs.values():
        if tensor is None:
            continue

        idx = tensor.data.indexes.get(AxisId("batch"))  # pyright: ignore[reportUnknownVariableType]
        if isinstance(idx, MultiIndex):
            outputs = {
                k: v.assign_batch_multi_index(idx)
                if v is not None and AxisId("batch") in v.dims
                else v
                for k, v in outputs.items()
            }
            break

    return outputs


def get_unstacked_dim_order(
    stacked_order: Sequence[AxisId],
    *,
    unstacked_reference_order: Sequence[AxisId] | None,
    stacked_dims: Sequence[AxisId] | None,
) -> list[AxisId]:
    if not stacked_dims and unstacked_reference_order is None:
        return list(stacked_order)
    if unstacked_reference_order is not None and list(stacked_order) == [
        d for d in unstacked_reference_order if d in stacked_order
    ]:
        # no differences in non-stacking axes, so we can preserve the unstacked axis order
        return [
            d
            for d in unstacked_reference_order
            if d in stacked_order or (stacked_dims is not None and d in stacked_dims)
        ]
    elif stacked_dims is None:
        # no stacked dims provided, so we can just return the stacked_out_order
        return list(stacked_order)
    elif BATCH_AXIS_ID not in stacked_order:
        raise ValueError(
            f"stacked_order={stacked_order} does not contain a batch axis, but stacked_dims={stacked_dims} are provided"
        )
    else:
        # unstack batch axis in its place
        ret = list(stacked_order)
        b_idx = ret.index(AxisId("batch"))
        ret[b_idx : b_idx + 1] = (
            [
                d for d in unstacked_reference_order or () if d in stacked_dims
            ]  # keep unstacked axis order from reference
            + [
                d
                for d in stacked_dims
                if unstacked_reference_order is None
                or d not in unstacked_reference_order
            ]  # add any remaining stacked dims that are not in the unstacked reference order
        )
        return ret


def get_unstacked_shape(
    stacked_shape: PerAxis[int],
    *,
    unstacked_dim_order: Sequence[AxisId] | None,
    unstacked_batch_shape: PerAxis[int] | None,
) -> PerAxis[int]:
    """Combine a shape with an optional unstacked batch shape and apply an optional unstacked dim order to get the final unstacked shape.

    Returns `stacked_shape` if both `unstacked_dim_order` and `unstacked_batch_shape` are None.
    """
    unstacked_shape_unordered = {**stacked_shape, **(unstacked_batch_shape or {})}

    if unstacked_dim_order is not None:
        assert all(a in unstacked_dim_order for a in unstacked_shape_unordered), (
            f"unstacked_dim_order={unstacked_dim_order} does not contain all axes of the sample shape {stacked_shape} and unstacked sample batch shape {unstacked_batch_shape}"
        )
        order = unstacked_dim_order
    else:
        # if no explicit unstacked dim order is provided we insert the unstacked batch axes at the position of the batch axis in the sample shape
        order = list(stacked_shape)
        if unstacked_batch_shape:
            assert BATCH_AXIS_ID in stacked_shape, (
                f"unstacked sample batch shape {unstacked_batch_shape} is provided but the sample shape {stacked_shape} does not contain a batch axis"
            )
            b_idx = order.index(BATCH_AXIS_ID)
            order[b_idx : b_idx + 1] = list(unstacked_batch_shape)

    assert all(a in order for a in unstacked_shape_unordered), (
        f"unstacked_dim_order={unstacked_dim_order} does not contain all axes of the sample shape {stacked_shape} and unstacked sample batch shape {unstacked_batch_shape}"
    )
    return {a: unstacked_shape_unordered[a] for a in order}
