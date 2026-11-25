from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from ..axis import AxisId
from ..common import QuantileMethod
from ..tensor import Tensor
from .type_hints import cast_and_assert


def compute_quantiles(
    tensor: Tensor,
    q: float | Sequence[float],
    axes: Sequence[AxisId] | None,
    method: QuantileMethod,
) -> Tensor:
    tensor = tensor.as_dask()
    # Exact quantiles need all values along each reduced axis in the same chunk.
    tensor = Tensor.from_xarray(
        cast_and_assert(
            xr.DataArray,
            tensor.data.chunk(
                {d: -1 for d in tensor.dims if axes is None or d in axes}
            ),
        )
    )
    return tensor.quantile(q, dim=axes, method=method)
