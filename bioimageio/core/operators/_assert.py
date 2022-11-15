from typing import Sequence

import xarray as xr


def assert_shape(tensor: xr.DataArray, shape: Sequence[int]) -> xr.DataArray:
    assert tensor.shape == tuple(shape)
    return tensor
