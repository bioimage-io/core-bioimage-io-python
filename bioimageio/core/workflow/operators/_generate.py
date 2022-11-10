from typing import Sequence, Union

import numpy as np
import xarray as xr


def generate_random_uniform_tensor(
    shape: Sequence[Union[int, str]], axes: Sequence[str], *, low: Union[int, float] = 0, high: Union[int, float] = 1
) -> xr.DataArray:
    """generate a tensor with uniformly distributed samples in the interval [low, high)
    Returns:
        xr.DataArray: random tensor
    """
    assert len(shape) == len(axes)
    return xr.DataArray(np.random.uniform(low=low, high=high, size=[int(s) for s in shape]), dims=tuple(axes))
