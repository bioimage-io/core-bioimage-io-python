from typing import Callable

import xarray as xr

Transform = Callable[[xr.DataArray], xr.DataArray]
