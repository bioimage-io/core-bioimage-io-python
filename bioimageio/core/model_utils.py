from functools import singledispatch
from typing import Any, List, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.utils import download

# @singledispatch
# def is_valid_tensor(description: object, tensor: Union[NDArray[Any], xr.DataArray]) -> bool:
#     raise NotImplementedError(type(description))

# is_valid_tensor.register
# def _(description: v0_4.InputTensor, tensor: Union[NDArray[Any], xr.DataArray]):


@singledispatch
def get_test_input_tensors(model: object) -> List[xr.DataArray]:
    raise NotImplementedError(type(model))


@get_test_input_tensors.register
def _(model: v0_4.Model):
    data = [np.load(download(ipt).path) for ipt in model.test_inputs]
    assert all(isinstance(d, np.ndarray) for d in data)


# @get_test_input_tensors.register
# def _(model: v0_5.Model):
