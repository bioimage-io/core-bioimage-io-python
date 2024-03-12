from typing import List

import xarray as xr

from bioimageio.spec.model import AnyModelDescr, v0_4
from bioimageio.spec.utils import load_array


def get_test_inputs(model: AnyModelDescr) -> List[xr.DataArray]:
    if isinstance(model, v0_4.ModelDescr):
        return [
            xr.DataArray(load_array(tt), dims=tuple(d.axes))
            for d, tt in zip(model.inputs, model.test_inputs)
        ]
    else:
        return [
            xr.DataArray(
                load_array(d.test_tensor), dims=tuple(str(a.id) for a in d.axes)
            )
            for d in model.inputs
        ]


def get_test_outputs(model: AnyModelDescr) -> List[xr.DataArray]:
    if isinstance(model, v0_4.ModelDescr):
        return [
            xr.DataArray(load_array(tt), dims=tuple(d.axes))
            for d, tt in zip(model.outputs, model.test_outputs)
        ]
    else:
        return [
            xr.DataArray(
                load_array(d.test_tensor), dims=tuple(str(a.id) for a in d.axes)
            )
            for d in model.outputs
        ]
