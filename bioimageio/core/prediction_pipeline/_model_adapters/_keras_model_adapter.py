import warnings
from typing import List, Optional

import keras
import xarray as xr

from bioimageio.core.resource_io import nodes
from ._model_adapter import ModelAdapter


class KerasModelAdapter(ModelAdapter):
    def __init__(self, *, bioimageio_model: nodes.Model, devices: Optional[List[str]] = None):
        self.spec = bioimageio_model
        self.name = self.spec.name

        # TODO keras device management
        if devices is not None:
            warnings.warn(f"Device management is not implemented for tensorflow yet, ignoring the devices {devices}")
        self.devices = []

        weight_file = self.spec.weights["keras_hdf5"].source
        self._model = keras.models.load_model(weight_file)
        self._output_axes = [tuple(out.axes) for out in bioimageio_model.outputs]

    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        result = self._model.predict(*[ipt.data for ipt in input_tensors])
        if not isinstance(result, (tuple, list)):
            result = [result]

        assert len(result) == len(self._output_axes)
        return [xr.DataArray(r, dims=axes) for r, axes, in zip(result, self._output_axes)]
