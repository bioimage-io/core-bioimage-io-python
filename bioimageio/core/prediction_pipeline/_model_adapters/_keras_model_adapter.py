import warnings
from typing import List, Optional

import keras
import xarray as xr

from bioimageio.spec.model import nodes
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
        self.model = keras.models.load_model(weight_file)

    def forward(self, input_tensor: xr.DataArray) -> xr.DataArray:
        res = self.model.predict(input_tensor.data)
        return res
