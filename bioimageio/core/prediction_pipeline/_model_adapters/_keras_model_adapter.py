import warnings
from typing import List, Optional, Sequence

import keras
import xarray as xr

from ._model_adapter import ModelAdapter


class KerasModelAdapter(ModelAdapter):
    def _load(self, *, devices: Optional[Sequence[str]] = None) -> None:
        # TODO keras device management
        if devices is not None:
            warnings.warn(f"Device management is not implemented for keras yet, ignoring the devices {devices}")

        weight_file = self.bioimageio_model.weights["keras_hdf5"].source
        self._model = keras.models.load_model(weight_file)
        self._output_axes = [tuple(out.axes) for out in self.bioimageio_model.outputs]

    def _unload(self) -> None:
        warnings.warn("Device management is not implemented for keras yet, cannot unload model")

    def _forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        result = self._model.predict(*input_tensors)
        if not isinstance(result, (tuple, list)):
            result = [result]

        assert len(result) == len(self._output_axes)
        return [xr.DataArray(r, dims=axes) for r, axes, in zip(result, self._output_axes)]
