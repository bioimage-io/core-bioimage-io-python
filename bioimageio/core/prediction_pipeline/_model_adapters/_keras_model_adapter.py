import warnings
from typing import List, Optional, Sequence
from marshmallow import missing

# by default, we use the keras integrated with tensorflow
try:
    from tensorflow import keras
    import tensorflow as tf

    TF_VERSION = tf.__version__
except Exception:
    import keras

    TF_VERSION = None
import xarray as xr

from ._model_adapter import ModelAdapter


class KerasModelAdapter(ModelAdapter):
    def _load(self, *, devices: Optional[Sequence[str]] = None) -> None:
        model_tf_version = self.bioimageio_model.weights["keras_hdf5"].tensorflow_version
        if model_tf_version is missing:
            model_tf_version = None
        else:
            model_tf_version = (int(model_tf_version.major), int(model_tf_version.minor))

        if TF_VERSION is None or model_tf_version is None:
            warnings.warn("Could not check tensorflow versions. The prediction results may be wrong.")
        elif tuple(model_tf_version[:2]) != tuple(map(int, TF_VERSION.split(".")))[:2]:
            warnings.warn(
                f"Model tensorflow version {model_tf_version} does not match {TF_VERSION}."
                "The prediction results may be wrong"
            )

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
