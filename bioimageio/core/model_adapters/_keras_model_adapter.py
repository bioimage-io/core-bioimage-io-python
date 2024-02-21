import warnings
from typing import Any, List, Optional, Sequence, Union

from numpy.typing import NDArray
from packaging.version import Version

# by default, we use the keras integrated with tensorflow
try:
    import tensorflow as tf
    from tensorflow import keras

    tf_version = Version(tf.__version__)
except Exception:
    import keras

    tf_version = None
import xarray as xr

from bioimageio.spec._internal.io_utils import download
from bioimageio.spec.model import v0_4, v0_5

from ._model_adapter import ModelAdapter


class KerasModelAdapter(ModelAdapter):
    def __init__(
        self, *, model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr], devices: Optional[Sequence[str]] = None
    ) -> None:
        super().__init__()
        if model_description.weights.keras_hdf5 is None:
            raise ValueError("model has not keras_hdf5 weights specified")
        model_tf_version = model_description.weights.keras_hdf5.tensorflow_version

        if tf_version is None or model_tf_version is None:
            warnings.warn("Could not check tensorflow versions.")
        elif model_tf_version > tf_version:
            warnings.warn(
                f"The model specifies a newer tensorflow version than installed: {model_tf_version} > {tf_version}."
            )
        elif (model_tf_version.major, model_tf_version.minor) != (tf_version.major, tf_version.minor):
            warnings.warn(f"Model tensorflow version {model_tf_version} does not match {tf_version}.")

        # TODO keras device management
        if devices is not None:
            warnings.warn(f"Device management is not implemented for keras yet, ignoring the devices {devices}")

        weight_path = download(model_description.weights.keras_hdf5.source).path

        self._network = keras.models.load_model(weight_path)
        self._output_axes = [tuple(out.axes) for out in model_description.outputs]

    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        _result: Union[  # pyright: ignore[reportUnknownVariableType]
            Sequence[NDArray[Any]], NDArray[Any]
        ] = self._network.predict(*input_tensors)
        if isinstance(_result, (tuple, list)):
            result: Sequence[NDArray[Any]] = _result
        else:
            result = [_result]  # type: ignore

        assert len(result) == len(self._output_axes)
        return [xr.DataArray(r, dims=axes) for r, axes, in zip(result, self._output_axes)]

    def unload(self) -> None:
        warnings.warn("Device management is not implemented for keras yet, cannot unload model")
