from typing import Any, List, Optional, Sequence, Union

from loguru import logger
from numpy.typing import NDArray

from bioimageio.core.common import Tensor
from bioimageio.spec._internal.io_utils import download
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import Version

from ._model_adapter import ModelAdapter

# by default, we use the keras integrated with tensorflow
try:
    import tensorflow as tf  # pyright: ignore[reportMissingImports]
    from tensorflow import (  # pyright: ignore[reportMissingImports]
        keras,  # pyright: ignore[reportUnknownVariableType]
    )

    tf_version = Version(tf.__version__)  # pyright: ignore[reportUnknownArgumentType]
except Exception:
    try:
        import keras  # pyright: ignore[reportMissingImports]
    except Exception:
        keras = None

    tf_version = None


class KerasModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ) -> None:
        assert keras is not None
        super().__init__()
        if model_description.weights.keras_hdf5 is None:
            raise ValueError("model has not keras_hdf5 weights specified")
        model_tf_version = model_description.weights.keras_hdf5.tensorflow_version

        if tf_version is None or model_tf_version is None:
            logger.warning("Could not check tensorflow versions.")
        elif model_tf_version > tf_version:
            logger.warning(
                "The model specifies a newer tensorflow version than installed: {} > {}.",
                model_tf_version,
                tf_version,
            )
        elif (model_tf_version.major, model_tf_version.minor) != (
            tf_version.major,
            tf_version.minor,
        ):
            logger.warning(
                "Model tensorflow version {} does not match {}.",
                model_tf_version,
                tf_version,
            )

        # TODO keras device management
        if devices is not None:
            logger.warning(
                "Device management is not implemented for keras yet, ignoring the devices {}",
                devices,
            )

        weight_path = download(model_description.weights.keras_hdf5.source).path

        self._network = keras.models.load_model(weight_path)
        self._output_axes = [tuple(out.axes) for out in model_description.outputs]

    def forward(self, *input_tensors: Optional[Tensor]) -> List[Optional[Tensor]]:
        _result: Union[Sequence[NDArray[Any]], NDArray[Any]]
        _result = self._network.predict(  # pyright: ignore[reportUnknownVariableType]
            *input_tensors
        )
        if isinstance(_result, (tuple, list)):
            result: Sequence[NDArray[Any]] = _result
        else:
            result = [_result]  # type: ignore

        assert len(result) == len(self._output_axes)
        return [Tensor(r, dims=axes) for r, axes, in zip(result, self._output_axes)]

    def unload(self) -> None:
        logger.warning(
            "Device management is not implemented for keras yet, cannot unload model"
        )
