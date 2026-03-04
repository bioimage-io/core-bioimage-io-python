import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Sequence, Union

from keras.src.legacy.saving import (  # pyright: ignore[reportMissingTypeStubs]
    legacy_h5_format,
)
from loguru import logger
from numpy.typing import NDArray

from bioimageio.core.utils._compare import warn_about_version
from bioimageio.spec._internal.version_type import Version
from bioimageio.spec.model import v0_4, v0_5

from .._settings import settings
from ..digest_spec import get_axes_infos
from ..utils._type_guards import is_list, is_tuple
from ._model_adapter import ModelAdapter

os.environ["KERAS_BACKEND"] = settings.keras_backend


# by default, we use the keras integrated with tensorflow
# TODO: check if we should prefer keras
try:
    import tensorflow as tf  # pyright: ignore[reportMissingTypeStubs]
    from tensorflow import (  # pyright: ignore[reportMissingTypeStubs]
        keras,  # pyright: ignore[reportUnknownVariableType,reportAttributeAccessIssue]
    )

    tf_version = Version(tf.__version__)
except Exception:
    import keras  # pyright: ignore[reportMissingTypeStubs]

    tf_version = None


class KerasModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(model_description=model_description)

        if (
            not isinstance(model_description, v0_4.ModelDescr)
            and model_description.weights.keras_v3 is not None
        ):
            weight_reader = model_description.weights.keras_v3.get_reader()
            backend, backend_version = model_description.weights.keras_v3.backend
        elif model_description.weights.keras_hdf5 is not None:
            backend = "legacy_tensorflow"
            backend_version = model_description.weights.keras_hdf5.tensorflow_version
            weight_reader = model_description.weights.keras_hdf5.get_reader()
        else:
            raise ValueError("model has no Keras weights")

        if backend != "legacy_tensorflow" and backend != settings.keras_backend:
            logger.warning(
                "Model specifies Keras backend '{}', but environment variable KERAS_BACKEND is set to '{}'."
                + " Attempting to load model with KERAS_BACKEND='{}' (this may fail if the model is not compatible with this backend).",
                backend,
                settings.keras_backend,
                settings.keras_backend,
            )

        if (backend == "legacy_tensorflow") or (
            backend == settings.keras_backend == "tensorflow"
        ):
            warn_about_version("tensorflow", backend_version, tf_version)
        elif backend == settings.keras_backend == "torch":
            import torch

            torch_version = Version(torch.__version__)
            warn_about_version("torch", backend_version, torch_version)
        elif backend == settings.keras_backend == "jax":
            import jax

            jax_version = Version(jax.__version__)
            warn_about_version("jax", backend_version, jax_version)

        # TODO keras device management
        if devices is not None:
            logger.warning(
                "Device management is not implemented for keras yet, ignoring the devices {}",
                devices,
            )

        if weight_reader.suffix in (".h5", "hdf5"):
            import h5py  # pyright: ignore[reportMissingTypeStubs]

            h5_file = h5py.File(weight_reader, mode="r")
            self._network = legacy_h5_format.load_model_from_hdf5(h5_file)
        else:
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / weight_reader.original_file_name
                with temp_path.open("wb") as f:
                    shutil.copyfileobj(weight_reader, f)

                self._network = keras.models.load_model(temp_path)

        self._output_axes = [
            tuple(a.id for a in get_axes_infos(out))
            for out in model_description.outputs
        ]

    def _forward_impl(  # pyright: ignore[reportUnknownParameterType]
        self, input_arrays: Sequence[Optional[NDArray[Any]]]
    ):
        network_output = self._network.predict(*input_arrays)  # type: ignore
        if is_list(network_output) or is_tuple(network_output):
            return network_output
        else:
            return [network_output]  # pyright: ignore[reportUnknownVariableType]

    def unload(self) -> None:
        logger.warning(
            "Device management is not implemented for keras yet, cannot unload model"
        )
