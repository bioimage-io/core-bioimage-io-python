from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from keras.src.legacy.saving import (  # pyright: ignore[reportMissingTypeStubs]
    legacy_h5_format,
)
from loguru import logger
from numpy.typing import NDArray

from bioimageio.spec._internal.version_type import Version
from bioimageio.spec.model import v0_4

from .._model_adapter import LocalModelAdapter
from .._settings import settings
from ..utils._compare import warn_about_version
from ..utils._type_guards import is_list, is_tuple

os.environ["KERAS_BACKEND"] = settings.keras_backend


# by default, we use the keras integrated with tensorflow
# TODO: check if we should prefer keras
try:
    import tensorflow as tf
    from tensorflow import keras

    tf_version = Version(tf.__version__)  # pyright: ignore[reportUnknownArgumentType]
except Exception:
    import keras  # pyright: ignore[reportMissingTypeStubs]

    tf_version = None


class KerasModelAdapter(LocalModelAdapter[None, Any]):
    def _parse_devices(self, devices: Sequence[str] | None) -> tuple[None]:
        # TODO keras device management
        if devices is not None:
            logger.warning(
                "Device management is not implemented for keras yet, ignoring the devices {}",
                devices,
            )
        return (None,)

    def _init_model_on_device(self, device: None) -> Any:
        if (
            not isinstance(self._model_descr, v0_4.ModelDescr)
            and self._model_descr.weights.keras_v3 is not None
        ):
            weight_reader = self._model_descr.weights.keras_v3.get_reader()
            backend, backend_version = self._model_descr.weights.keras_v3.backend
        elif self._model_descr.weights.keras_hdf5 is not None:
            backend = "legacy_tensorflow"
            backend_version = self._model_descr.weights.keras_hdf5.tensorflow_version
            weight_reader = self._model_descr.weights.keras_hdf5.get_reader()
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

        if weight_reader.suffix in (".h5", "hdf5"):
            import h5py  # pyright: ignore[reportMissingTypeStubs]

            h5_file = h5py.File(weight_reader, mode="r")
            return legacy_h5_format.load_model_from_hdf5(h5_file)  # pyright: ignore[reportUnknownVariableType]
        else:
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / weight_reader.original_file_name
                with temp_path.open("wb") as f:
                    shutil.copyfileobj(weight_reader, f)

                return keras.models.load_model(temp_path)  # pyright: ignore[reportUnknownVariableType]

    def _forward_impl(
        self,
        device: None,
        model: Any,
        input_arrays: Sequence[NDArray[Any] | None],
    ):
        network_output = model.predict(*input_arrays)
        if is_list(network_output) or is_tuple(network_output):
            return network_output
        else:
            return [network_output]

    def _cleanup_pre_model_deletion(self, device: None, model: Any) -> None:
        return

    def _cleanup_post_model_deletion(self, device: None) -> None:
        return
