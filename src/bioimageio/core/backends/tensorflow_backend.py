from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from loguru import logger
from numpy.typing import NDArray

from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5

from .._model_adapter import LocalModelAdapter
from ..io import ensure_unzipped


class TensorflowModelAdapter(LocalModelAdapter[None, Any]):
    """Adapter for TensorFlow 1 models"""

    weight_format = "tensorflow_saved_model_bundle"

    def __init__(
        self,
        model_description: v0_4.ModelDescr | v0_5.ModelDescr,
        devices: Sequence[str] | None = None,
    ):

        if model_description.weights.tensorflow_saved_model_bundle is None:
            raise ValueError("No `tensorflow_saved_model_bundle` weights found")

        if isinstance(model_description, v0_4.ModelDescr):
            self._weight_src = (
                model_description.weights.tensorflow_saved_model_bundle.source
            )
        else:
            self._weight_src = model_description.weights.tensorflow_saved_model_bundle

        self._graph = None
        self._io_names: tuple[list[str], list[str]] | None = None
        super().__init__(model_description=model_description, devices=devices)

    def _parse_devices(self, devices: Sequence[str] | None) -> tuple[None]:
        if devices is not None:
            logger.warning(
                f"Device management is not implemented for tensorflow yet, ignoring the devices {devices}"
            )
        return (None,)

    def _init_model_on_device(self, device: str | None) -> Any:

        # TODO: check how to load tf weights without unzipping
        weight_file = ensure_unzipped(
            self._weight_src, Path("bioimageio_unzipped_tf_weights")
        )

        # TODO read from spec
        tag = (  # pyright: ignore[reportUnknownVariableType]
            tf.saved_model.tag_constants.SERVING
        )
        signature_key = (  # pyright: ignore[reportUnknownVariableType]
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        )

        self._graph = tf.Graph()
        with self._graph.as_default():
            sess = tf.Session(graph=self._graph)  # pyright: ignore[reportUnknownVariableType]
            # load the model and the signature
            graph_def = tf.saved_model.loader.load(  # pyright: ignore[reportUnknownVariableType]
                sess, [tag], str(weight_file)
            )
            signature = (  # pyright: ignore[reportUnknownVariableType]
                graph_def.signature_def
            )

            # get the tensors into the graph
            in_names = [  # pyright: ignore[reportUnknownVariableType]
                signature[signature_key].inputs[key].name for key in self._input_ids
            ]
            out_names = [  # pyright: ignore[reportUnknownVariableType]
                signature[signature_key].outputs[key].name for key in self._output_ids
            ]
            self._io_names = (in_names, out_names)

        return sess  # pyright: ignore[reportUnknownVariableType]

    def _forward_impl(
        self, device: None, model: Any, input_arrays: Sequence[NDArray[Any] | None]
    ):
        assert self._io_names is not None
        assert self._graph is not None

        in_names, out_names = self._io_names
        in_tf_tensors = [self._graph.get_tensor_by_name(name) for name in in_names]
        out_tf_tensors = [self._graph.get_tensor_by_name(name) for name in out_names]

        # run prediction
        res = model.run(
            dict(zip(out_names, out_tf_tensors)),
            dict(zip(in_tf_tensors, input_arrays)),
        )
        # from dict to list of tensors
        res = [res[out] for out in out_names]

        return res

    def _cleanup_pre_model_deletion(self, device: str | None, model: Any) -> None:
        return

    def _cleanup_post_model_deletion(self, device: str | None) -> None:
        return


class KerasModelAdapter(LocalModelAdapter[None, Any]):
    def __init__(
        self,
        model_description: v0_4.ModelDescr | v0_5.ModelDescr,
        devices: Sequence[str] | None = None,
    ):
        if model_description.weights.tensorflow_saved_model_bundle is None:
            raise ValueError("No `tensorflow_saved_model_bundle` weights found")

        if isinstance(model_description, v0_4.ModelDescr):
            self._weight_src = (
                model_description.weights.tensorflow_saved_model_bundle.source
            )
        else:
            self._weight_src = model_description.weights.tensorflow_saved_model_bundle

        super().__init__(model_description=model_description, devices=devices)

    def _parse_devices(self, devices: Sequence[str] | None) -> tuple[None]:
        if devices is not None:
            logger.warning(
                f"Device management is not implemented for tensorflow yet, ignoring the devices {devices}"
            )
        return (None,)

    def _init_model_on_device(self, device: None) -> Any:
        # TODO: check how to load tf weights without unzipping
        weight_file = str(
            ensure_unzipped(self._weight_src, Path("bioimageio_unzipped_tf_weights"))
        )

        try:
            tfsm_layer = tf.keras.layers.TFSMLayer(  # pyright: ignore[reportUnknownVariableType]
                weight_file,
                call_endpoint="serve",
            )
        except Exception as e:
            try:
                tfsm_layer = tf.keras.layers.TFSMLayer(  # pyright: ignore[reportUnknownVariableType]
                    weight_file, call_endpoint="serving_default"
                )
            except Exception as ee:
                logger.opt(exception=ee).info(
                    "keras.layers.TFSMLayer error for alternative call_endpoint='serving_default'"
                )
                raise e

        return tfsm_layer  # pyright: ignore[reportUnknownVariableType]

    def _forward_impl(  # pyright: ignore[reportUnknownParameterType]
        self, device: None, model: Any, input_arrays: Sequence[NDArray[Any] | None]
    ):
        assert tf is not None
        tf_tensor = [
            None if ipt is None else tf.convert_to_tensor(ipt) for ipt in input_arrays
        ]
        result = model(*tf_tensor)
        assert isinstance(result, dict)

        # TODO: Use RDF's `outputs[i].id` here
        result = list(  # pyright: ignore[reportUnknownVariableType]
            result.values()  # pyright: ignore[reportUnknownArgumentType]
        )

        return [  # pyright: ignore[reportUnknownVariableType]
            (None if r is None else r if isinstance(r, np.ndarray) else r.numpy())
            for r in result  # pyright: ignore[reportUnknownVariableType]
        ]

    def _cleanup_pre_model_deletion(self, device: str | None, model: Any) -> None:
        return

    def _cleanup_post_model_deletion(self, device: str | None) -> None:
        return


def create_tf_model_adapter(
    model_description: AnyModelDescr, devices: Sequence[str] | None = None
):
    tf_version = v0_5.Version(tf.__version__)  # type: ignore[reportUnknownVariableType]
    weights = model_description.weights.tensorflow_saved_model_bundle
    if weights is None:
        raise ValueError("No `tensorflow_saved_model_bundle` weights found")

    model_tf_version = weights.tensorflow_version
    if model_tf_version is None:
        logger.warning(
            "The model does not specify the tensorflow version."
            + f"Cannot check if it is compatible with intalled tensorflow {tf_version}."
        )
    elif model_tf_version > tf_version:
        logger.warning(
            f"The model specifies a newer tensorflow version than installed: {model_tf_version} > {tf_version}."
        )
    elif (model_tf_version.major, model_tf_version.minor) != (
        tf_version.major,
        tf_version.minor,
    ):
        logger.warning(
            "The tensorflow version specified by the model does not match the installed: "
            + f"{model_tf_version} != {tf_version}."
        )

    if tf_version.major <= 1:
        return TensorflowModelAdapter(model_description, devices=devices)
    else:
        return KerasModelAdapter(model_description, devices=devices)
