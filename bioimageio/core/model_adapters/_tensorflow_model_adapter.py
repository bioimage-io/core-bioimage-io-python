import warnings
import zipfile
from typing import List, Literal, Optional, Sequence, Union

import numpy as np

from bioimageio.core.tensor import Tensor
from bioimageio.spec.common import FileSource
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.utils import download

from ._model_adapter import ModelAdapter

try:
    import tensorflow as tf  # pyright: ignore[reportMissingImports]
except Exception:
    tf = None


class TensorflowModelAdapterBase(ModelAdapter):
    weight_format: Literal["keras_hdf5", "tensorflow_saved_model_bundle"]

    def __init__(
        self,
        *,
        devices: Optional[Sequence[str]] = None,
        weights: Union[
            v0_4.KerasHdf5WeightsDescr,
            v0_4.TensorflowSavedModelBundleWeightsDescr,
            v0_5.KerasHdf5WeightsDescr,
            v0_5.TensorflowSavedModelBundleWeightsDescr,
        ],
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
    ):
        if tf is None:
            raise ImportError("tensorflow")

        super().__init__()
        self.model_description = model_description
        tf_version = v0_5.Version(
            tf.__version__  # pyright: ignore[reportUnknownArgumentType]
        )
        model_tf_version = weights.tensorflow_version
        if model_tf_version is None:
            warnings.warn(
                "The model does not specify the tensorflow version."
                + f"Cannot check if it is compatible with intalled tensorflow {tf_version}."
            )
        elif model_tf_version > tf_version:
            warnings.warn(
                f"The model specifies a newer tensorflow version than installed: {model_tf_version} > {tf_version}."
            )
        elif (model_tf_version.major, model_tf_version.minor) != (
            tf_version.major,
            tf_version.minor,
        ):
            warnings.warn(
                "The tensorflow version specified by the model does not match the installed: "
                + f"{model_tf_version} != {tf_version}."
            )

        self.use_keras_api = (
            tf_version.major > 1
            or self.weight_format == KerasModelAdapter.weight_format
        )

        # TODO tf device management
        if devices is not None:
            warnings.warn(
                f"Device management is not implemented for tensorflow yet, ignoring the devices {devices}"
            )

        weight_file = self.require_unzipped(weights.source)
        self._network = self._get_network(weight_file)
        self._internal_output_axes = [
            (
                tuple(out.axes)
                if isinstance(out.axes, str)
                else tuple(a.id for a in out.axes)
            )
            for out in model_description.outputs
        ]

    def require_unzipped(self, weight_file: FileSource):
        loacl_weights_file = download(weight_file).path
        if zipfile.is_zipfile(loacl_weights_file):
            out_path = loacl_weights_file.with_suffix(".unzipped")
            with zipfile.ZipFile(loacl_weights_file, "r") as f:
                f.extractall(out_path)

            return out_path
        else:
            return loacl_weights_file

    def _get_network(  # pyright: ignore[reportUnknownParameterType]
        self, weight_file: FileSource
    ):
        weight_file = self.require_unzipped(weight_file)
        assert tf is not None
        if self.use_keras_api:
            return tf.keras.models.load_model(
                weight_file, compile=False
            )  # pyright: ignore[reportUnknownVariableType]
        else:
            # NOTE in tf1 the model needs to be loaded inside of the session, so we cannot preload the model
            return str(weight_file)

    # TODO currently we relaod the model every time. it would be better to keep the graph and session
    # alive in between of forward passes (but then the sessions need to be properly opened / closed)
    def _forward_tf(  # pyright: ignore[reportUnknownParameterType]
        self, *input_tensors: Optional[Tensor]
    ):
        assert tf is not None
        input_keys = [
            ipt.name if isinstance(ipt, v0_4.InputTensorDescr) else ipt.id
            for ipt in self.model_description.inputs
        ]
        output_keys = [
            out.name if isinstance(out, v0_4.OutputTensorDescr) else out.id
            for out in self.model_description.outputs
        ]
        # TODO read from spec
        tag = (  # pyright: ignore[reportUnknownVariableType]
            tf.saved_model.tag_constants.SERVING
        )
        signature_key = (  # pyright: ignore[reportUnknownVariableType]
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        )

        graph = tf.Graph()  # pyright: ignore[reportUnknownVariableType]
        with graph.as_default():
            with tf.Session(
                graph=graph
            ) as sess:  # pyright: ignore[reportUnknownVariableType]
                # load the model and the signature
                graph_def = tf.saved_model.loader.load(  # pyright: ignore[reportUnknownVariableType]
                    sess, [tag], self._network
                )
                signature = (  # pyright: ignore[reportUnknownVariableType]
                    graph_def.signature_def
                )

                # get the tensors into the graph
                in_names = [  # pyright: ignore[reportUnknownVariableType]
                    signature[signature_key].inputs[key].name for key in input_keys
                ]
                out_names = [  # pyright: ignore[reportUnknownVariableType]
                    signature[signature_key].outputs[key].name for key in output_keys
                ]
                in_tensors = [  # pyright: ignore[reportUnknownVariableType]
                    graph.get_tensor_by_name(name)
                    for name in in_names  # pyright: ignore[reportUnknownVariableType]
                ]
                out_tensors = [  # pyright: ignore[reportUnknownVariableType]
                    graph.get_tensor_by_name(name)
                    for name in out_names  # pyright: ignore[reportUnknownVariableType]
                ]

                # run prediction
                res = sess.run(  # pyright: ignore[reportUnknownVariableType]
                    dict(
                        zip(
                            out_names,  # pyright: ignore[reportUnknownArgumentType]
                            out_tensors,  # pyright: ignore[reportUnknownArgumentType]
                        )
                    ),
                    dict(
                        zip(
                            in_tensors,  # pyright: ignore[reportUnknownArgumentType]
                            input_tensors,
                        )
                    ),
                )
                # from dict to list of tensors
                res = [  # pyright: ignore[reportUnknownVariableType]
                    res[out]
                    for out in out_names  # pyright: ignore[reportUnknownVariableType]
                ]

        return res  # pyright: ignore[reportUnknownVariableType]

    def _forward_keras(  # pyright: ignore[reportUnknownParameterType]
        self, *input_tensors: Optional[Tensor]
    ):
        assert self.use_keras_api
        assert not isinstance(self._network, str)
        assert tf is not None
        tf_tensor = [  # pyright: ignore[reportUnknownVariableType]
            None if ipt is None else tf.convert_to_tensor(ipt) for ipt in input_tensors
        ]

        try:
            result = (  # pyright: ignore[reportUnknownVariableType]
                self._network.forward(*tf_tensor)
            )
        except AttributeError:
            result = (  # pyright: ignore[reportUnknownVariableType]
                self._network.predict(*tf_tensor)
            )

        if not isinstance(result, (tuple, list)):
            result = [result]  # pyright: ignore[reportUnknownVariableType]

        return [  # pyright: ignore[reportUnknownVariableType]
            (
                None
                if r is None
                else r if isinstance(r, np.ndarray) else tf.make_ndarray(r)
            )
            for r in result  # pyright: ignore[reportUnknownVariableType]
        ]

    def forward(self, *input_tensors: Optional[Tensor]) -> List[Optional[Tensor]]:
        data = [None if ipt is None else ipt.data for ipt in input_tensors]
        if self.use_keras_api:
            result = self._forward_keras(  # pyright: ignore[reportUnknownVariableType]
                *data
            )
        else:
            result = self._forward_tf(  # pyright: ignore[reportUnknownVariableType]
                *data
            )

        return [
            None if r is None else Tensor(r, dims=axes)
            for r, axes in zip(  # pyright: ignore[reportUnknownVariableType]
                result,  # pyright: ignore[reportUnknownArgumentType]
                self._internal_output_axes,
            )
        ]

    def unload(self) -> None:
        warnings.warn(
            "Device management is not implemented for keras yet, cannot unload model"
        )


class TensorflowModelAdapter(TensorflowModelAdapterBase):
    weight_format = "tensorflow_saved_model_bundle"

    def __init__(
        self,
        *,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ):
        if model_description.weights.tensorflow_saved_model_bundle is None:
            raise ValueError("missing tensorflow_saved_model_bundle weights")

        super().__init__(
            devices=devices,
            weights=model_description.weights.tensorflow_saved_model_bundle,
            model_description=model_description,
        )


class KerasModelAdapter(TensorflowModelAdapterBase):
    weight_format = "keras_hdf5"

    def __init__(
        self,
        *,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ):
        if model_description.weights.keras_hdf5 is None:
            raise ValueError("missing keras_hdf5 weights")

        super().__init__(
            model_description=model_description,
            devices=devices,
            weights=model_description.weights.keras_hdf5,
        )
