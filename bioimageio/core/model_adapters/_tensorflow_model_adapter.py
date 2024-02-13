import warnings
import zipfile
from typing import List, Literal, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
import xarray as xr

from bioimageio.spec.common import FileSource, RelativeFilePath
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.utils import download

from ._model_adapter import ModelAdapter


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
        super().__init__()
        self.model_description = model_description
        tf_version = v0_5.Version(tf.__version__)
        model_tf_version = weights.tensorflow_version
        if model_tf_version is None:
            warnings.warn(
                "The model does not specify the tensorflow version."
                f"Cannot check if it is compatible with intalled tensorflow {tf_version}."
            )
        elif model_tf_version > tf_version:
            warnings.warn(
                f"The model specifies a newer tensorflow version than installed: {model_tf_version} > {tf_version}."
            )
        elif (model_tf_version.major, model_tf_version.minor) != (tf_version.major, tf_version.minor):
            warnings.warn(
                "The tensorflow version specified by the model does not match the installed: "
                f"{model_tf_version} != {tf_version}."
            )

        self.use_keras_api = tf_version.major > 1 or self.weight_format == KerasModelAdapter.weight_format

        # TODO tf device management
        if devices is not None:
            warnings.warn(f"Device management is not implemented for tensorflow yet, ignoring the devices {devices}")

        weight_file = self.require_unzipped(weights.source)
        self._network = self._get_network(weight_file)
        self._internal_output_axes = [
            tuple(out.axes) if isinstance(out.axes, str) else tuple(a.id for a in out.axes)
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

    def _get_network(self, weight_file: FileSource):
        weight_file = self.require_unzipped(weight_file)
        if self.use_keras_api:
            return tf.keras.models.load_model(weight_file, compile=False)
        else:
            # NOTE in tf1 the model needs to be loaded inside of the session, so we cannot preload the model
            return str(weight_file)

    # TODO currently we relaod the model every time. it would be better to keep the graph and session
    # alive in between of forward passes (but then the sessions need to be properly opened / closed)
    def _forward_tf(self, *input_tensors):
        input_keys = [
            ipt.name if isinstance(ipt, v0_4.InputTensorDescr) else ipt.id for ipt in self.model_description.inputs
        ]
        output_keys = [
            out.name if isinstance(out, v0_4.OutputTensorDescr) else out.id for out in self.model_description.outputs
        ]

        # TODO read from spec
        tag = tf.saved_model.tag_constants.SERVING
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                # load the model and the signature
                graph_def = tf.saved_model.loader.load(sess, [tag], self._network)
                signature = graph_def.signature_def

                # get the tensors into the graph
                in_names = [signature[signature_key].inputs[key].name for key in input_keys]
                out_names = [signature[signature_key].outputs[key].name for key in output_keys]
                in_tensors = [graph.get_tensor_by_name(name) for name in in_names]
                out_tensors = [graph.get_tensor_by_name(name) for name in out_names]

                # run prediction
                res = sess.run(dict(zip(out_names, out_tensors)), dict(zip(in_tensors, input_tensors)))
                # from dict to list of tensors
                res = [res[out] for out in out_names]

        return res

    def _forward_keras(self, *input_tensors: xr.DataArray):
        assert self.use_keras_api
        assert not isinstance(self._network, str)
        tf_tensor = [tf.convert_to_tensor(ipt) for ipt in input_tensors]

        try:
            result = self._network.forward(*tf_tensor)
        except AttributeError:
            result = self._network.predict(*tf_tensor)

        if not isinstance(result, (tuple, list)):
            result = [result]

        return [r if isinstance(r, np.ndarray) else tf.make_ndarray(r) for r in result]

    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        data = [ipt.data for ipt in input_tensors]
        if self.use_keras_api:
            result = self._forward_keras(*data)
        else:
            result = self._forward_tf(*data)

        return [xr.DataArray(r, dims=axes) for r, axes in zip(result, self._internal_output_axes)]

    def unload(self) -> None:
        warnings.warn("Device management is not implemented for keras yet, cannot unload model")


class TensorflowModelAdapter(TensorflowModelAdapterBase):
    weight_format = "tensorflow_saved_model_bundle"

    def __init__(
        self, *, model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr], devices: Optional[Sequence[str]] = None
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
        self, *, model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr], devices: Optional[Sequence[str]] = None
    ):
        if model_description.weights.keras_hdf5 is None:
            raise ValueError("missing keras_hdf5 weights")

        super().__init__(
            model_description=model_description,
            devices=devices,
            weights=model_description.weights.keras_hdf5,
        )
