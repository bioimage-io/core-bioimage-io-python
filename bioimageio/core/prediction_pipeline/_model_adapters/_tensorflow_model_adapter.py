import warnings
import zipfile
from typing import List, Optional

import numpy as np
import tensorflow as tf
import xarray as xr

from bioimageio.core.resource_io import nodes
from ._model_adapter import ModelAdapter


class TensorflowModelAdapterBase(ModelAdapter):
    def require_unzipped(self, weight_file):
        if zipfile.is_zipfile(weight_file):
            out_path = weight_file.with_suffix("")
            with zipfile.ZipFile(weight_file, "r") as f:
                f.extractall(out_path)
            return out_path
        return weight_file

    def _load_model(self, weight_file):
        weight_file = self.require_unzipped(weight_file)
        if self.use_keras_api:
            return tf.keras.models.load_model(weight_file)
        else:
            # NOTE in tf1 the model needs to be loaded inside of the session, so we cannot preload the model
            return str(weight_file)

    def __init__(self, *, bioimageio_model: nodes.Model, weight_format: str, devices: Optional[List[str]] = None):
        self.spec = bioimageio_model

        try:
            tf_version = self.spec.weights[weight_format].tensorflow_version.version
        except AttributeError:
            tf_version = (1, 14, 0)
        tf_major_ver = tf_version[0]
        assert tf_major_ver in (1, 2)
        self.use_keras_api = tf_major_ver > 1 or weight_format == "keras_hdf5"

        # TODO tf device management
        if devices is not None:
            warnings.warn(f"Device management is not implemented for tensorflow yet, ignoring the devices {devices}")

        weight_file = self.require_unzipped(self.spec.weights[weight_format].source)
        self._model = self._load_model(weight_file)
        self._internal_output_axes = [tuple(out.axes) for out in bioimageio_model.outputs]

    # TODO currently we relaod the model every time. it would be better to keep the graph and session
    # alive in between of forward passes (but then the sessions need to be properly opened / closed)
    def _forward_tf(self, *input_tensors):
        input_keys = [ipt.name for ipt in self.spec.inputs]
        output_keys = [out.name for out in self.spec.outputs]

        # TODO read from spec
        tag = tf.saved_model.tag_constants.SERVING
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:

                # load the model and the signature
                graph_def = tf.saved_model.loader.load(sess, [tag], self._model)
                signature = graph_def.signature_def

                # get the tensors into the graph
                in_names = [signature[signature_key].inputs[key].name for key in input_keys]
                out_names = [signature[signature_key].outputs[key].name for key in output_keys]
                in_tensors = [graph.get_tensor_by_name(name) for name in in_names]
                out_tensors = [graph.get_tensor_by_name(name) for name in out_names]

                # run prediction
                res = sess.run(dict(zip(out_names, out_tensors)), dict(zip(in_tensors, input_tensors)))

        return res

    def _forward_keras(self, input_tensors):
        tf_tensor = [tf.convert_to_tensor(ipt) for ipt in input_tensors]
        result = self._model.forward(*tf_tensor)
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


class TensorflowModelAdapter(TensorflowModelAdapterBase):
    def __init__(self, *, bioimageio_model: nodes.Model, devices=List[str]):
        weight_format = "tensorflow_saved_model_bundle"
        super().__init__(bioimageio_model=bioimageio_model, weight_format=weight_format, devices=devices)


class KerasModelAdapter(TensorflowModelAdapterBase):
    def __init__(self, *, bioimageio_model: nodes.Model, devices=List[str]):
        weight_format = "keras_hdf5"
        super().__init__(bioimageio_model=bioimageio_model, weight_format=weight_format, devices=devices)
