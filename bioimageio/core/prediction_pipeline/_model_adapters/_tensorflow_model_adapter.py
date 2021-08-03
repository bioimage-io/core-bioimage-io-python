from typing import List

import numpy as np
import tensorflow as tf
import xarray as xr

from bioimageio.core.utils import get_nn_instance
from bioimageio.spec.model import nodes
from ._model_adapter import ModelAdapter


class TensorflowModelAdapterBase(ModelAdapter):
    def __init__(self, *, bioimageio_model: nodes.Model, weight_format: str, devices=List[str]):
        spec = bioimageio_model
        self.name = spec.name

        spec.inputs[0]
        _output = spec.outputs[0]
        # FIXME: TF probably uses different axis names
        self._internal_output_axes = _output.axes

        # FIXME why do we call get_nn_instance here?
        self.model = get_nn_instance(bioimageio_model)
        self.devices = []
        tf_model = tf.keras.models.load_model(spec.weights[weight_format].source)
        self.model.set_model(tf_model)

    def forward(self, input_tensor: xr.DataArray) -> xr.DataArray:
        tf_tensor = tf.convert_to_tensor(input_tensor.data)

        res = self.model.forward(tf_tensor)

        if not isinstance(res, np.ndarray):
            res = tf.make_ndarray(res)

        return xr.DataArray(res, dims=tuple(self._internal_output_axes))


class TensorflowModelAdapter(TensorflowModelAdapterBase):
    def __init__(self, *, bioimageio_model: nodes.Model, weight_format: str, devices=List[str]):
        weight_format = "tensorflow_saved_model_bundle"
        super().__init__(bioimageio_model=bioimageio_model, weight_format=weight_format, devices=devices)


class KerasModelAdapter(TensorflowModelAdapterBase):
    def __init__(self, *, bioimageio_model: nodes.Model, weight_format: str, devices=List[str]):
        weight_format = "keras_hdf5"
        super().__init__(bioimageio_model=bioimageio_model, weight_format=weight_format, devices=devices)
