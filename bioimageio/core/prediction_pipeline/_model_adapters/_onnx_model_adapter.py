import logging
import warnings
from typing import List, Optional

import onnxruntime as rt
import xarray as xr

from bioimageio.core.resource_io import nodes
from ._model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class ONNXModelAdapter(ModelAdapter):
    def __init__(self, *, bioimageio_model: nodes.Model, devices: Optional[List[str]] = None):
        spec = bioimageio_model
        self.name = spec.name

        if len(spec.inputs) != 1 or len(spec.outputs) != 1:
            raise NotImplementedError("Only single input, single output models are supported")

        assert len(spec.inputs) == 1
        assert len(spec.outputs) == 1

        spec.inputs[0]
        _output = spec.outputs[0]

        self._internal_output_axes = _output.axes

        self._session = rt.InferenceSession(str(spec.weights["onnx"].source))
        onnx_inputs = self._session.get_inputs()
        assert len(onnx_inputs) == 1, f"expected onnx model to have one input got {len(onnx_inputs)}"
        self._input_name = onnx_inputs[0].name
        # TODO onnx device management
        self.devices = []
        if devices is not None:
            warnings.warn(f"Device management is not implemented for onnx yet, ignoring the devices {devices}")

    def forward(self, input: xr.DataArray) -> xr.DataArray:
        result = self._session.run(None, {self._input_name: input.data})[0]
        return xr.DataArray(result, dims=tuple(self._internal_output_axes))
