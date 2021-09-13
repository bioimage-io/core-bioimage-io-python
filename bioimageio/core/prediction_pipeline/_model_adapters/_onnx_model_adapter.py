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

        self._internal_output_axes = [tuple(out.axes) for out in bioimageio_model.outputs]

        self._session = rt.InferenceSession(str(spec.weights["onnx"].source))
        onnx_inputs = self._session.get_inputs()
        self._input_names = [ipt.name for ipt in onnx_inputs]

        if devices is not None:
            warnings.warn(f"Device management is not implemented for onnx yet, ignoring the devices {devices}")

    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        assert len(input_tensors) == len(self._input_names)
        input_arrays = [ipt.data for ipt in input_tensors]
        result = self._session.run(None, dict(zip(self._input_names, input_arrays)))
        if not isinstance(result, (list, tuple)):
            result = []

        return [xr.DataArray(r, dims=axes) for r, axes in zip(result, self._internal_output_axes)]
