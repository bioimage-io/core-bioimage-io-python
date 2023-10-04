import logging
import warnings
from typing import List, Optional, Sequence

import onnxruntime as rt
import xarray as xr

from ._model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class ONNXModelAdapter(ModelAdapter):
    def _load(self, *, devices: Optional[Sequence[str]] = None):
        self._internal_output_axes = [tuple(out.axes) for out in self.bioimageio_model.outputs]
        assert self.bioimageio_model.weights.onnx is not None
        self._session = rt.InferenceSession(str(self.bioimageio_model.weights.onnx.source))
        onnx_inputs = self._session.get_inputs()  # type: ignore
        self._input_names: List[str] = [ipt.name for ipt in onnx_inputs]  # type: ignore

        if devices is not None:
            warnings.warn(f"Device management is not implemented for onnx yet, ignoring the devices {devices}")

    def _forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        assert len(input_tensors) == len(self._input_names)
        input_arrays = [ipt.data for ipt in input_tensors]
        result = self._session.run(None, dict(zip(self._input_names, input_arrays)))  # type: ignore
        if not isinstance(result, (list, tuple)):
            result = []

        return [xr.DataArray(r, dims=axes) for r, axes in zip(result, self._internal_output_axes)]  # type: ignore

    def _unload(self) -> None:
        warnings.warn("Device management is not implemented for onnx yet, cannot unload model")
