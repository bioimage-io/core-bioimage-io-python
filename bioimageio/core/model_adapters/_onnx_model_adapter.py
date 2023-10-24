import logging
import warnings
from typing import Any, List, Optional, Sequence, Union

import onnxruntime as rt
import xarray as xr
from numpy.typing import NDArray

from bioimageio.spec.model import v0_4, v0_5

from ._model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class ONNXModelAdapter(ModelAdapter):
    def __init__(self, *, model_description: Union[v0_4.Model, v0_5.Model], devices: Optional[Sequence[str]] = None):
        super().__init__()
        self._internal_output_axes = [
            tuple(out.axes) if isinstance(out.axes, str) else tuple(a.id for a in out.axes)
            for out in model_description.outputs
        ]
        if model_description.weights.onnx is None:
            raise ValueError("No ONNX weights specified for {model_description.name}")

        self._session = rt.InferenceSession(str(model_description.weights.onnx.source))
        onnx_inputs = self._session.get_inputs()  # type: ignore
        self._input_names: List[str] = [ipt.name for ipt in onnx_inputs]  # type: ignore

        if devices is not None:
            warnings.warn(f"Device management is not implemented for onnx yet, ignoring the devices {devices}")

    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        assert len(input_tensors) == len(self._input_names)
        input_arrays = [ipt.data for ipt in input_tensors]
        result: Union[  # pyright: ignore[reportUnknownVariableType]
            Sequence[NDArray[Any]], NDArray[Any]
        ] = self._session.run(None, dict(zip(self._input_names, input_arrays)))
        if not isinstance(result, (list, tuple)):
            result = []

        return [xr.DataArray(r, dims=axes) for r, axes in zip(result, self._internal_output_axes)]

    def unload(self) -> None:
        warnings.warn("Device management is not implemented for onnx yet, cannot unload model")
