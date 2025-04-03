# pyright: reportUnknownVariableType=false
import warnings
from typing import Any, List, Optional, Sequence, Union

import onnxruntime as rt  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.utils import download

from ..model_adapters import ModelAdapter
from ..utils._type_guards import is_list, is_tuple


class ONNXModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ):
        super().__init__(model_description=model_description)

        if model_description.weights.onnx is None:
            raise ValueError("No ONNX weights specified for {model_description.name}")

        local_path = download(model_description.weights.onnx.source).path
        self._session = rt.InferenceSession(local_path.read_bytes())
        onnx_inputs = self._session.get_inputs()
        self._input_names: List[str] = [ipt.name for ipt in onnx_inputs]

        if devices is not None:
            warnings.warn(
                f"Device management is not implemented for onnx yet, ignoring the devices {devices}"
            )

    def _forward_impl(
        self, input_arrays: Sequence[Optional[NDArray[Any]]]
    ) -> List[Optional[NDArray[Any]]]:
        result: Any = self._session.run(
            None, dict(zip(self._input_names, input_arrays))
        )
        if is_list(result) or is_tuple(result):
            result_seq = list(result)
        else:
            result_seq = [result]

        return result_seq

    def unload(self) -> None:
        warnings.warn(
            "Device management is not implemented for onnx yet, cannot unload model"
        )
