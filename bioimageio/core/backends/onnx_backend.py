import warnings
from typing import Any, List, Optional, Sequence, Union

import onnxruntime as rt

from bioimageio.spec._internal.type_guards import is_list, is_tuple
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.utils import download

from ..digest_spec import get_axes_infos
from ..model_adapters import ModelAdapter
from ..tensor import Tensor


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

        self._session = rt.InferenceSession(
            str(download(model_description.weights.onnx.source).path)
        )
        onnx_inputs = self._session.get_inputs()  # type: ignore
        self._input_names: List[str] = [ipt.name for ipt in onnx_inputs]  # type: ignore

        if devices is not None:
            warnings.warn(
                f"Device management is not implemented for onnx yet, ignoring the devices {devices}"
            )

    def forward(self, *input_tensors: Optional[Tensor]) -> List[Optional[Tensor]]:
        assert len(input_tensors) == len(self._input_names)
        input_arrays = [None if ipt is None else ipt.data.data for ipt in input_tensors]
        result: Any = self._session.run(
            None, dict(zip(self._input_names, input_arrays))
        )
        if is_list(result) or is_tuple(result):
            result_seq = result
        else:
            result_seq = [result]

        return [
            None if r is None else Tensor(r, dims=axes)
            for r, axes in zip(result_seq, self._internal_output_axes)
        ]

    def unload(self) -> None:
        warnings.warn(
            "Device management is not implemented for onnx yet, cannot unload model"
        )
