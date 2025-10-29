# pyright: reportUnknownVariableType=false
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import onnxruntime as rt  # pyright: ignore[reportMissingTypeStubs]
from bioimageio.spec.model import v0_4, v0_5
from loguru import logger
from numpy.typing import NDArray

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

        onnx_descr = model_description.weights.onnx
        if onnx_descr is None:
            raise ValueError("No ONNX weights specified for {model_description.name}")

        providers = None
        if hasattr(rt, "get_available_providers"):
            providers = rt.get_available_providers()

        if (
            isinstance(onnx_descr, v0_5.OnnxWeightsDescr)
            and onnx_descr.external_data is not None
        ):
            src = onnx_descr.source.absolute()
            src_data = onnx_descr.external_data.source.absolute()
            if (
                isinstance(src, Path)
                and isinstance(src_data, Path)
                and src.parent == src_data.parent
            ):
                logger.debug(
                    "Loading ONNX model with external data from {}",
                    src.parent,
                )
                self._session = rt.InferenceSession(
                    src,
                    providers=providers,  # pyright: ignore[reportUnknownArgumentType]
                )
            else:
                src_reader = onnx_descr.get_reader()
                src_data_reader = onnx_descr.external_data.get_reader()
                with tempfile.TemporaryDirectory() as tmpdir:
                    logger.debug(
                        "Loading ONNX model with external data from {}",
                        tmpdir,
                    )
                    src = Path(tmpdir) / src_reader.original_file_name
                    src_data = Path(tmpdir) / src_data_reader.original_file_name
                    with src.open("wb") as f:
                        shutil.copyfileobj(src_reader, f)
                    with src_data.open("wb") as f:
                        shutil.copyfileobj(src_data_reader, f)

                    self._session = rt.InferenceSession(
                        src,
                        providers=providers,  # pyright: ignore[reportUnknownArgumentType]
                    )
        else:
            # load single source file from bytes (without external data, so probably <2GB)
            logger.debug(
                "Loading ONNX model from bytes (read from {})", onnx_descr.source
            )
            reader = onnx_descr.get_reader()
            self._session = rt.InferenceSession(
                reader.read(),
                providers=providers,  # pyright: ignore[reportUnknownArgumentType]
            )

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
