# pyright: reportUnknownVariableType=false
import shutil
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union, cast

import onnxruntime as rt  # pyright: ignore[reportMissingTypeStubs]
from exceptiongroup import ExceptionGroup
from loguru import logger
from numpy.typing import NDArray

from bioimageio.spec.model import v0_4, v0_5

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

        available_providers: Any = None
        if hasattr(rt, "get_available_providers"):
            available_providers = cast(Any, rt.get_available_providers())

        if is_list(available_providers):
            if len(available_providers) == 0:
                providers = [None]
            else:
                providers = available_providers
        else:
            providers = [available_providers]

        if devices is not None:
            available_devices = [d for d in devices if d in providers]
            unavailable_devices = [d for d in devices if d not in providers]
            if available_devices:
                if unavailable_devices:
                    logger.warning(
                        "The following requested devices are not available for ONNX Runtime and will be ignored: {}.\nSelected available providers/devices are: {}\nOther available providers are: {}",
                        unavailable_devices,
                        available_devices,
                        [p for p in providers if p not in devices],
                    )

                providers = available_devices

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
                source_context = nullcontext(src)
            else:
                src_reader = onnx_descr.get_reader()
                src_data_reader = onnx_descr.external_data.get_reader()

                @contextmanager
                def source_context_func():
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
                        yield src

                source_context = source_context_func()

        else:
            # load single source file from bytes (without external data, so probably <2GB)
            logger.debug(
                "Loading ONNX model from bytes (read from {})", onnx_descr.source
            )
            source_context = nullcontext(onnx_descr.get_reader().read())

        with source_context as s:
            assert isinstance(s, bytes) or s.exists()

            # try providers in order until one works
            # TODO: check if issue with backup providers is fixed and evaluate handing over all available providers
            # currently (onnxruntime 1.23.2) if a higher priority providers fails a RUNTIME_EXCEPTION may be raised
            # stating 'model_path must not be empty' instead of trying the next provider, see # TODO: reference issue
            provider_exceptions: List[Exception] = []
            for p in providers:
                try:
                    self._session = rt.InferenceSession(
                        s,
                        providers=None if p is None else [p],
                    )
                except Exception as e:
                    provider_exceptions.append(e)
                else:
                    for bad_p, e in zip(
                        providers[: len(provider_exceptions)], provider_exceptions
                    ):
                        logger.warning(
                            "Failed to load ONNX model with provider {}: {}",
                            bad_p,
                            e,
                        )

                    break
            else:
                raise ExceptionGroup(
                    "Failed to load ONNX model with any of the available providers.",
                    provider_exceptions,
                )

        onnx_inputs = self._session.get_inputs()
        self._input_names: List[str] = [ipt.name for ipt in onnx_inputs]

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
        logger.warning("Model unloading not implemented for ONNX, cannot unload model")
