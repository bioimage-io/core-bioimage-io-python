# pyright: reportUnknownVariableType=false
from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional, cast

import onnxruntime as rt  # pyright: ignore[reportMissingTypeStubs]
from loguru import logger
from numpy.typing import NDArray

from bioimageio.spec.model import v0_4, v0_5

from .._model_adapter import LocalModelAdapter
from ..utils._type_guards import is_list, is_tuple


class ONNXModelAdapter(LocalModelAdapter[Optional[str], rt.InferenceSession]):
    def __init__(
        self,
        model_description: v0_4.ModelDescr | v0_5.ModelDescr,
        devices: Sequence[str] | None = None,
    ):
        onnx_descr = model_description.weights.onnx
        if onnx_descr is None:
            raise ValueError("No ONNX weights specified for {model_description.name}")

        self._onnx_descr = onnx_descr
        self._input_names: list[str] | None = None
        super().__init__(model_description=model_description, devices=devices)

    def _parse_devices(
        self, devices: Sequence[str] | None
    ) -> Sequence[str | None]:
        available_providers: Any = None
        if hasattr(rt, "get_available_providers"):
            available_providers = cast(Any, rt.get_available_providers())

        if is_list(available_providers):
            if len(available_providers) == 0:
                providers = [None]
            else:
                providers = available_providers
        else:
            available_providers = [available_providers]
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
            elif not available_providers:
                logger.error(
                    "ONNX Runtime does not report any available providers. Attempting to load model with default providers, but this will likely fail."
                )
            else:
                logger.warning(
                    "None of the requested devices are available for ONNX Runtime, falling back to default, available providers: {}",
                    available_providers,
                )
        return providers

    def _init_model_on_device(self, device: str | None) -> rt.InferenceSession:
        onnx_descr = self._onnx_descr
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
            session = rt.InferenceSession(
                s,
                providers=None if device is None else [device],
            )

        onnx_inputs = session.get_inputs()
        onnx_input_names = [str(ipt.name) for ipt in onnx_inputs]  # pyright: ignore[reportUnknownArgumentType]
        if self._input_names is None:
            self._input_names = onnx_input_names
        elif self._input_names != onnx_input_names:
            raise RuntimeError(
                f"Input names of the ONNX model {onnx_input_names} do not match expected input names {self._input_names} from previous model initialization."
            )

        return session

    def _forward_impl(
        self,
        device: str | None,
        model: rt.InferenceSession,
        input_arrays: Sequence[NDArray[Any] | None],
    ) -> list[NDArray[Any] | None]:
        assert self._input_names is not None, "set during model initialization"
        result: Any = model.run(None, dict(zip(self._input_names, input_arrays)))
        if is_list(result) or is_tuple(result):
            result_seq = list(result)
        else:
            result_seq = [result]

        return result_seq

    def _cleanup_pre_model_deletion(
        self, device: str | None, model: rt.InferenceSession
    ) -> None:
        return

    def _cleanup_post_model_deletion(self, device: str | None) -> None:
        return
