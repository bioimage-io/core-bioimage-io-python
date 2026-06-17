# pyright: reportUnknownVariableType=false
import gc
from typing import Any, List, Optional, Sequence, Union

import torch
from loguru import logger
from numpy.typing import NDArray

from bioimageio.spec.model import v0_4, v0_5

from .._model_adapter import LocalModelAdapter
from ..utils._type_guards import is_list, is_tuple
from .pytorch_backend import get_devices


class TorchscriptModelAdapter(LocalModelAdapter[torch.device, Any]):
    def __init__(
        self,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ):
        if model_description.weights.torchscript is None:
            raise ValueError(
                f"No torchscript weights found for model {model_description.name}"
            )

        self._weight_reader = model_description.weights.torchscript.get_reader()
        super().__init__(model_description=model_description, devices=devices)

    def _parse_devices(
        self, devices: Optional[Sequence[str]]
    ) -> Sequence[torch.device]:
        return get_devices(devices)

    def _init_model_on_device(self, device: torch.device) -> Any:
        model = torch.jit.load(self._weight_reader)
        model.to(device)
        try:
            model.eval()
        except Exception as e:
            logger.warning(
                f"Failed to set model to evaluation mode for torchscript model on {device}: {e}"
            )
        return model

    def _forward_impl(
        self,
        device: torch.device,
        model: Any,
        input_arrays: Sequence[Optional[NDArray[Any]]],
    ) -> List[Optional[NDArray[Any]]]:
        with torch.no_grad():
            torch_tensor = [
                None if a is None else torch.from_numpy(a).to(device)
                for a in input_arrays
            ]
            output: Any = model.forward(*torch_tensor)
            if is_list(output) or is_tuple(output):
                output_seq: Sequence[Any] = output
            else:
                output_seq = [output]

            return [
                (
                    None
                    if r is None
                    else r.cpu().numpy()
                    if isinstance(r, torch.Tensor)
                    else r
                )
                for r in output_seq
            ]

    def _cleanup_pre_model_deletion(self, device: torch.device, model: Any) -> None:
        return

    def _cleanup_post_model_deletion(self, device: torch.device) -> None:
        _ = gc.collect()  # deallocate memory
        if device.type == "cuda":
            torch.cuda.empty_cache()  # release reserved memory
