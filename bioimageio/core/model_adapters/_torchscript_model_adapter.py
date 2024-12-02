import gc
import warnings
from typing import Any, List, Optional, Sequence, Union

import torch

from bioimageio.spec._internal.type_guards import is_list, is_ndarray, is_tuple
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.utils import download

from ..digest_spec import get_axes_infos
from ..tensor import Tensor
from ._model_adapter import ModelAdapter


class TorchscriptModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        if model_description.weights.torchscript is None:
            raise ValueError(
                f"No torchscript weights found for model {model_description.name}"
            )

        weight_path = download(model_description.weights.torchscript.source).path
        if devices is None:
            self.devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        else:
            self.devices = [torch.device(d) for d in devices]

        if len(self.devices) > 1:
            warnings.warn(
                "Multiple devices for single torchscript model not yet implemented"
            )

        self._model = torch.jit.load(weight_path)
        self._model.to(self.devices[0])
        self._model = self._model.eval()
        self._internal_output_axes = [
            tuple(a.id for a in get_axes_infos(out))
            for out in model_description.outputs
        ]

    def forward(self, *batch: Optional[Tensor]) -> List[Optional[Tensor]]:
        with torch.no_grad():
            torch_tensor = [
                None if b is None else torch.from_numpy(b.data.data).to(self.devices[0])
                for b in batch
            ]
            _result: Any = self._model.forward(*torch_tensor)
            if is_list(_result) or is_tuple(_result):
                result: Sequence[Any] = _result
            else:
                result = [_result]

            result = [
                (
                    None
                    if r is None
                    else r.cpu().numpy() if isinstance(r, torch.Tensor) else r
                )
                for r in result
            ]

        assert len(result) == len(self._internal_output_axes)
        return [
            None if r is None else Tensor(r, dims=axes) if is_ndarray(r) else r
            for r, axes in zip(result, self._internal_output_axes)
        ]

    def unload(self) -> None:
        self._devices = None
        del self._model
        _ = gc.collect()  # deallocate memory
        torch.cuda.empty_cache()  # release reserved memory
