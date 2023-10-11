import gc
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import xarray as xr

from bioimageio.core.utils import import_callable
from bioimageio.spec.model import AnyModel
from bioimageio.spec.model.v0_4 import PytorchStateDictWeights as PytorchStateDictWeights04

from ._model_adapter import ModelAdapter


class PytorchModelAdapter(ModelAdapter):
    def _load(self, *, devices: Optional[Sequence[str]] = None):
        self._model = self.get_nn_instance(self.bioimageio_model)

        if devices is None:
            self._devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        else:
            self._devices = [torch.device(d) for d in devices]

        if len(self._devices) > 1:
            warnings.warn("Multiple devices for single pytorch model not yet implemented")

        self._model.to(self._devices[0])

        assert isinstance(self._model, torch.nn.Module)
        weights = self.bioimageio_model.weights.get("pytorch_state_dict")
        if weights is not None and weights.source:
            state: Any = torch.load(weights.source, map_location=self._devices[0])
            _ = self._model.load_state_dict(state)

        _ = self._model.eval()
        self._internal_output_axes = [tuple(out.axes) for out in self.bioimageio_model.outputs]

    def _forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        assert self._devices is not None
        with torch.no_grad():
            tensors = [torch.from_numpy(ipt.data) for ipt in input_tensors]
            tensors = [t.to(self._devices[0]) for t in tensors]
            result: Union[Tuple[Any, ...], List[Any], Any] = self._model(*tensors)
            if not isinstance(result, (tuple, list)):
                result = [result]

            result = [r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in result]

        return [xr.DataArray(r, dims=axes) for r, axes in zip(result, self._internal_output_axes)]

    def _unload(self) -> None:
        self._devices = None
        del self._model
        _ = gc.collect()  # deallocate memory
        torch.cuda.empty_cache()  # release reserved memory

    @staticmethod
    def get_nn_instance(model: AnyModel):
        weight_spec = model.weights.pytorch_state_dict
        assert weight_spec is not None
        arch = import_callable(
            weight_spec.architecture,
            sha256=weight_spec.architecture_sha256
            if isinstance(weight_spec, PytorchStateDictWeights04)
            else weight_spec.sha256,
        )
        model_kwargs = (
            weight_spec.kwargs
            if isinstance(weight_spec, PytorchStateDictWeights04)
            else weight_spec.architecture.kwargs
        )
        return arch(**model_kwargs)
