import gc
import warnings
from typing import List, Optional, Sequence

import torch
import xarray as xr
from marshmallow import missing

from bioimageio.core.resource_io import nodes
from ._model_adapter import ModelAdapter


class PytorchModelAdapter(ModelAdapter):
    def _load(self, *, devices: Optional[Sequence[str]] = None):
        self._model = self.get_nn_instance(self.bioimageio_model)

        if devices is None:
            self._devices: Optional[List[torch.device]] = [
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            ]
        else:
            self._devices = [torch.device(d) for d in devices]

        if len(self._devices) > 1:
            warnings.warn("Multiple devices for single pytorch model not yet implemented")

        self._model.to(self._devices[0])

        assert isinstance(self._model, torch.nn.Module)
        weights = self.bioimageio_model.weights.get("pytorch_state_dict")
        if weights is not None and weights.source:
            state = torch.load(weights.source, map_location=self._devices[0])
            self._model.load_state_dict(state)

        self._model.eval()
        self._internal_output_axes = [tuple(out.axes) for out in self.bioimageio_model.outputs]

    def _forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        assert self._devices is not None
        with torch.no_grad():
            tensors = [torch.from_numpy(ipt.data) for ipt in input_tensors]
            tensors = [t.to(self._devices[0]) for t in tensors]
            result = self._model(*tensors)
            if not isinstance(result, (tuple, list)):
                result = [result]

            result = [r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in result]

        return [xr.DataArray(r, dims=axes) for r, axes in zip(result, self._internal_output_axes)]

    def _unload(self) -> None:
        self._devices = None
        del self._model
        gc.collect()  # deallocate memory
        torch.cuda.empty_cache()  # release reserved memory

    @staticmethod
    def get_nn_instance(model_node: nodes.Model, **kwargs):
        weight_spec = model_node.weights.get("pytorch_state_dict")
        assert weight_spec is not None
        assert isinstance(weight_spec.architecture, nodes.ImportedSource)
        model_kwargs = weight_spec.kwargs
        joined_kwargs = {} if model_kwargs is missing else dict(model_kwargs)
        joined_kwargs.update(kwargs)
        return weight_spec.architecture(**joined_kwargs)
