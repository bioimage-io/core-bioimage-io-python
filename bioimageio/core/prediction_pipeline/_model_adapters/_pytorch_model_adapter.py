from typing import List, Optional

import torch
import xarray as xr
from marshmallow import missing

from bioimageio.core.resource_io import nodes
from ._model_adapter import ModelAdapter


class PytorchModelAdapter(ModelAdapter):
    def __init__(self, *, bioimageio_model: nodes.Model, devices: Optional[List[str]] = None):
        self._model = self.get_nn_instance(bioimageio_model)

        if devices is None:
            self._devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        else:
            self._devices = [torch.device(d) for d in devices]
        self._model.to(self._devices[0])

        assert isinstance(self._model, torch.nn.Module)
        weights = bioimageio_model.weights.get("pytorch_state_dict")
        if weights is not None and weights.source:
            state = torch.load(weights.source, map_location=self._devices[0])
            self._model.load_state_dict(state)

        self._internal_output_axes = [tuple(out.axes) for out in bioimageio_model.outputs]

    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        with torch.no_grad():
            tensors = [torch.from_numpy(ipt.data) for ipt in input_tensors]
            tensors = [t.to(self._devices[0]) for t in tensors]
            result = self._model(*tensors)
            if not isinstance(result, (tuple, list)):
                result = [result]

            result = [r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in result]

        return [xr.DataArray(r, dims=axes) for r, axes in zip(result, self._internal_output_axes)]

    @staticmethod
    def get_nn_instance(model_node: nodes.Model, **kwargs):
        assert isinstance(model_node.source, nodes.ImportedSource)

        joined_kwargs = {} if model_node.kwargs is missing else dict(model_node.kwargs)
        joined_kwargs.update(kwargs)
        return model_node.source(**joined_kwargs)
