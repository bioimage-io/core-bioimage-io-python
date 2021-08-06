import logging
from typing import List, Optional

import torch
import xarray as xr
from marshmallow import missing

from bioimageio.spec.model import nodes
from ._model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class PytorchModelAdapter(ModelAdapter):
    def __init__(self, *, bioimageio_model: nodes.Model, devices: Optional[List[str]] = None):
        self._internal_output_axes = bioimageio_model.outputs[0].axes
        self.model = self.get_nn_instance(bioimageio_model)

        if devices is None:
            self.devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        else:
            self.devices = [torch.device(d) for d in devices]
        self.model.to(self.devices[0])

        assert isinstance(self.model, torch.nn.Module)
        weights = bioimageio_model.weights.get("pytorch_state_dict")
        if weights is not None and weights.source:
            state = torch.load(weights.source, map_location=self.devices[0])
            self.model.load_state_dict(state)

    def forward(self, input_tensor: xr.DataArray) -> xr.DataArray:
        with torch.no_grad():
            tensor = torch.from_numpy(input_tensor.data)
            tensor = tensor.to(self.devices[0])
            result = self.model(*[tensor])
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()

        return xr.DataArray(result, dims=tuple(self._internal_output_axes))

    @staticmethod
    def get_nn_instance(model_node: nodes.Model, **kwargs):
        assert isinstance(model_node.source, nodes.ImportedSource)

        joined_kwargs = {} if model_node.kwargs is missing else dict(model_node.kwargs)
        joined_kwargs.update(kwargs)
        return model_node.source(**joined_kwargs)
