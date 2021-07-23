import logging
from typing import Sequence

import torch
import xarray as xr

from bioimageio.core.utils import get_nn_instance
from bioimageio.spec.model import nodes
from ._model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class PytorchModelAdapter(ModelAdapter):
    def __init__(self, *, bioimageio_model: nodes.Model, devices=Sequence[str]):
        self._internal_output_axes = bioimageio_model.outputs[0].axes
        self.model = get_nn_instance(bioimageio_model)
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
