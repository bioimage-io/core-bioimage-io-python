from typing import List, Optional

import numpy as np
import torch
import xarray as xr

from bioimageio.core.resource_io import nodes
from ._model_adapter import ModelAdapter


class TorchscriptModelAdapter(ModelAdapter):
    def __init__(self, *, bioimageio_model: nodes.Model, devices: Optional[List[str]] = None):
        spec = bioimageio_model
        self.name = spec.name

        _input = spec.inputs[0]
        _output = spec.outputs[0]

        self._internal_input_axes = _input.axes
        self._internal_output_axes = _output.axes

        if devices is None:
            self.devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        else:
            self.devices = [torch.device(d) for d in devices]

        weight_path = str(spec.weights["pytorch_script"].source.resolve())
        self.model = torch.jit.load(weight_path)
        self.model.to(self.devices[0])

    def forward(self, batch: xr.DataArray) -> xr.DataArray:
        with torch.no_grad():
            torch_tensor = torch.from_numpy(batch.data)
            result = self.model.forward(torch_tensor)

            if not isinstance(result, np.ndarray):
                result = result.cpu().numpy()

        return xr.DataArray(result, dims=tuple(self._internal_output_axes))
