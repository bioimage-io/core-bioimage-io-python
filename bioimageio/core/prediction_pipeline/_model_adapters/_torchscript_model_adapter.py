from typing import List, Optional

import numpy as np
import torch
import xarray as xr

from bioimageio.core.resource_io import nodes
from ._model_adapter import ModelAdapter


class TorchscriptModelAdapter(ModelAdapter):
    def __init__(self, *, bioimageio_model: nodes.Model, devices: Optional[List[str]] = None):
        weight_path = str(bioimageio_model.weights["pytorch_script"].source.resolve())
        if devices is None:
            devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        else:
            devices = [torch.device(d) for d in devices]

        self._model = torch.jit.load(weight_path)
        self._model.to(devices[0])
        self._internal_output_axes = [tuple(out.axes) for out in bioimageio_model.outputs]

    def forward(self, *batch: xr.DataArray) -> List[xr.DataArray]:
        with torch.no_grad():
            torch_tensor = [torch.from_numpy(b.data) for b in batch]
            result = self._model.forward(*torch_tensor)
            if not isinstance(result, (tuple, list)):
                result = [result]

            result = [r.cpu().numpy() if not isinstance(r, np.ndarray) else r for r in result]

        assert len(result) == len(self._internal_output_axes)
        return [xr.DataArray(r, dims=axes) for r, axes in zip(result, self._internal_output_axes)]
