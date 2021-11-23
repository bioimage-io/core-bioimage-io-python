import gc
import warnings
from typing import List, Optional

import numpy as np
import torch
import xarray as xr

from ._model_adapter import ModelAdapter


class TorchscriptModelAdapter(ModelAdapter):
    def _load(self, *, devices: Optional[List[str]] = None):
        weight_path = str(self.bioimageio_model.weights["pytorch_script"].source.resolve())
        if devices is None:
            self.devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        else:
            self.devices = [torch.device(d) for d in devices]

        if len(self.devices) > 1:
            warnings.warn("Multiple devices for single torchscript model not yet implemented")

        self._model = torch.jit.load(weight_path)
        self._model.to(self.devices[0])
        self._internal_output_axes = [tuple(out.axes) for out in self.bioimageio_model.outputs]

    def _forward(self, *batch: xr.DataArray) -> List[xr.DataArray]:
        with torch.no_grad():
            torch_tensor = [torch.from_numpy(b.data).to(self.devices[0]) for b in batch]
            result = self._model.forward(*torch_tensor)
            if not isinstance(result, (tuple, list)):
                result = [result]

            result = [r.cpu().numpy() if not isinstance(r, np.ndarray) else r for r in result]

        assert len(result) == len(self._internal_output_axes)
        return [xr.DataArray(r, dims=axes) for r, axes in zip(result, self._internal_output_axes)]

    def _unload(self) -> None:
        self._devices = None
        del self._model
        gc.collect()  # deallocate memory
        torch.cuda.empty_cache()  # release reserved memory
