import gc
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import xarray as xr

from bioimageio.core.utils import import_callable
from bioimageio.spec.model import v0_4, v0_5

from ._model_adapter import ModelAdapter


class PytorchModelAdapter(ModelAdapter):
    _devices: Optional[List[torch.device]] = None

    def _load(self, *, devices: Optional[Sequence[str]] = None):
        if self.model_description.weights.pytorch_state_dict is None:
            raise ValueError("missing pytorch_state_dict weights")

        self._network = self.get_network(self.model_description.weights.pytorch_state_dict)
        self._devices = self.get_devices(devices)
        self._network = self._network.to(self._devices[0])

        weights = self.model_description.weights.pytorch_state_dict
        state: Any = torch.load(weights.source, map_location=self._devices[0])
        _ = self._network.load_state_dict(state)

        self._network = self._network.eval()

    def _forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        assert self._devices is not None
        with torch.no_grad():
            tensors = [torch.from_numpy(ipt.data) for ipt in input_tensors]
            tensors = [t.to(self._devices[0]) for t in tensors]
            result: Union[Tuple[Any, ...], List[Any], Any] = self._network(*tensors)
            if not isinstance(result, (tuple, list)):
                result = [result]

            result = [r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in result]
            if len(result) > len(self.model_description.outputs):
                raise ValueError(
                    f"Expected at most {len(self.model_description.outputs)} outpus, but got {len(result)}"
                )

        return [
            xr.DataArray(r, dims=tuple(a if isinstance(a, str) else a.id for a in out.axes))
            for r, out in zip(result, self.model_description.outputs)
        ]

    def _unload(self) -> None:
        self._devices = None
        del self._network
        _ = gc.collect()  # deallocate memory
        torch.cuda.empty_cache()  # release reserved memory

    @staticmethod
    def get_network(weight_spec: Union[v0_4.PytorchStateDictWeights, v0_5.PytorchStateDictWeights]):
        arch = import_callable(
            weight_spec.architecture,
            sha256=weight_spec.architecture_sha256
            if isinstance(weight_spec, v0_4.PytorchStateDictWeights)
            else weight_spec.sha256,
        )
        model_kwargs = (
            weight_spec.kwargs
            if isinstance(weight_spec, v0_4.PytorchStateDictWeights)
            else weight_spec.architecture.kwargs
        )
        network = arch(**model_kwargs)
        if not isinstance(network, torch.nn.Module):
            raise ValueError(f"calling {weight_spec.architecture.callable} did not return a torch.nn.Module")

        return network

    @staticmethod
    def get_devices(devices: Optional[Sequence[str]] = None):
        if not devices:
            torch_devices = [torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")]
        else:
            torch_devices = [torch.device(d) for d in devices]

        if len(torch_devices) > 1:
            warnings.warn(
                f"Multiple devices for single pytorch model not yet implemented; ignoring {torch_devices[1:]}"
            )
            torch_devices = torch_devices[:1]

        return torch_devices
