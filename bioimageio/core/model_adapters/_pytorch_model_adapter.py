import gc
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.utils import download

from ..axis import AxisId
from ..digest_spec import get_axes_infos, import_callable
from ..tensor import Tensor
from ._model_adapter import ModelAdapter

try:
    import torch
except Exception as e:
    torch = None
    torch_error = str(e)
else:
    torch_error = None


class PytorchModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        outputs: Union[
            Sequence[v0_4.OutputTensorDescr], Sequence[v0_5.OutputTensorDescr]
        ],
        weights: Union[
            v0_4.PytorchStateDictWeightsDescr, v0_5.PytorchStateDictWeightsDescr
        ],
        devices: Optional[Sequence[str]] = None,
    ):
        if torch is None:
            raise ImportError(f"failed to import torch: {torch_error}")

        super().__init__()
        self.output_dims = [tuple(a.id for a in get_axes_infos(out)) for out in outputs]
        self._network = self.get_network(weights)
        self._devices = self.get_devices(devices)
        self._network = self._network.to(self._devices[0])

        self._primary_device = self._devices[0]
        state: Any = torch.load(
            download(weights).path,
            map_location=self._primary_device,  # pyright: ignore[reportUnknownArgumentType]
        )
        self._network.load_state_dict(state)

        self._network = self._network.eval()

    def forward(self, *input_tensors: Optional[Tensor]) -> List[Optional[Tensor]]:
        if torch is None:
            raise ImportError("torch")

        torch.use_deterministic_algorithms(self.use_deterministic_algorithms==)
        if self.use_deterministic_algorithms:
            _ = torch.manual_seed(0)
            np.random.seed(0)

        with torch.no_grad():
            tensors = [
                None if ipt is None else torch.from_numpy(ipt.data.data)
                for ipt in input_tensors
            ]
            tensors = [
                (
                    None
                    if t is None
                    else t.to(
                        self._primary_device  # pyright: ignore[reportUnknownArgumentType]
                    )
                )
                for t in tensors
            ]
            result: Union[Tuple[Any, ...], List[Any], Any]
            result = self._network(  # pyright: ignore[reportUnknownVariableType]
                *tensors
            )
            if not isinstance(result, (tuple, list)):
                result = [result]

            result = [
                (
                    None
                    if r is None
                    else r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r
                )
                for r in result  # pyright: ignore[reportUnknownVariableType]
            ]
            if len(result) > len(self.output_dims):
                raise ValueError(
                    f"Expected at most {len(self.output_dims)} outputs, but got {len(result)}"
                )

        return [
            None if r is None else Tensor(r, dims=out)
            for r, out in zip(result, self.output_dims)
        ]

    def unload(self) -> None:
        del self._network
        _ = gc.collect()  # deallocate memory
        assert torch is not None
        torch.cuda.empty_cache()  # release reserved memory

    @staticmethod
    def get_network(  # pyright: ignore[reportUnknownParameterType]
        weight_spec: Union[
            v0_4.PytorchStateDictWeightsDescr, v0_5.PytorchStateDictWeightsDescr
        ],
    ) -> "torch.nn.Module":  # pyright: ignore[reportInvalidTypeForm]
        if torch is None:
            raise ImportError("torch")
        arch = import_callable(
            weight_spec.architecture,
            sha256=(
                weight_spec.architecture_sha256
                if isinstance(weight_spec, v0_4.PytorchStateDictWeightsDescr)
                else weight_spec.sha256
            ),
        )
        model_kwargs = (
            weight_spec.kwargs
            if isinstance(weight_spec, v0_4.PytorchStateDictWeightsDescr)
            else weight_spec.architecture.kwargs
        )
        network = arch(**model_kwargs)
        if not isinstance(network, torch.nn.Module):
            raise ValueError(
                f"calling {weight_spec.architecture.callable} did not return a torch.nn.Module"
            )

        return network

    @staticmethod
    def get_devices(  # pyright: ignore[reportUnknownParameterType]
        devices: Optional[Sequence[str]] = None,
    ) -> List["torch.device"]:  # pyright: ignore[reportInvalidTypeForm]
        if torch is None:
            raise ImportError("torch")
        if not devices:
            torch_devices = [
                (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            ]
        else:
            torch_devices = [torch.device(d) for d in devices]

        if len(torch_devices) > 1:
            warnings.warn(
                f"Multiple devices for single pytorch model not yet implemented; ignoring {torch_devices[1:]}"
            )
            torch_devices = torch_devices[:1]

        return torch_devices
