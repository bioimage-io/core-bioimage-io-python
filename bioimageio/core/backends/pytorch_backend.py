import gc
import warnings
from contextlib import nullcontext
from io import TextIOWrapper
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import torch
from loguru import logger
from torch import nn
from typing_extensions import assert_never

from bioimageio.spec.common import ZipPath
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.utils import download

from ..digest_spec import get_axes_infos, import_callable
from ..tensor import Tensor
from ._model_adapter import ModelAdapter


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
        devices: Optional[Sequence[Union[str, torch.device]]] = None,
        mode: Literal["eval", "train"] = "eval",
    ):
        super().__init__()
        self.output_dims = [tuple(a.id for a in get_axes_infos(out)) for out in outputs]
        devices = get_devices(devices)
        self._model = load_torch_model(weights, load_state=True, devices=devices)
        if mode == "eval":
            self._model = self._model.eval()
        elif mode == "train":
            self._model = self._model.train()
        else:
            assert_never(mode)

        self._mode: Literal["eval", "train"] = mode
        self._primary_device = devices[0]

    def forward(self, *input_tensors: Optional[Tensor]) -> List[Optional[Tensor]]:
        if self._mode == "eval":
            ctxt = torch.no_grad
        elif self._mode == "train":
            ctxt = nullcontext
        else:
            assert_never(self._mode)

        with ctxt():
            tensors = [
                None if ipt is None else torch.from_numpy(ipt.data.data)
                for ipt in input_tensors
            ]
            tensors = [
                (None if t is None else t.to(self._primary_device)) for t in tensors
            ]
            result: Union[Tuple[Any, ...], List[Any], Any]
            result = self._model(*tensors)
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
        del self._model
        _ = gc.collect()  # deallocate memory
        assert torch is not None
        torch.cuda.empty_cache()  # release reserved memory


def load_torch_model(
    weight_spec: Union[
        v0_4.PytorchStateDictWeightsDescr, v0_5.PytorchStateDictWeightsDescr
    ],
    *,
    load_state: bool = False,
    devices: Optional[Sequence[Union[str, torch.device]]] = None,
) -> nn.Module:
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
    if not isinstance(network, nn.Module):
        raise ValueError(
            f"calling {weight_spec.architecture.callable} did not return a torch.nn.Module"
        )

    if load_state or devices:
        use_devices = get_devices(devices)
        network = network.to(use_devices[0])
        if load_state:
            network = load_torch_state_dict(
                network,
                path=download(weight_spec).path,
                devices=use_devices,
            )
    return network


def load_torch_state_dict(
    model: nn.Module,
    path: Union[Path, ZipPath],
    devices: Sequence[torch.device],
) -> nn.Module:
    model = model.to(devices[0])
    with path.open("rb") as f:
        assert not isinstance(f, TextIOWrapper)
        state = torch.load(f, map_location=devices[0])

    incompatible = model.load_state_dict(state)
    if incompatible.missing_keys:
        logger.warning("Missing state dict keys: {}", incompatible.missing_keys)

    if incompatible.unexpected_keys:
        logger.warning("Unexpected state dict keys: {}", incompatible.unexpected_keys)
    return model


def get_devices(
    devices: Optional[Sequence[Union[torch.device, str]]] = None,
) -> List[torch.device]:
    if not devices:
        torch_devices = [
            (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        ]
    else:
        torch_devices = [torch.device(d) for d in devices]

    if len(torch_devices) > 1:
        warnings.warn(
            f"Multiple devices for single pytorch model not yet implemented; ignoring {torch_devices[1:]}"
        )
        torch_devices = torch_devices[:1]

    return torch_devices
