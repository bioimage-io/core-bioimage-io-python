import gc
import warnings
from contextlib import nullcontext
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Union

import torch
from loguru import logger
from numpy.typing import NDArray
from torch import nn
from typing_extensions import assert_never

from bioimageio.spec._internal.version_type import Version
from bioimageio.spec.common import BytesReader, ZipPath
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.utils import download

from ..digest_spec import import_callable
from ..utils._type_guards import is_list, is_ndarray, is_tuple
from ._model_adapter import ModelAdapter


class PytorchModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        model_description: AnyModelDescr,
        devices: Optional[Sequence[Union[str, torch.device]]] = None,
        mode: Literal["eval", "train"] = "eval",
    ):
        super().__init__(model_description=model_description)
        weights = model_description.weights.pytorch_state_dict
        if weights is None:
            raise ValueError("No `pytorch_state_dict` weights found")

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

    def _forward_impl(
        self, input_arrays: Sequence[Optional[NDArray[Any]]]
    ) -> List[Optional[NDArray[Any]]]:
        tensors = [
            None if a is None else torch.from_numpy(a).to(self._primary_device)
            for a in input_arrays
        ]

        if self._mode == "eval":
            ctxt = torch.no_grad
        elif self._mode == "train":
            ctxt = nullcontext
        else:
            assert_never(self._mode)

        with ctxt():
            model_out = self._model(*tensors)

        if is_tuple(model_out) or is_list(model_out):
            model_out_seq = model_out
        else:
            model_out_seq = model_out = [model_out]

        result: List[Optional[NDArray[Any]]] = []
        for i, r in enumerate(model_out_seq):
            if r is None:
                result.append(None)
            elif isinstance(r, torch.Tensor):
                r_np: NDArray[Any] = (  # pyright: ignore[reportUnknownVariableType]
                    r.detach().cpu().numpy()
                )
                result.append(r_np)
            elif is_ndarray(r):
                result.append(r)
            else:
                raise TypeError(f"Model output[{i}] has unexpected type {type(r)}.")

        return result

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
    load_state: bool = True,
    devices: Optional[Sequence[Union[str, torch.device]]] = None,
) -> nn.Module:
    custom_callable = import_callable(
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
    torch_model = custom_callable(**model_kwargs)

    if not isinstance(torch_model, nn.Module):
        if isinstance(
            weight_spec.architecture,
            (v0_4.CallableFromFile, v0_4.CallableFromDepencency),
        ):
            callable_name = weight_spec.architecture.callable_name
        else:
            callable_name = weight_spec.architecture.callable

        raise ValueError(f"Calling {callable_name} did not return a torch.nn.Module.")

    if load_state or devices:
        use_devices = get_devices(devices)
        torch_model = torch_model.to(use_devices[0])
        if load_state:
            torch_model = load_torch_state_dict(
                torch_model,
                path=download(weight_spec),
                devices=use_devices,
            )
    return torch_model


def load_torch_state_dict(
    model: nn.Module,
    path: Union[Path, ZipPath, BytesReader],
    devices: Sequence[torch.device],
) -> nn.Module:
    model = model.to(devices[0])
    if isinstance(path, (Path, ZipPath)):
        ctxt = path.open("rb")
    else:
        ctxt = nullcontext(BytesIO(path.read()))

    with ctxt as f:
        assert not isinstance(f, TextIOWrapper)
        if Version(str(torch.__version__)) < Version("1.13"):
            state = torch.load(f, map_location=devices[0])
        else:
            state = torch.load(f, map_location=devices[0], weights_only=True)

    incompatible = model.load_state_dict(state)
    if (
        isinstance(incompatible, tuple)
        and hasattr(incompatible, "missing_keys")
        and hasattr(incompatible, "unexpected_keys")
    ):
        if incompatible.missing_keys:
            logger.warning("Missing state dict keys: {}", incompatible.missing_keys)

        if hasattr(incompatible, "unexpected_keys") and incompatible.unexpected_keys:
            logger.warning(
                "Unexpected state dict keys: {}", incompatible.unexpected_keys
            )
    else:
        logger.warning(
            "`model.load_state_dict()` unexpectedly returned: {} "
            + "(expected named tuple with `missing_keys` and `unexpected_keys` attributes)",
            (s[:20] + "..." if len(s := str(incompatible)) > 20 else s),
        )

    return model


def get_devices(
    devices: Optional[Sequence[Union[torch.device, str]]] = None,
) -> List[torch.device]:
    if not devices:
        torch_devices = [
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ]
    else:
        torch_devices = [torch.device(d) for d in devices]

    if len(torch_devices) > 1:
        warnings.warn(
            f"Multiple devices for single pytorch model not yet implemented; ignoring {torch_devices[1:]}"
        )
        torch_devices = torch_devices[:1]

    return torch_devices
