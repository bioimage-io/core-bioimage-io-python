from pathlib import Path
from typing import Any, Tuple, Union

import torch
from torch.jit import ScriptModule

from bioimageio.spec._internal.version_type import Version
from bioimageio.spec.model.v0_5 import ModelDescr, TorchscriptWeightsDescr

from .. import __version__
from ..backends.pytorch_backend import load_torch_model


def convert(
    model_descr: ModelDescr,
    output_path: Path,
    *,
    use_tracing: bool = True,
) -> TorchscriptWeightsDescr:
    """
    Convert model weights from the PyTorch `state_dict` format to TorchScript.

    Args:
        model_descr:
            The model description object that contains the model and its weights in the PyTorch `state_dict` format.
        output_path:
            The file path where the TorchScript model will be saved.
        use_tracing:
            Whether to use tracing or scripting to export the TorchScript format.
            - `True`: Use tracing, which is recommended for models with straightforward control flow.
            - `False`: Use scripting, which is better for models with dynamic control flow (e.g., loops, conditionals).

    Raises:
        ValueError:
            If the provided model does not have weights in the PyTorch `state_dict` format.

    Returns:
        A descriptor object that contains information about the exported TorchScript weights.
    """
    state_dict_weights_descr = model_descr.weights.pytorch_state_dict
    if state_dict_weights_descr is None:
        raise ValueError(
            "The provided model does not have weights in the pytorch state dict format"
        )

    input_data = model_descr.get_input_test_arrays()

    with torch.no_grad():
        input_data = [torch.from_numpy(inp) for inp in input_data]
        model = load_torch_model(state_dict_weights_descr, load_state=True)
        scripted_model: Union[  # pyright: ignore[reportUnknownVariableType]
            ScriptModule, Tuple[Any, ...]
        ] = (
            torch.jit.trace(model, input_data)
            if use_tracing
            else torch.jit.script(model)
        )
        assert not isinstance(scripted_model, tuple), scripted_model

    scripted_model.save(output_path)

    return TorchscriptWeightsDescr(
        source=output_path.absolute(),
        pytorch_version=Version(torch.__version__),
        parent="pytorch_state_dict",
        comment=(
            f"Converted with bioimageio.core {__version__}"
            + f" with use_tracing={use_tracing}."
        ),
    )
