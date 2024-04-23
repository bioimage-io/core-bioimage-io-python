from typing import Union

from bioimageio.core.model_adapters._pytorch_model_adapter import PytorchModelAdapter
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.utils import download

try:
    import torch
except ImportError:
    torch = None


# additional convenience for pytorch state dict, eventually we want this in python-bioimageio too
# and for each weight format
def load_torch_model(  # pyright: ignore[reportUnknownParameterType]
    node: Union[v0_4.PytorchStateDictWeightsDescr, v0_5.PytorchStateDictWeightsDescr],
):
    assert torch is not None
    model = (  # pyright: ignore[reportUnknownVariableType]
        PytorchModelAdapter.get_network(node)
    )
    state = torch.load(download(node.source).path, map_location="cpu")
    model.load_state_dict(state)  # FIXME: check incompatible keys?
    return model.eval()  # pyright: ignore[reportUnknownVariableType]
