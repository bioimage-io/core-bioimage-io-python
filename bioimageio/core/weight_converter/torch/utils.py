import torch
from marshmallow import missing


# NOTE: copied from tiktorch; this should go into python-bioimageio and then we use it from there
def get_nn_instance(node, **kwargs):
    joined_kwargs = {} if node.kwargs is missing else dict(node.kwargs)  # type: ignore
    joined_kwargs.update(kwargs)
    model = node.source(**joined_kwargs)
    return model


# additional convenience for pytorch state dict, eventually we want this in python-bioimageio too
# and for each weight format
def load_model(node):
    model = get_nn_instance(node)
    state = torch.load(node.weights["pytorch_state_dict"].source)
    model.load_state_dict(state)
    model.eval()
    return model
