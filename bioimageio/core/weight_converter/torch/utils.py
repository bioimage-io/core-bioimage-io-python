import torch
from bioimageio.core.prediction_pipeline._model_adapters._pytorch_model_adapter import PytorchModelAdapter


# additional convenience for pytorch state dict, eventually we want this in python-bioimageio too
# and for each weight format
def load_model(node):
    model = PytorchModelAdapter.get_nn_instance(node)
    state = torch.load(node.weights["pytorch_state_dict"].source)
    model.load_state_dict(state)
    model.eval()
    return model
