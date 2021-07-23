import abc
from typing import Callable, List, Optional, Type

import xarray as xr
from bioimageio.spec.model import nodes

#: Known weigh types in order of priority
#: First match wins
_WEIGHT_FORMATS = ["pytorch_state_dict", "tensorflow_saved_model_bundle", "pytorch_script", "onnx"]


class ModelAdapter(abc.ABC):
    """
    Represents model *without* any preprocessing and postprocessing
    """

    @abc.abstractmethod
    def __init__(self, *, bioimageio_model: nodes.Model, devices=List[str]):
        ...

    @abc.abstractmethod
    def forward(self, input_tensor: xr.DataArray) -> xr.DataArray:
        """
        Run forward pass of model to get model predictions
        Note: model is responsible converting it's data representation to
        xarray.DataArray
        """
        ...

    # Training methods
    @property
    def max_num_iterations(self) -> int:
        return 0

    @property
    def iteration_count(self) -> int:
        return 0

    def set_break_callback(self, thunk: Callable[[], bool]) -> None:
        pass

    def set_max_num_iterations(self, val: int) -> None:
        pass


def get_weight_formats() -> List[str]:
    """
    Return list of supported weight types
    """
    return _WEIGHT_FORMATS.copy()


def create_model_adapter(
    *, bioimageio_model: nodes.Model, devices=List[str], weight_format: Optional[str] = None
) -> ModelAdapter:
    """
    Creates model adapter based on the passed spec
    Note: All specific adapters should happen inside this function to prevent different framework
    initializations interfering with each other
    """
    spec = bioimageio_model
    weights = bioimageio_model.weights
    weight_formats = get_weight_formats()

    if weight_format is not None:
        if weight_format not in weight_formats:
            raise ValueError(f"Weight format {weight_format} is not in supported formats {_WEIGHT_FORMATS}")
        weight_formats = [weight_format]

    for weight in weight_formats:
        if weight in weights:
            adapter_cls = _get_model_adapter(weight)
            return adapter_cls(bioimageio_model=bioimageio_model, devices=devices)

    raise NotImplementedError(f"No supported weight_formats in {spec.weights.keys()}")


def _get_model_adapter(weight_format: str) -> Type[ModelAdapter]:
    """
    Return adapter class based on the weight format
    Note: All specific adapters should happen inside this function to prevent different framework
    initializations interfering with each other
    """
    if weight_format == "pytorch_state_dict":
        from ._pytorch_model_adapter import PytorchModelAdapter

        return PytorchModelAdapter

    elif weight_format == "tensorflow_saved_model_bundle":
        from ._tensorflow_model_adapter import TensorflowModelAdapter

        return TensorflowModelAdapter

    elif weight_format == "onnx":
        from ._onnx_model_adapter import ONNXModelAdapter

        return ONNXModelAdapter

    elif weight_format == "pytorch_script":
        from ._torchscript_model_adapter import TorchscriptModelAdapter

        return TorchscriptModelAdapter

    else:
        raise ValueError(f"Weight format {weight_format} is not supported.")
