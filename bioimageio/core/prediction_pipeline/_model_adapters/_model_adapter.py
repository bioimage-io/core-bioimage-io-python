import abc
from typing import List, Optional, Sequence, Type

import xarray as xr

from bioimageio.core.resource_io import nodes

#: Known weight formats in order of priority
#: First match wins
_WEIGHT_FORMATS = ["pytorch_state_dict", "tensorflow_saved_model_bundle", "pytorch_script", "onnx", "keras_hdf5"]


class ModelAdapter(abc.ABC):
    """
    Represents model *without* any preprocessing and postprocessing
    """

    def __init__(self, *, bioimageio_model: nodes.Model, devices: Optional[Sequence[str]] = None):
        self.bioimageio_model = bioimageio_model
        self.default_devices = devices
        self.loaded = False

    def __enter__(self):
        """load on entering context"""
        assert not self.loaded
        self.load()  # using default_devices
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """unload on exiting context"""
        assert self.loaded
        self.unload()
        return False

    def load(self, *, devices: Optional[Sequence[str]] = None) -> None:
        """
        Note: Use ModelAdapter as context to not worry about calling unload()!
        Load model onto devices. If devices is None, self.default_devices are chosen
        (which may be None as well, in which case a framework dependent default is chosen)
        """
        self._load(devices=devices or self.default_devices)
        self.loaded = True

    @abc.abstractmethod
    def _load(self, *, devices: Optional[Sequence[str]] = None) -> None:
        """
        Load model onto devices. If devices is None a framework dependent default is chosen
        """
        ...

    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """
        Load model if unloaded/outside context; then run forward pass of model to get model predictions
        """
        if not self.loaded:
            self.load()

        assert self.loaded
        return self._forward(*input_tensors)

    @abc.abstractmethod
    def _forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """
        Run forward pass of model to get model predictions
        Note: model is responsible converting it's data representation to
        xarray.DataArray
        """
        ...

    def unload(self):
        """
        Unload model from any devices, freeing their memory.
        Note: Use ModelAdapter as context to not worry about calling unload()!
        """
        # implementation of non-state-machine logic in _unload()
        assert self.loaded
        self._unload()
        self.loaded = False

    @abc.abstractmethod
    def _unload(self) -> None:
        """
        Unload model from any devices, freeing their memory.
        """
        ...


def get_weight_formats() -> List[str]:
    """
    Return list of supported weight types
    """
    return _WEIGHT_FORMATS.copy()


def create_model_adapter(
    *, bioimageio_model: nodes.Model, devices=Optional[List[str]], weight_format: Optional[str] = None
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

    raise RuntimeError(
        f"weight format {weight_format} not among weight formats listed in model: {list(spec.weights.keys())}"
    )


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

    elif weight_format == "keras_hdf5":
        # keras can either be installed as a separate package or used as part of tensorflow
        # we try to first import the keras model adapter using the separate package and,
        # if it is not available, try to load the one using tf
        try:
            from ._keras_model_adapter import KerasModelAdapter
        except ImportError:
            from ._tensorflow_model_adapter import KerasModelAdapter

        return KerasModelAdapter

    else:
        raise ValueError(f"Weight format {weight_format} is not supported.")
