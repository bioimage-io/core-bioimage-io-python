import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union, final

from bioimageio.spec.model import v0_4, v0_5

from .tensor import Tensor

WeightsFormat = Union[v0_4.WeightsFormat, v0_5.WeightsFormat]

__all__ = [
    "ModelAdapter",
    "create_model_adapter",
    "get_weight_formats",
]

# Known weight formats in order of priority
# First match wins
DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER: Tuple[WeightsFormat, ...] = (
    "pytorch_state_dict",
    "tensorflow_saved_model_bundle",
    "torchscript",
    "onnx",
    "keras_hdf5",
)


class ModelAdapter(ABC):
    """
    Represents model *without* any preprocessing or postprocessing.

    ```
    from bioimageio.core import load_description

    model = load_description(...)

    # option 1:
    adapter = ModelAdapter.create(model)
    adapter.forward(...)
    adapter.unload()

    # option 2:
    with ModelAdapter.create(model) as adapter:
        adapter.forward(...)
    ```
    """

    @final
    @classmethod
    def create(
        cls,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        *,
        devices: Optional[Sequence[str]] = None,
        weight_format_priority_order: Optional[Sequence[WeightsFormat]] = None,
    ):
        """
        Creates model adapter based on the passed spec
        Note: All specific adapters should happen inside this function to prevent different framework
        initializations interfering with each other
        """
        from ._create_model_adapter import create_model_adapter

        return create_model_adapter(
            model_description,
            devices=devices,
            weight_format_priority_order=weight_format_priority_order,
        )

    @final
    def load(self, *, devices: Optional[Sequence[str]] = None) -> None:
        warnings.warn("Deprecated. ModelAdapter is loaded on initialization")

    @abstractmethod
    def forward(self, *input_tensors: Optional[Tensor]) -> List[Optional[Tensor]]:
        """
        Run forward pass of model to get model predictions
        """
        # TODO: handle tensor.transpose in here and make _forward_impl the abstract impl

    @abstractmethod
    def unload(self):
        """
        Unload model from any devices, freeing their memory.
        The moder adapter should be considered unusable afterwards.
        """


def get_weight_formats() -> List[str]:
    """
    Return list of supported weight types
    """
    return list(DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER)
