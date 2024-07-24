import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union, final

from bioimageio.spec.model import v0_4, v0_5

from ..tensor import Tensor

WeightsFormat = Union[v0_4.WeightsFormat, v0_5.WeightsFormat]

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
        if not isinstance(model_description, (v0_4.ModelDescr, v0_5.ModelDescr)):
            raise TypeError(
                f"expected v0_4.ModelDescr or v0_5.ModelDescr, but got {type(model_description)}"
            )

        weights = model_description.weights
        errors: List[Tuple[WeightsFormat, Exception]] = []
        weight_format_priority_order = (
            DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER
            if weight_format_priority_order is None
            else weight_format_priority_order
        )
        # limit weight formats to the ones present
        weight_format_priority_order = [
            w for w in weight_format_priority_order if getattr(weights, w) is not None
        ]

        for wf in weight_format_priority_order:
            if wf == "pytorch_state_dict" and weights.pytorch_state_dict is not None:
                try:
                    from ._pytorch_model_adapter import PytorchModelAdapter

                    return PytorchModelAdapter(
                        outputs=model_description.outputs,
                        weights=weights.pytorch_state_dict,
                        devices=devices,
                    )
                except Exception as e:
                    errors.append((wf, e))
            elif (
                wf == "tensorflow_saved_model_bundle"
                and weights.tensorflow_saved_model_bundle is not None
            ):
                try:
                    from ._tensorflow_model_adapter import TensorflowModelAdapter

                    return TensorflowModelAdapter(
                        model_description=model_description, devices=devices
                    )
                except Exception as e:
                    errors.append((wf, e))
            elif wf == "onnx" and weights.onnx is not None:
                try:
                    from ._onnx_model_adapter import ONNXModelAdapter

                    return ONNXModelAdapter(
                        model_description=model_description, devices=devices
                    )
                except Exception as e:
                    errors.append((wf, e))
            elif wf == "torchscript" and weights.torchscript is not None:
                try:
                    from ._torchscript_model_adapter import TorchscriptModelAdapter

                    return TorchscriptModelAdapter(
                        model_description=model_description, devices=devices
                    )
                except Exception as e:
                    errors.append((wf, e))
            elif wf == "keras_hdf5" and weights.keras_hdf5 is not None:
                # keras can either be installed as a separate package or used as part of tensorflow
                # we try to first import the keras model adapter using the separate package and,
                # if it is not available, try to load the one using tf
                try:
                    from ._keras_model_adapter import (
                        KerasModelAdapter,
                        keras,  # type: ignore
                    )

                    if keras is None:
                        from ._tensorflow_model_adapter import KerasModelAdapter

                    return KerasModelAdapter(
                        model_description=model_description, devices=devices
                    )
                except Exception as e:
                    errors.append((wf, e))

        assert errors
        if len(weight_format_priority_order) == 1:
            assert len(errors) == 1
            raise ValueError(
                f"The '{weight_format_priority_order[0]}' model adapter could not be created"
                + f" in this environment:\n{errors[0][1].__class__.__name__}({errors[0][1]}).\n\n"
            )

        else:
            error_list = "\n - ".join(
                f"{wf}: {e.__class__.__name__}({e})" for wf, e in errors
            )
            raise ValueError(
                "None of the weight format specific model adapters could be created"
                + f" in this environment. Errors are:\n\n{error_list}.\n\n"
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


create_model_adapter = ModelAdapter.create
