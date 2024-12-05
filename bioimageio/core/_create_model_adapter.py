import warnings
from abc import abstractmethod
from typing import List, Optional, Sequence, Tuple, Union, final

from bioimageio.spec.model import v0_4, v0_5

from ._model_adapter import (
    DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER,
    ModelAdapter,
    WeightsFormat,
)
from .tensor import Tensor


def create_model_adapter(
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
                from .model_adapters_old._pytorch_model_adapter import (
                    PytorchModelAdapter,
                )

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
                from .model_adapters_old._tensorflow_model_adapter import (
                    TensorflowModelAdapter,
                )

                return TensorflowModelAdapter(
                    model_description=model_description, devices=devices
                )
            except Exception as e:
                errors.append((wf, e))
        elif wf == "onnx" and weights.onnx is not None:
            try:
                from .model_adapters_old._onnx_model_adapter import ONNXModelAdapter

                return ONNXModelAdapter(
                    model_description=model_description, devices=devices
                )
            except Exception as e:
                errors.append((wf, e))
        elif wf == "torchscript" and weights.torchscript is not None:
            try:
                from .model_adapters_old._torchscript_model_adapter import (
                    TorchscriptModelAdapter,
                )

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
                from .backend.keras import (
                    KerasModelAdapter,
                    keras,  # type: ignore
                )

                if keras is None:
                    from .model_adapters_old._tensorflow_model_adapter import (
                        KerasModelAdapter,
                    )

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
        ) from errors[0][1]

    else:
        error_list = "\n - ".join(
            f"{wf}: {e.__class__.__name__}({e})" for wf, e in errors
        )
        raise ValueError(
            "None of the weight format specific model adapters could be created"
            + f" in this environment. Errors are:\n\n{error_list}.\n\n"
        )
