from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from exceptiongroup import ExceptionGroup
from typing_extensions import assert_never

from bioimageio.spec.model import v0_4, v0_5

from ..common import SupportedWeightsFormat

# Known weight formats in order of priority
# First match wins
DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER: Tuple[SupportedWeightsFormat, ...] = (
    "pytorch_state_dict",
    "tensorflow_saved_model_bundle",
    "torchscript",
    "onnx",
    "keras_v3",
    "keras_hdf5",
)


def create_model_adapter(
    model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
    *,
    devices: Optional[Sequence[str]] = None,
    weight_format_priority_order: Optional[Sequence[SupportedWeightsFormat]] = None,
):
    """Creates model adapter for `model_descritption`"""
    if not isinstance(model_description, (v0_4.ModelDescr, v0_5.ModelDescr)):
        raise TypeError(
            f"expected v0_4.ModelDescr or v0_5.ModelDescr, but got {type(model_description)}"
        )

    weights = model_description.weights
    errors: List[Exception] = []
    weight_format_priority_order = (
        DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER
        if weight_format_priority_order is None
        else weight_format_priority_order
    )
    # limit weight formats to the ones present
    weight_format_priority_order_present: Sequence[SupportedWeightsFormat] = [
        w for w in weight_format_priority_order if getattr(weights, w, None) is not None
    ]
    if not weight_format_priority_order_present:
        raise ValueError(
            f"None of the specified weight formats ({weight_format_priority_order}) is present ({weight_format_priority_order_present})"
        )

    for wf in weight_format_priority_order_present:
        if wf == "pytorch_state_dict":
            assert weights.pytorch_state_dict is not None
            try:
                from .pytorch_backend import PytorchModelAdapter

                return PytorchModelAdapter(model_description, devices=devices)
            except Exception as e:
                errors.append(e)
        elif wf == "tensorflow_saved_model_bundle":
            assert weights.tensorflow_saved_model_bundle is not None
            try:
                from .tensorflow_backend import create_tf_model_adapter

                return create_tf_model_adapter(model_description, devices=devices)
            except Exception as e:
                errors.append(e)
        elif wf == "onnx":
            assert weights.onnx is not None
            try:
                from .onnx_backend import ONNXModelAdapter

                return ONNXModelAdapter(model_description, devices=devices)
            except Exception as e:
                errors.append(e)
        elif wf == "torchscript":
            assert weights.torchscript is not None
            try:
                from .torchscript_backend import TorchscriptModelAdapter

                return TorchscriptModelAdapter(model_description, devices=devices)
            except Exception as e:
                errors.append(e)
        elif wf == "keras_hdf5":
            assert weights.keras_hdf5 is not None
            # keras can either be installed as a separate package or used as part of tensorflow
            # we try to first import the keras model adapter using the separate package and,
            # if it is not available, try to load the one using tf
            try:
                try:
                    from .keras_backend import KerasModelAdapter
                except Exception:
                    from .tensorflow_backend import KerasModelAdapter

                return KerasModelAdapter(model_description, devices=devices)
            except Exception as e:
                errors.append(e)
        elif wf == "keras_v3":
            assert not isinstance(weights, v0_4.WeightsDescr), (
                "keras_v3 weights not supported for v0.4 specs"
            )
            assert weights.keras_v3 is not None
            try:
                from .keras_backend import KerasModelAdapter

                return KerasModelAdapter(model_description, devices=devices)
            except Exception as e:
                errors.append(e)
        else:
            assert_never(wf)

    assert errors
    if len(weight_format_priority_order) == 1:
        assert len(errors) == 1
        raise errors[0]

    else:
        msg = (
            "None of the weight format specific model adapters could be created"
            + " in this environment."
        )
        raise ExceptionGroup(msg, errors)
