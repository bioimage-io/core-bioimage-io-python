import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from exceptiongroup import ExceptionGroup

from bioimageio.core.backends import create_model_adapter
from bioimageio.spec.model import v0_5

FORMATS = [
    ("pytorch_state_dict", "pytorch_backend", "PytorchModelAdapter"),
    ("tensorflow_saved_model_bundle", "tensorflow_backend", "create_tf_model_adapter"),
    ("torchscript", "torchscript_backend", "TorchscriptModelAdapter"),
    ("onnx", "onnx_backend", "ONNXModelAdapter"),
    ("keras_v3", "keras_backend", "KerasModelAdapter"),
    ("keras_hdf5", "keras_backend", "KerasModelAdapter"),
]


def model_with_weights(*formats: str):
    return v0_5.ModelDescr.model_construct(
        weights=SimpleNamespace(**{name: object() for name in formats})
    )


def install_backend(monkeypatch: pytest.MonkeyPatch, name: str, symbol: str):
    module = ModuleType(f"bioimageio.core.backends.{name}")
    factory = Mock()
    setattr(module, symbol, factory)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return factory


@pytest.mark.parametrize("format,module,symbol", FORMATS)
def test_select_backend(
    monkeypatch: pytest.MonkeyPatch, format: Any, module: str, symbol: str
):
    factory = install_backend(monkeypatch, module, symbol)
    model = model_with_weights(format)
    assert (
        create_model_adapter(
            model, devices=["cpu"], weight_format_priority_order=[format]
        )
        is factory.return_value
    )
    factory.assert_called_once_with(model, devices=["cpu"])


def test_selection_respects_priority_and_falls_back(monkeypatch: pytest.MonkeyPatch):
    pytorch = install_backend(monkeypatch, "pytorch_backend", "PytorchModelAdapter")
    onnx = install_backend(monkeypatch, "onnx_backend", "ONNXModelAdapter")
    model = model_with_weights("pytorch_state_dict", "onnx")
    assert create_model_adapter(model) is pytorch.return_value
    onnx.assert_not_called()
    pytorch.side_effect = RuntimeError("torch unavailable")
    assert create_model_adapter(model) is onnx.return_value


@pytest.mark.parametrize("format,module,symbol", FORMATS)
def test_single_backend_error_is_preserved(
    monkeypatch: pytest.MonkeyPatch, format: Any, module: str, symbol: str
):
    factory = install_backend(monkeypatch, module, symbol)
    error = RuntimeError("cannot load weights")
    factory.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        _ = create_model_adapter(
            model_with_weights(format), weight_format_priority_order=[format]
        )
    assert caught.value is error


def test_all_backend_errors_are_reported(monkeypatch: pytest.MonkeyPatch):
    errors = [RuntimeError("torch"), ValueError("onnx")]
    install_backend(
        monkeypatch, "pytorch_backend", "PytorchModelAdapter"
    ).side_effect = errors[0]
    install_backend(
        monkeypatch, "onnx_backend", "ONNXModelAdapter"
    ).side_effect = errors[1]
    with pytest.raises(ExceptionGroup) as caught:
        _ = create_model_adapter(model_with_weights("pytorch_state_dict", "onnx"))
    assert list(caught.value.exceptions) == errors


def test_keras_import_falls_back_to_tensorflow(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "bioimageio.core.backends.keras_backend", None)
    factory = install_backend(monkeypatch, "tensorflow_backend", "KerasModelAdapter")
    model = model_with_weights("keras_hdf5")
    assert create_model_adapter(model) is factory.return_value


def test_selection_rejects_wrong_description_and_missing_weights():
    with pytest.raises(TypeError, match="expected v0_4.ModelDescr"):
        _ = create_model_adapter(Mock())
    with pytest.raises(ValueError, match="None of the specified weight formats"):
        _ = create_model_adapter(model_with_weights())
    with pytest.raises(ValueError, match="None of the specified weight formats"):
        _ = create_model_adapter(
            model_with_weights("onnx"), weight_format_priority_order=[]
        )
