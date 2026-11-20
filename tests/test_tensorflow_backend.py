from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

_ = pytest.importorskip("tensorflow")

from bioimageio.core.backends import tensorflow_backend as backend
from bioimageio.spec.model import v0_4, v0_5

# pyright: reportPrivateUsage=false


@pytest.mark.parametrize(
    "adapter_type", [backend.TensorflowModelAdapter, backend.KerasModelAdapter]
)
@pytest.mark.parametrize("old", [False, True])
def test_adapter_weight_sources(adapter_type: Any, old: bool):
    weight = SimpleNamespace(source="weights.zip")
    model = (v0_4.ModelDescr if old else v0_5.ModelDescr).model_construct(
        inputs=[],
        outputs=[],
        weights=SimpleNamespace(tensorflow_saved_model_bundle=weight),
    )
    adapter = adapter_type(model)
    assert adapter._weight_src == ("weights.zip" if old else weight)
    assert adapter._parse_devices(None) == (None,)
    assert adapter._parse_devices(["cpu"]) == (None,)
    adapter._cleanup_pre_model_deletion(None, Mock())
    adapter._cleanup_post_model_deletion(None)
    model.weights.tensorflow_saved_model_bundle = None
    with pytest.raises(ValueError, match="No `tensorflow_saved_model_bundle` weights"):
        _ = adapter_type(model)


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_keras_saved_model_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failures: int
):
    adapter = object.__new__(backend.KerasModelAdapter)
    adapter._weight_src = Mock()
    monkeypatch.setattr(backend, "ensure_unzipped", Mock(return_value=tmp_path))
    first, second = RuntimeError("serve failed"), RuntimeError("default failed")
    layer = Mock()
    factory = Mock(
        side_effect=[first, second]
        if failures == 2
        else [first, layer]
        if failures
        else [layer]
    )
    monkeypatch.setattr(
        backend,
        "tf",
        SimpleNamespace(
            keras=SimpleNamespace(layers=SimpleNamespace(TFSMLayer=factory))
        ),
    )
    if failures == 2:
        with pytest.raises(RuntimeError) as caught:
            _ = adapter._init_model_on_device(None)
        assert caught.value is first
    else:
        assert adapter._init_model_on_device(None) is layer
    assert [call.kwargs["call_endpoint"] for call in factory.call_args_list] == (
        ["serve"] if not failures else ["serve", "serving_default"]
    )


def test_keras_forward_converts_inputs_and_outputs(monkeypatch: pytest.MonkeyPatch):
    data = np.arange(3)
    converted = Mock()
    convert = Mock(return_value=converted)
    monkeypatch.setattr(backend, "tf", SimpleNamespace(convert_to_tensor=convert))
    tensor_output = Mock()
    tensor_output.numpy.return_value = data + 1
    model = Mock(
        return_value={"raw": data, "converted": tensor_output, "optional": None}
    )
    adapter = object.__new__(backend.KerasModelAdapter)
    result = cast(list[Any], adapter._forward_impl(None, model, [data, None]))
    convert.assert_called_once_with(data)
    model.assert_called_once_with(converted, None)
    np.testing.assert_array_equal(result[0], data)
    np.testing.assert_array_equal(result[1], data + 1)
    assert result[2] is None


def test_tf1_load_and_forward(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    adapter = object.__new__(backend.TensorflowModelAdapter)
    adapter._weight_src = Mock()
    adapter._input_ids = [v0_5.TensorId("input")]
    adapter._output_ids = [v0_5.TensorId("output")]
    graph = Mock()
    graph.as_default.return_value = Mock(
        __enter__=Mock(), __exit__=Mock(return_value=False)
    )
    input_tensor, output_tensor = object(), object()
    graph.get_tensor_by_name.side_effect = [input_tensor, output_tensor]
    session = Mock()
    data = np.arange(3)
    session.run.return_value = {"output:0": data}
    signature = SimpleNamespace(
        inputs={"input": SimpleNamespace(name="input:0")},
        outputs={"output": SimpleNamespace(name="output:0")},
    )
    loader = Mock(return_value=SimpleNamespace(signature_def={"default": signature}))
    fake_tf = SimpleNamespace(
        Graph=Mock(return_value=graph),
        Session=Mock(return_value=session),
        saved_model=SimpleNamespace(
            tag_constants=SimpleNamespace(SERVING="serve"),
            signature_constants=SimpleNamespace(
                DEFAULT_SERVING_SIGNATURE_DEF_KEY="default"
            ),
            loader=SimpleNamespace(load=loader),
        ),
    )
    monkeypatch.setattr(backend, "tf", fake_tf)
    monkeypatch.setattr(backend, "ensure_unzipped", Mock(return_value=tmp_path))
    assert adapter._init_model_on_device(None) is session
    loader.assert_called_once_with(session, ["serve"], str(tmp_path))
    result = adapter._forward_impl(None, session, [data])
    np.testing.assert_array_equal(result[0], data)
    session.run.assert_called_once_with(
        {"output:0": output_tensor}, {input_tensor: data}
    )


@pytest.mark.parametrize(
    "installed,specified,legacy",
    [
        ("1.15.0", "1.15.0", True),
        ("2.15.0", None, False),
        ("2.15.0", "2.16.0", False),
        ("2.15.0", "2.14.0", False),
        ("2.15.0", "2.15.0", False),
    ],
)
def test_tf_version_dispatch(
    monkeypatch: pytest.MonkeyPatch, installed: str, specified: str | None, legacy: bool
):
    monkeypatch.setattr(backend, "tf", SimpleNamespace(__version__=installed))
    tf1, tf2 = Mock(), Mock()
    monkeypatch.setattr(backend, "TensorflowModelAdapter", tf1)
    monkeypatch.setattr(backend, "KerasModelAdapter", tf2)
    model = v0_5.ModelDescr.model_construct(
        weights=SimpleNamespace(
            tensorflow_saved_model_bundle=SimpleNamespace(
                tensorflow_version=None
                if specified is None
                else v0_5.Version(specified)
            )
        )
    )
    factory, other = (tf1, tf2) if legacy else (tf2, tf1)
    assert (
        backend.create_tf_model_adapter(model, devices=["cpu"]) is factory.return_value
    )
    factory.assert_called_once_with(model, devices=["cpu"])
    other.assert_not_called()
    model.weights.tensorflow_saved_model_bundle = None
    with pytest.raises(ValueError, match="No `tensorflow_saved_model_bundle` weights"):
        _ = backend.create_tf_model_adapter(model)
