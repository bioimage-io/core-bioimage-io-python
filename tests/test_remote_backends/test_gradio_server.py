from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import gradio as gr
import pytest

import bioimageio.core
from bioimageio.core._description_serializer import DescriptionSerializer
from bioimageio.core.remote_backends.gradio import server
from bioimageio.spec.common import Sha256
from bioimageio.spec.model import v0_4, v0_5

# pyright: reportPrivateUsage=false


@pytest.mark.parametrize("digest", ["", "1" * 64])
def test_model_source_loading(monkeypatch: pytest.MonkeyPatch, digest: str):
    load = Mock()
    monkeypatch.setattr(server, "load_model_description", load)
    assert server._get_model("model-source", sha256=digest) is load.return_value
    load.assert_called_once_with(
        "model-source", sha256=Sha256(digest) if digest else None
    )


@pytest.mark.parametrize("old", [False, True])
def test_uploaded_description(monkeypatch: pytest.MonkeyPatch, old: bool):
    model = (v0_4.ModelDescr if old else v0_5.ModelDescr).model_construct()
    deserialize = Mock(return_value=model)
    monkeypatch.setattr(DescriptionSerializer, "deserialize_from_string", deserialize)
    assert server._get_model("x" * 2084, sha256="") is model
    deserialize.assert_called_once_with("x" * 2084)
    deserialize.return_value = SimpleNamespace(type="dataset", format_version="0.3")
    with pytest.raises(
        ValueError, match="not a valid model description: got dataset 0.3"
    ):
        _ = server._get_model("x" * 2084, sha256="")


def test_empty_model_cache_miss():
    with pytest.raises(ValueError, match="Model source cannot be empty"):
        _ = server._get_model_adapter("", sha256="unit-test-empty-cache", devices=None)


def test_server_status_and_load(monkeypatch: pytest.MonkeyPatch):
    get_adapter = Mock()
    monkeypatch.setattr(server, "_get_model_adapter", get_adapter)
    assert server.load_model("source", "digest", devices=["cpu"]) == {
        "message": "Model loaded successfully"
    }
    get_adapter.assert_called_once_with("source", sha256="digest", devices=["cpu"])
    assert server.get_status() == {"message": "alive"}
    assert bioimageio.core.__version__ in server.root()["message"]


def test_app_routes_bind_device_configuration(monkeypatch: pytest.MonkeyPatch):
    endpoints: dict[str, Any] = {}

    def register(name: str):
        def decorate(func: Any):
            endpoints[name] = func
            return func

        return decorate

    app = Mock()
    app.api.side_effect = register
    app.get.side_effect = register
    monkeypatch.setattr(gr, "Server", Mock(return_value=app))
    handlers = {
        name: Mock()
        for name in ("predict", "load_model", "get_status", "test_model", "root")
    }
    for name, handler in handlers.items():
        monkeypatch.setattr(server, name, handler)
    devices = ["cpu"]
    assert server._create_app(devices=devices) is app
    devices.append("other")
    assert endpoints["predict"]("model", "hash", {}) is handlers["predict"].return_value
    assert handlers["predict"].call_args.kwargs == {"devices": ("cpu",)}
    for name in ("load_model", "test_model"):
        assert endpoints[name]("model", "hash") is handlers[name].return_value
        handlers[name].assert_called_once_with("model", "hash", devices=("cpu",))
    assert endpoints["get_status"]() is handlers["get_status"].return_value
    assert endpoints["/"]() is handlers["root"].return_value
