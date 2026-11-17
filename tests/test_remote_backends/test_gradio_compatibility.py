from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

# pyright: reportPrivateUsage=false
import bioimageio.core
from bioimageio.core.axis import AxisId, AxisSize, PerAxis
from bioimageio.core.common import MemberId, PerMember
from bioimageio.core.remote_backends.gradio import client, server
from bioimageio.core.remote_backends.gradio.serializer import GradioSampleSerializer
from bioimageio.core.sample import Sample
from bioimageio.core.tensor import Tensor


@pytest.mark.parametrize(
    "blocksize", [None, 2, {MemberId("raw"): {AxisId("x"): AxisSize(2, 4, 2)}}]
)
def test_prediction_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    blocksize: int | PerMember[PerAxis[int | AxisSize]] | None,
):
    sample = Sample(
        id="sample",
        members={MemberId("raw"): Tensor(np.arange(4), dims=["x"])},
        stat={},
    )
    adapter = Mock()
    pipeline = Mock()
    pipeline.predict_without_blocking.return_value = sample
    pipeline.predict_with_blocking.return_value = sample
    pipeline.predict_with_fixed_blocking.return_value = sample
    create = Mock(return_value=pipeline)
    monkeypatch.setattr(server, "_get_model_adapter", Mock(return_value=adapter))
    monkeypatch.setattr(bioimageio.core, "create_prediction_pipeline", create)
    result = client._call_predict_api(
        client._DummyClient(),
        "model",
        "hash",
        GradioSampleSerializer.serialize_sample(sample),
        blocksize=blocksize,
        skip_preprocessing=True,
        skip_postprocessing=False,
        skip_input_padding=True,
        skip_output_cropping=False,
        batch_size=2,
    )
    actual = GradioSampleSerializer.deserialize_sample(result)
    np.testing.assert_array_equal(
        actual.members[MemberId("raw")].to_numpy(), np.arange(4)
    )
    create.assert_called_once_with(
        adapter.model_descr, model_adapter=adapter, fixed_dataset_statistics={}
    )
    kwargs = {
        "skip_preprocessing": True,
        "skip_postprocessing": False,
        "skip_input_padding": True,
        "skip_output_cropping": False,
    }
    if blocksize is None:
        pipeline.predict_without_blocking.assert_called_once()
        assert pipeline.predict_without_blocking.call_args.kwargs == kwargs
    elif isinstance(blocksize, int):
        assert pipeline.predict_with_blocking.call_args.kwargs == dict(
            kwargs, ns=2, batch_size=2
        )
    else:
        assert pipeline.predict_with_fixed_blocking.call_args.args[1] == blocksize
        assert pipeline.predict_with_fixed_blocking.call_args.kwargs == kwargs


def test_cache_miss_retries_with_model():
    remote = Mock()
    remote.submit.return_value.result.side_effect = [
        ValueError("cache miss"),
        {"id": "result"},
    ]
    result = client._call_predict_api(
        remote,
        "uploaded model",
        "hash",
        {"id": "input"},
        blocksize=None,
        skip_preprocessing=False,
        skip_postprocessing=False,
        skip_input_padding=False,
        skip_output_cropping=False,
        batch_size=None,
    )
    assert result == {"id": "result"}
    assert [call.kwargs["model"] for call in remote.submit.call_args_list] == [
        "",
        "uploaded model",
    ]


def test_pipeline_implements_current_interface():
    assert not client.GradioPredictionPipeline.__abstractmethods__


def test_prediction_api_type_hints():
    from inspect import signature
    from typing import get_type_hints

    hints = get_type_hints(server.predict)
    assert set(signature(server.predict).parameters) <= hints.keys()


def test_cached_adapter_is_loaded(monkeypatch: pytest.MonkeyPatch):
    adapter = Mock()
    monkeypatch.setattr(server, "_get_model", Mock())
    monkeypatch.setattr(server, "create_model_adapter", Mock(return_value=adapter))
    first = server._get_model_adapter(
        "model", sha256="test-cached-adapter-is-loaded", devices=None
    )
    second = server._get_model_adapter(
        "", sha256="test-cached-adapter-is-loaded", devices=None
    )
    assert first is second is adapter
    adapter.load.assert_called_once_with()


def test_explicit_server_devices_select_separate_cached_adapters(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(server, "_get_model", Mock())
    automatic_adapter, cpu_adapter = Mock(), Mock()
    create = Mock(side_effect=[automatic_adapter, cpu_adapter])
    monkeypatch.setattr(server, "create_model_adapter", create)

    automatic = server._get_model_adapter(
        "model", sha256="test-device-cache", devices=None
    )
    cpu = server._get_model_adapter(
        "model", sha256="test-device-cache", devices=["cpu"]
    )
    assert (
        server._get_model_adapter("", sha256="test-device-cache", devices=["cpu"])
        is cpu
    )
    assert automatic is automatic_adapter
    assert cpu is cpu_adapter
    assert [call.kwargs["devices"] for call in create.call_args_list] == [
        None,
        ("cpu",),
    ]
    automatic_adapter.load.assert_called_once_with()
    cpu_adapter.load.assert_called_once_with()

    test_model = Mock()
    monkeypatch.setattr(bioimageio.core, "test_model", test_model)
    _ = server.test_model("", sha256="test-device-cache", devices=["cpu"])
    test_model.assert_called_once_with(cpu_adapter.model_descr, devices=["cpu"])


def test_main_binds_devices_to_new_app(monkeypatch: pytest.MonkeyPatch):
    launch = Mock(return_value=(None, "http://localhost:7861/", None))
    app = Mock(launch=launch)
    create_app = Mock(return_value=app)
    monkeypatch.setattr(server, "_create_app", create_app)

    devices = ["cpu"]
    assert server.main(port=7861, devices=devices) == "http://localhost:7861/"
    create_app.assert_called_once_with(devices=devices)
    launch.assert_called_once()
