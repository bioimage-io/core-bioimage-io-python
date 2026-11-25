from typing import Any
from unittest.mock import Mock

import pytest

from bioimageio.core._description_serializer import DescriptionSerializer
from bioimageio.core._settings import settings
from bioimageio.core.axis import AxisId, AxisSize
from bioimageio.core.common import MemberId
from bioimageio.core.remote_backends.gradio import client
from bioimageio.core.sample import Sample
from bioimageio.core.stat_measures import DatasetMean
from bioimageio.spec import ValidationSummary
from bioimageio.spec.model import v0_5

# pyright: reportPrivateUsage=false


@pytest.fixture
def model():
    return v0_5.ModelDescr.model_construct(inputs=[], outputs=[])


@pytest.fixture
def remote(monkeypatch: pytest.MonkeyPatch):
    connection = Mock()
    monkeypatch.setattr(client, "Client", Mock(return_value=connection))
    monkeypatch.setattr(
        DescriptionSerializer,
        "serialize_to_string_and_hash",
        Mock(return_value=("package", "digest")),
    )
    return connection


@pytest.mark.parametrize("pipeline", [False, True])
def test_server_configuration(
    monkeypatch: pytest.MonkeyPatch,
    model: v0_5.ModelDescr,
    remote: Mock,
    pipeline: bool,
):
    factory = client.GradioPredictionPipeline if pipeline else client.GradioModelAdapter
    monkeypatch.setattr(settings, "gradio_server", None)
    with pytest.raises(ValueError, match="No gradio server specified"):
        _ = factory(model)
    monkeypatch.setattr(settings, "gradio_server", "configured-server")
    assert factory(model).server == "configured-server"
    assert factory(model, server="explicit-server").server == "explicit-server"


@pytest.mark.parametrize("method", ["load", "test"])
@pytest.mark.parametrize(
    "outcomes",
    [
        [True],
        [False, True],
        [RuntimeError("cache miss"), True],
        [RuntimeError("offline"), RuntimeError("offline")],
        [False, False],
    ],
)
def test_adapter_retries(
    model: v0_5.ModelDescr,
    remote: Mock,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    outcomes: list[Any],
):
    adapter = client.GradioModelAdapter(model, server="local")
    remote.submit.return_value.result.side_effect = outcomes
    validate = Mock()
    monkeypatch.setattr(ValidationSummary, "model_validate_json", validate)
    result = getattr(adapter, method)()
    assert [call.kwargs["model"] for call in remote.submit.call_args_list] == (
        [""] if len(outcomes) == 1 else ["", "package"]
    )
    for call in remote.submit.call_args_list:
        assert call.kwargs["api_name"] == f"/{method}_model"
        assert call.kwargs["sha256"] == "digest"
    if method == "test":
        if outcomes[-1] is True:
            assert result is validate.return_value
            validate.assert_called_once_with(True)
        else:
            assert result is None
            validate.assert_not_called()


def test_adapter_forward_skips_remote_processing(
    model: v0_5.ModelDescr, remote: Mock, monkeypatch: pytest.MonkeyPatch
):
    adapter = client.GradioModelAdapter(model, server="local")
    predict = Mock()
    monkeypatch.setattr(client, "_call_predict_api", predict)
    assert adapter._forward_impl({}) is predict.return_value
    predict.assert_called_once_with(
        remote,
        "package",
        "digest",
        {},
        blocksize=None,
        skip_preprocessing=True,
        skip_postprocessing=True,
        skip_input_padding=True,
        skip_output_cropping=True,
        batch_size=None,
    )
    adapter.unload()
    assert not adapter.is_loaded


def test_pipeline_connection_lifecycle(model: v0_5.ModelDescr, remote: Mock):
    pipeline = client.GradioPredictionPipeline(model, server="local")
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = pipeline.get_status()
    pipeline.unload()
    pipeline.load()
    remote.submit.return_value.result.return_value = {"message": "alive"}
    assert pipeline.get_status() == {"message": "alive"}
    remote.submit.assert_called_once_with(api_name="/get_status")
    remote.submit.return_value.result.assert_called_once_with(timeout=30)
    pipeline.unload()
    remote.close.assert_called_once_with()
    assert pipeline._client is None
    pipeline.load()
    assert pipeline._client is remote


def test_in_process_pipeline(model: v0_5.ModelDescr, remote: Mock):
    pipeline = client.GradioPredictionPipeline(model, server="run_in_client_process")
    pipeline.load()
    assert pipeline.get_status() == {"message": "dummy client"}
    pipeline.unload()
    assert pipeline._client is None
    remote.submit.assert_not_called()


@pytest.mark.parametrize("blocking", [False, True])
def test_pipeline_prediction(
    model: v0_5.ModelDescr,
    remote: Mock,
    monkeypatch: pytest.MonkeyPatch,
    blocking: bool,
):
    measure = DatasetMean(member_id=MemberId("input"))
    pipeline = client.GradioPredictionPipeline(
        model,
        server="local",
        precomputed_statistics={measure: 2.0},
        default_batch_size=3,
    )
    sample = Sample(id="sample", members={}, stat={})
    kwargs = {
        "skip_preprocessing": True,
        "skip_postprocessing": False,
        "skip_input_padding": True,
        "skip_output_cropping": False,
    }
    block_shape = {MemberId("input"): {AxisId("x"): AxisSize(4, 8, 2)}}

    def predict_sample() -> Sample:
        if blocking:
            return pipeline._predict_with_fixed_blocking_impl(
                sample, block_shape, **kwargs
            )
        return pipeline._predict_without_blocking_impl(sample, **kwargs)

    with pytest.raises(RuntimeError, match="not initialized"):
        _ = predict_sample()
    pipeline.load()
    serializer = Mock()
    monkeypatch.setattr(pipeline, "_serializer", serializer)
    predict = Mock()
    monkeypatch.setattr(client, "_call_predict_api", predict)
    assert predict_sample() is serializer.deserialize_sample.return_value
    assert sample.stat[measure] == 2.0
    serializer.serialize_sample.assert_called_once_with(sample)
    serializer.deserialize_sample.assert_called_once_with(predict.return_value)
    predict.assert_called_once_with(
        remote,
        "package",
        "digest",
        serialized_input_sample=serializer.serialize_sample.return_value,
        blocksize=block_shape if blocking else None,
        batch_size=3,
        **kwargs,
    )


def test_prediction_retry_propagates_upload_failure():
    remote = Mock()
    remote.submit.return_value.result.side_effect = [
        RuntimeError("cache miss"),
        ValueError("invalid model"),
    ]
    with pytest.raises(ValueError, match="invalid model"):
        _ = client._call_predict_api(
            remote, "package", "digest", {}, None, False, False, False, False, None
        )
    assert remote.submit.call_count == 2


def test_mixed_fixed_block_sizes_are_json_compatible():
    remote = Mock()
    result = client._call_predict_api(
        remote,
        "package",
        "digest",
        {},
        {MemberId("raw"): {AxisId("x"): AxisSize(4, 8, 2), AxisId("y"): 16}},
        False,
        False,
        False,
        False,
        None,
    )
    assert result is remote.submit.return_value.result.return_value
    assert remote.submit.call_args.kwargs["blocksize"] == {
        "raw": {"x": {"min": 4, "max": 8, "step": 2}, "y": 16}
    }
    assert remote.submit.call_args.kwargs["model"] == ""
    remote.submit.return_value.result.assert_called_once_with(timeout=300)


def test_dummy_client_rejects_unknown_api():
    dummy: Any = client._DummyClient()
    with pytest.raises(ValueError, match="Unknown api_name"):
        _ = dummy.submit(api_name="/unknown")
