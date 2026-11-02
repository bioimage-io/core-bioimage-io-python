from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any, Literal
from unittest.mock import Mock

import dask.array as da
import numpy as np
import pytest
from dask import config
from exceptiongroup import ExceptionGroup
from numpy.typing import NDArray

from bioimageio.core import AxisId, MemberId, Tensor
from bioimageio.core._model_adapter import LocalModelAdapter
from bioimageio.spec.model import v0_5

# pyright: reportPrivateUsage=false


class _ArrayAdapter(LocalModelAdapter[str, str]):
    """Exercise adapter scheduling and lifecycle without a framework or hardware."""

    def _parse_devices(self, devices: Sequence[str] | None) -> Sequence[str]:
        return devices or ["cpu"]

    def _init_model_on_device(self, device: str) -> str:
        return device

    def _forward_impl(
        self, device: str, model: str, input_arrays: Sequence[NDArray[Any] | None]
    ) -> list[NDArray[Any] | None]:
        assert model == device
        array = input_arrays[0]
        assert array is not None
        return [array + 1]

    def _cleanup_pre_model_deletion(self, device: str, model: str) -> None:
        assert device == model

    def _cleanup_post_model_deletion(self, device: str) -> None:
        pass


@pytest.fixture
def adapter() -> _ArrayAdapter:
    model = v0_5.ModelDescr.model_construct(
        inputs=[
            v0_5.InputTensorDescr(
                id=MemberId("input"),
                axes=[
                    v0_5.SpaceInputAxis(
                        id=AxisId("x"), size=v0_5.ParameterizedSize(min=1, step=1)
                    )
                ],
            )
        ],
        outputs=[
            v0_5.OutputTensorDescr(
                id=MemberId("output"),
                axes=[
                    v0_5.SpaceOutputAxis(
                        id=AxisId("x"),
                        size=v0_5.SizeReference(
                            tensor_id=MemberId("input"), axis_id=AxisId("x")
                        ),
                    )
                ],
            )
        ],
    )
    return _ArrayAdapter(model, devices=["cpu"], computation_mode="eager")


@pytest.mark.parametrize("mode", ["eager", "lazy"])
@pytest.mark.parametrize("lazy_input", [False, True])
def test_forward_evaluation_mode(
    adapter: _ArrayAdapter,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["eager", "lazy"],
    lazy_input: bool,
):
    adapter.computation_mode = mode
    forward = Mock(wraps=adapter._forward_impl)
    monkeypatch.setattr(adapter, "_forward_impl", forward)
    adapter.load()
    tensor = Tensor(np.arange(4, dtype=np.float32), dims=["x"])
    output = adapter.forward(
        {MemberId("input"): tensor.as_dask() if lazy_input else tensor}
    )[MemberId("output")]
    assert output is not None
    assert forward.call_count == (0 if mode == "lazy" else 1)
    assert isinstance(output.data.data, da.Array) == (mode == "lazy")
    assert output.dims == tensor.dims
    assert output.dtype == tensor.dtype
    np.testing.assert_array_equal(output.to_numpy(), np.arange(4) + 1)
    assert forward.call_count == 1
    adapter.unload()


def test_forward_requires_loaded_model(adapter: _ArrayAdapter):
    with pytest.raises(RuntimeError, match="Model must be loaded"):
        _ = adapter.forward({})


def test_lazy_forward_after_unload(
    adapter: _ArrayAdapter, monkeypatch: pytest.MonkeyPatch
):
    adapter.computation_mode = "lazy"
    adapter.load()
    output = adapter.forward(
        {MemberId("input"): Tensor(np.arange(4, dtype=np.float32), dims=["x"])}
    )[MemberId("output")]
    assert output is not None
    np.testing.assert_array_equal(output.to_numpy(), np.arange(4) + 1)

    adapter.unload()
    # A missing unloaded-state check must fail immediately rather than block.
    get = Mock(side_effect=AssertionError("Accessed unloaded queue"))
    monkeypatch.setattr(adapter._model_queue, "get", get)
    with pytest.raises(RuntimeError, match="Compute lazy outputs inside"):
        _ = output.to_numpy()
    get.assert_not_called()


def test_inference_failure_returns_device_to_queue(
    adapter: _ArrayAdapter, monkeypatch: pytest.MonkeyPatch
):
    expected = np.arange(4)
    forward = Mock(side_effect=[RuntimeError("inference failed"), [expected]])
    monkeypatch.setattr(adapter, "_forward_impl", forward)
    adapter.load()
    inputs = {MemberId("input"): Tensor(expected, dims=["x"])}
    with pytest.raises(RuntimeError, match="inference failed"):
        _ = adapter.forward(inputs)
    assert adapter._model_queue.qsize() == 1
    output = adapter.forward(inputs)[MemberId("output")]
    assert output is not None
    np.testing.assert_array_equal(output.to_numpy(), expected)
    assert adapter._stats.forward_calls == {"cpu": 2}
    assert adapter._stats.forward_time["cpu"] >= 0
    adapter.unload()


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_concurrent_forward_keeps_inputs_separate(
    adapter: _ArrayAdapter,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["eager", "lazy"],
):
    adapter._devices = ["first", "last"]
    adapter.computation_mode = mode
    barrier = Barrier(2, timeout=5)
    forward = adapter._forward_impl

    def synchronized_forward(
        device: str, model: str, arrays: Sequence[NDArray[Any] | None]
    ) -> list[NDArray[Any] | None]:
        _ = barrier.wait()
        return forward(device, model, arrays)

    monkeypatch.setattr(adapter, "_forward_impl", synchronized_forward)
    adapter.load()
    tensors = [
        Tensor(np.arange(4, dtype=np.float32) + i * 10, dims=["x"]) for i in range(4)
    ]

    def run(tensor: Tensor) -> NDArray[Any]:
        output = adapter.forward({MemberId("input"): tensor})[MemberId("output")]
        assert output is not None
        return output.to_numpy()

    # Run Dask tasks in these workers so both inference calls meet at the barrier.
    with (
        config.set(scheduler="synchronous"),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        outputs = list(executor.map(run, tensors))
    for tensor, output in zip(tensors, outputs):
        np.testing.assert_array_equal(output, tensor.to_numpy() + 1)
    assert sum(adapter._stats.forward_calls.values()) == len(tensors)
    adapter.unload()


@pytest.mark.parametrize("failed_device", [None, "first", "last"])
def test_load_preserves_device_priority_after_partial_failure(
    adapter: _ArrayAdapter,
    monkeypatch: pytest.MonkeyPatch,
    failed_device: str | None,
):
    adapter._devices = ["first", "last"]

    def initialize(device: str) -> str:
        if device == failed_device:
            raise RuntimeError("device unavailable")
        return device

    monkeypatch.setattr(adapter, "_init_model_on_device", initialize)
    forward = Mock(wraps=adapter._forward_impl)
    monkeypatch.setattr(adapter, "_forward_impl", forward)
    adapter.load()
    assert adapter.is_loaded
    _ = adapter.forward({MemberId("input"): Tensor(np.arange(4), dims=["x"])})
    assert forward.call_args.args[0] == (
        "last" if failed_device == "first" else "first"
    )
    adapter.unload()


@pytest.mark.parametrize("devices", [["cpu"], ["first", "last"]])
def test_load_reports_all_initialization_failures(
    adapter: _ArrayAdapter, monkeypatch: pytest.MonkeyPatch, devices: list[str]
):
    adapter._devices = devices
    errors = {device: ValueError(device) for device in devices}

    def fail(device: str) -> str:
        raise errors[device]

    monkeypatch.setattr(adapter, "_init_model_on_device", fail)
    if len(devices) == 1:
        with pytest.raises(ValueError) as caught:
            adapter.load()
        assert caught.value is errors[devices[0]]
    else:
        with pytest.raises(ExceptionGroup) as caught_group:
            adapter.load()
        assert caught_group.value.exceptions == tuple(errors.values())
    assert not adapter.is_loaded
    assert adapter._model_queue.empty()


@pytest.mark.parametrize("failed_cleanup", ["pre", "post"])
def test_unload_continues_after_cleanup_failure(
    adapter: _ArrayAdapter,
    monkeypatch: pytest.MonkeyPatch,
    failed_cleanup: str,
):
    adapter._devices = ["first", "last"]
    before = Mock(
        side_effect=RuntimeError("cleanup") if failed_cleanup == "pre" else None
    )
    after = Mock(
        side_effect=RuntimeError("cleanup") if failed_cleanup == "post" else None
    )
    monkeypatch.setattr(adapter, "_cleanup_pre_model_deletion", before)
    monkeypatch.setattr(adapter, "_cleanup_post_model_deletion", after)
    adapter.load()
    adapter.unload()
    assert not adapter.is_loaded
    assert adapter._model_queue.empty()
    assert [call.args[0] for call in before.call_args_list] == ["first", "last"]
    assert [call.args[0] for call in after.call_args_list] == ["first", "last"]
    adapter.unload()
    assert before.call_count == after.call_count == 2
