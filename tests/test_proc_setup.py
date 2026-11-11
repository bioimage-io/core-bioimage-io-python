from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from bioimageio.core._utils.type_hints import cast_and_assert
from bioimageio.core.common import MemberId
from bioimageio.core.proc_ops import (
    AddKnownDatasetStats,
    Clip,
    UpdateStats,
    ZeroMeanUnitVariance,
)
from bioimageio.core.proc_setup import setup_pre_and_postprocessing
from bioimageio.core.sample import Sample
from bioimageio.core.stat_measures import (
    DatasetMean,
    DatasetStd,
    SampleMean,
    SampleStd,
)
from bioimageio.core.tensor import Tensor
from bioimageio.spec.model import v0_5


def _iter_raises_if_consumed():
    raise AssertionError("dataset_for_initial_statistics should not be consumed")
    yield  # pragma: no cover


def test_statistics_include_preceding_processing():
    tid = MemberId("input")
    axes = [v0_5.AxisId("batch"), v0_5.AxisId("x")]
    input_descr = v0_5.InputTensorDescr.model_construct(
        id=tid,
        preprocessing=[
            v0_5.ClipDescr(kwargs=v0_5.ClipKwargs(min=-1, max=1)),
            v0_5.ZeroMeanUnitVarianceDescr(
                kwargs=v0_5.ZeroMeanUnitVarianceKwargs(axes=axes)
            ),
        ],
    )
    model = SimpleNamespace(inputs=[input_descr], outputs=[])
    data = np.asarray([[-10.0, 0.0, 10.0]], dtype=np.float32)

    def sample() -> Sample:
        return Sample(
            members={tid: Tensor(data, dims=("batch", "x"))}, stat={}, id=None
        )

    pre, _ = setup_pre_and_postprocessing(
        model=model,  # pyright: ignore[reportArgumentType]
        dataset_for_initial_statistics=[sample()],
    )
    actual = sample()
    for proc in pre:
        proc(actual)

    scale = (2.0 / 3.0) ** 0.5 + 1e-6
    expected: NDArray[np.float32] = np.asarray(
        [[-1.0 / scale, 0.0, 1.0 / scale]], dtype=np.float32
    )
    np.testing.assert_allclose(actual.members[tid].to_numpy(), expected)

    normalization = pre[-1]
    assert isinstance(normalization, ZeroMeanUnitVariance)
    required = normalization.get_required_measures(normalization.preceding_ops)
    assert all(len(measure.preceding_ops) == 1 for measure in required)
    assert all(isinstance(measure.preceding_ops[0], Clip) for measure in required)


def test_preceding_processing_uses_its_own_measures():
    tid = MemberId("input")
    input_descr = v0_5.InputTensorDescr.model_construct(
        id=tid,
        preprocessing=[
            v0_5.ClipDescr(
                kwargs=v0_5.ClipKwargs(
                    min_percentile=25,
                    max_percentile=75,
                    axes=[v0_5.AxisId("x")],
                )
            ),
            v0_5.ZeroMeanUnitVarianceDescr(
                kwargs=v0_5.ZeroMeanUnitVarianceKwargs(
                    axes=[v0_5.AxisId("batch"), v0_5.AxisId("x")]
                )
            ),
        ],
    )
    model = SimpleNamespace(inputs=[input_descr], outputs=[])
    data = np.asarray([[0.0, 1.0, 2.0, 100.0]], dtype=np.float32)

    def sample() -> Sample:
        return Sample(
            members={tid: Tensor(data, dims=("batch", "x"))}, stat={}, id=None
        )

    pre, _ = setup_pre_and_postprocessing(
        model=model,  # pyright: ignore[reportArgumentType]
        dataset_for_initial_statistics=[sample()],
    )
    actual = sample()
    for proc in pre:
        proc(actual)

    normalized = actual.members[tid].to_numpy()
    np.testing.assert_allclose(normalized.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(), 1.0, atol=1e-5)


def test_missing_dataset_stats_raise_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    tid = MemberId("input")
    mean = DatasetMean(member_id=tid)
    std = DatasetStd(member_id=tid)
    op = ZeroMeanUnitVariance(input=tid, output=tid, mean=mean, std=std)
    model = SimpleNamespace(inputs=[object()], outputs=[object()])

    def fake_get_described_procs(tensor_descrs: Any):
        return [op] if tensor_descrs is model.inputs else []

    monkeypatch.setattr(
        "bioimageio.core.proc_setup._get_described_procs", fake_get_described_procs
    )

    with pytest.raises(ValueError, match="dataset_for_initial_statistics"):
        _ = setup_pre_and_postprocessing(
            model=model,  # pyright: ignore[reportArgumentType]
            dataset_for_initial_statistics=(),
        )


def test_missing_dataset_stats_can_be_updated_during_prediction(
    monkeypatch: pytest.MonkeyPatch,
):
    tid = MemberId("input")
    mean = DatasetMean(member_id=tid)
    std = DatasetStd(member_id=tid)
    op = ZeroMeanUnitVariance(input=tid, output=tid, mean=mean, std=std)
    model = SimpleNamespace(inputs=[object()], outputs=[object()])

    def fake_get_described_procs(tensor_descrs: Any):
        return [op] if tensor_descrs is model.inputs else []

    monkeypatch.setattr(
        "bioimageio.core.proc_setup._get_described_procs", fake_get_described_procs
    )

    pre, _ = setup_pre_and_postprocessing(
        model=model,  # pyright: ignore[reportArgumentType]
        dataset_for_initial_statistics=(),
        keep_updating_initial_dataset_stats=True,
    )

    assert isinstance(pre[0], UpdateStats)


def test_fixed_sample_stats_are_not_computed_on_the_fly(
    monkeypatch: pytest.MonkeyPatch,
):
    tid = MemberId("input")
    mean = SampleMean(member_id=tid)
    std = SampleStd(member_id=tid)
    op = ZeroMeanUnitVariance(input=tid, output=tid, mean=mean, std=std)

    model = SimpleNamespace(inputs=[object()], outputs=[object()])

    def fake_get_described_procs(tensor_descrs: Any):
        return [op] if tensor_descrs is model.inputs else []

    monkeypatch.setattr(
        "bioimageio.core.proc_setup._get_described_procs", fake_get_described_procs
    )

    pre, _ = setup_pre_and_postprocessing(
        model=model,  # pyright: ignore[reportArgumentType]
        dataset_for_initial_statistics=_iter_raises_if_consumed(),
        fixed_dataset_stats={mean: 5.0, std: 2.0},
    )

    assert not any(isinstance(p, UpdateStats) for p in pre)
    assert isinstance(pre[0], AddKnownDatasetStats)
    assert pre[0].dataset_stats == {}

    sample = Sample(
        members={tid: Tensor.from_xarray(xr.DataArray(np.arange(4), dims=("x",)))},
        stat={},
        id=None,
    )

    with pytest.raises(KeyError):
        for proc in pre:
            proc(sample)


def test_fixed_sample_stats_must_be_present_in_sample_stat(
    monkeypatch: pytest.MonkeyPatch,
):
    tid = MemberId("input")
    mean = SampleMean(member_id=tid)
    std = SampleStd(member_id=tid)
    op = ZeroMeanUnitVariance(input=tid, output=tid, mean=mean, std=std)

    model = SimpleNamespace(inputs=[object()], outputs=[object()])

    def fake_get_described_procs(tensor_descrs: Any):
        return [op] if tensor_descrs is model.inputs else []

    monkeypatch.setattr(
        "bioimageio.core.proc_setup._get_described_procs", fake_get_described_procs
    )

    pre, _ = setup_pre_and_postprocessing(
        model=model,  # pyright: ignore[reportArgumentType]
        dataset_for_initial_statistics=(),
        fixed_dataset_stats={mean: 1.0, std: 2.0},
    )

    data = xr.DataArray(np.arange(4, dtype=np.float32), dims=("x",))
    sample = Sample(
        members={tid: Tensor.from_xarray(data)},
        stat={mean: 10.0, std: 5.0},
        id=None,
    )

    for proc in pre:
        proc(sample)

    expected = cast_and_assert(xr.DataArray, (data - 10.0) / (5.0 + op.eps))
    xr.testing.assert_allclose(expected, sample.members[tid].data)
