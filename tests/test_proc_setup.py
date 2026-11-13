from types import SimpleNamespace
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from bioimageio.core._op_base import get_relevant_preceding_ops
from bioimageio.core._utils.type_hints import cast_and_assert
from bioimageio.core.common import MemberId
from bioimageio.core.proc_ops import (
    AddKnownDatasetStats,
    Clip,
    EnsureDtype,
    ScaleRange,
    UpdateStats,
    ZeroMeanUnitVariance,
)
from bioimageio.core.proc_setup import setup_pre_and_postprocessing
from bioimageio.core.sample import Sample
from bioimageio.core.stat_measures import (
    DatasetMean,
    DatasetStd,
    SampleMean,
    SampleQuantile,
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
    required = normalization.get_required_measures()
    assert required == {normalization.mean, normalization.std}
    assert all(len(measure.preceding_ops) == 1 for measure in required)
    assert all(isinstance(measure.preceding_ops[0], Clip) for measure in required)


def test_statistics_provenance_filters_only_explicitly_preserving_ops():
    class PreservingClip(Clip):
        @property
        def affects_statistics(self) -> bool:
            return False

    tid = MemberId("input")
    cast_op = EnsureDtype(input=tid, output=tid, dtype="float32")
    ignored = PreservingClip(input=tid, output=tid, min=0.0, max=1.0)
    clip = Clip(input=tid, output=tid, min=0.0, max=1.0)
    relevant = get_relevant_preceding_ops(tid, (cast_op, ignored, clip))
    op = ZeroMeanUnitVariance(
        input=tid,
        output=tid,
        mean=SampleMean(member_id=tid, preceding_ops=relevant),
        std=SampleStd(member_id=tid, preceding_ops=relevant),
    )
    required = op.get_required_measures()
    assert all(measure.preceding_ops == (cast_op, clip) for measure in required)
    assert cast_op.affects_statistics


def test_dataset_statistics_are_calculated_at_their_preprocessing_stage(
    monkeypatch: pytest.MonkeyPatch,
):
    tid = MemberId("input")
    mean = DatasetMean(
        member_id=tid,
        axes=(v0_5.AxisId("batch"), v0_5.AxisId("x")),
    )
    std = DatasetStd(
        member_id=tid,
        axes=(v0_5.AxisId("batch"), v0_5.AxisId("x")),
    )
    clip = Clip(
        input=tid,
        output=tid,
        min=SampleQuantile(member_id=tid, axes=(v0_5.AxisId("x"),), q=0.25),
        max=SampleQuantile(member_id=tid, axes=(v0_5.AxisId("x"),), q=0.75),
    )
    normalization = ZeroMeanUnitVariance(
        input=tid,
        output=tid,
        mean=mean.model_copy(update={"preceding_ops": (clip,)}),
        std=std.model_copy(update={"preceding_ops": (clip,)}),
    )
    model = SimpleNamespace(inputs=[object()], outputs=[object()])

    def fake_get_described_procs(tensor_descrs: Any, **kwargs: Any):
        return [clip, normalization] if tensor_descrs is model.inputs else []

    monkeypatch.setattr(
        "bioimageio.core.proc_setup._get_described_procs", fake_get_described_procs
    )

    samples = [
        Sample(
            members={
                tid: Tensor(
                    np.asarray([[-10.0, 0.0, 10.0]], dtype=np.float32),
                    dims=("batch", "x"),
                )
            },
            stat={},
            id=None,
        ),
        Sample(
            members={
                tid: Tensor(
                    np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
                    dims=("batch", "x"),
                )
            },
            stat={},
            id=None,
        ),
    ]
    pre, _ = setup_pre_and_postprocessing(
        model=model,  # pyright: ignore[reportArgumentType]
        dataset_for_initial_statistics=samples,
    )

    assert isinstance(pre[0], UpdateStats)
    assert isinstance(pre[1], Clip)
    assert isinstance(pre[2], UpdateStats)
    processed_samples: list[Sample] = []
    for sample in samples:
        sample = sample.copy()
        for proc in pre:
            proc(sample)
        processed_samples.append(sample)

        assert mean.model_copy(update={"preceding_ops": (clip,)}) in sample.stat
        assert std.model_copy(update={"preceding_ops": (clip,)}) in sample.stat

    expected_scale = (25.0 / 3.0) ** 0.5 + normalization.eps
    np.testing.assert_allclose(
        cast(NDArray[np.float32], processed_samples[0].members[tid].to_numpy()),
        np.asarray([[-5.0 / expected_scale, 0.0, 5.0 / expected_scale]]),
    )


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


def test_multistage_preprocessing_and_input_referenced_output_scaling():
    input_id = MemberId("input")
    output_id = MemberId("output")
    axes = [v0_5.AxisId("batch"), v0_5.AxisId("x")]
    input_descr = v0_5.InputTensorDescr.model_construct(
        id=input_id,
        preprocessing=[
            v0_5.ClipDescr(
                kwargs=v0_5.ClipKwargs(
                    min_percentile=25,
                    max_percentile=75,
                    axes=[v0_5.AxisId("x")],
                )
            ),
            v0_5.ZeroMeanUnitVarianceDescr(
                kwargs=v0_5.ZeroMeanUnitVarianceKwargs(axes=axes)
            ),
        ],
    )
    output_descr = v0_5.OutputTensorDescr.model_construct(
        id=output_id,
        postprocessing=[
            v0_5.ScaleRangeDescr(
                kwargs=v0_5.ScaleRangeKwargs(
                    reference_tensor=input_id,
                    axes=axes,
                    min_percentile=25,
                    max_percentile=75,
                )
            )
        ],
    )
    model = SimpleNamespace(inputs=[input_descr], outputs=[output_descr])

    def sample(data: list[float]) -> Sample:
        return Sample(
            members={
                input_id: Tensor(
                    cast(
                        da.Array,
                        da.from_array(
                            np.asarray([data], dtype=np.float32), chunks="auto"
                        ),
                    ),
                    dims=("batch", "x"),
                )
            },
            stat={},
            id=None,
        )

    initial_samples = [sample([0, 1, 2, 100]), sample([0, 2, 4, 8])]
    pre, post = setup_pre_and_postprocessing(
        model=model,  # pyright: ignore[reportArgumentType]
        dataset_for_initial_statistics=initial_samples,
    )
    actual = initial_samples[0].copy()
    for proc in pre:
        proc(actual)

    actual.members[output_id] = Tensor(
        cast(
            da.Array,
            da.from_array(
                np.asarray([[0.75, 2.875, 5.0, 10.0]], dtype=np.float32),
                chunks="auto",
            ),
        ),
        dims=("batch", "x"),
    )
    for proc in post:
        proc(actual)

    scale = post[-1]
    assert isinstance(scale, ScaleRange)
    assert scale.lower.member_id == input_id
    assert scale.lower.preceding_ops == tuple(
        proc for proc in pre if isinstance(proc, (Clip, ZeroMeanUnitVariance))
    )
    low = actual.stat[scale.lower]
    high = actual.stat[scale.upper]
    low_value = low.to_numpy().item() if isinstance(low, Tensor) else low
    high_value = high.to_numpy().item() if isinstance(high, Tensor) else high
    expected = (np.asarray([[0.75, 2.875, 5.0, 10.0]]) - low_value) / (
        high_value - low_value + scale.eps
    )
    np.testing.assert_allclose(actual.members[output_id].to_numpy(), expected)


def test_dataset_statistics_on_model_outputs_are_not_supported():
    output_id = MemberId("output")
    output_descr = v0_5.OutputTensorDescr.model_construct(
        id=output_id,
        postprocessing=[
            v0_5.ScaleRangeDescr(
                kwargs=v0_5.ScaleRangeKwargs(
                    reference_tensor=output_id,
                    axes=[v0_5.AxisId("batch"), v0_5.AxisId("x")],
                )
            )
        ],
    )
    model = SimpleNamespace(inputs=[], outputs=[output_descr])

    with pytest.raises(NotImplementedError, match="statistics on model outputs"):
        _ = setup_pre_and_postprocessing(
            model=model,  # pyright: ignore[reportArgumentType]
            dataset_for_initial_statistics=(),
        )


def test_missing_dataset_stats_raise_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    tid = MemberId("input")
    mean = DatasetMean(member_id=tid)
    std = DatasetStd(member_id=tid)
    op = ZeroMeanUnitVariance(input=tid, output=tid, mean=mean, std=std)
    model = SimpleNamespace(inputs=[object()], outputs=[object()])

    def fake_get_described_procs(tensor_descrs: Any, **kwargs: Any):
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

    def fake_get_described_procs(tensor_descrs: Any, **kwargs: Any):
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


def test_sample_statistics_do_not_consume_initial_dataset(
    monkeypatch: pytest.MonkeyPatch,
):
    tid = MemberId("input")
    mean = SampleMean(member_id=tid)
    std = SampleStd(member_id=tid)
    op = ZeroMeanUnitVariance(input=tid, output=tid, mean=mean, std=std)
    model = SimpleNamespace(inputs=[object()], outputs=[object()])

    def fake_get_described_procs(tensor_descrs: Any, **kwargs: Any):
        return [op] if tensor_descrs is model.inputs else []

    monkeypatch.setattr(
        "bioimageio.core.proc_setup._get_described_procs", fake_get_described_procs
    )

    pre, _ = setup_pre_and_postprocessing(
        model=model,  # pyright: ignore[reportArgumentType]
        dataset_for_initial_statistics=_iter_raises_if_consumed(),
    )

    sample = Sample(
        members={
            tid: Tensor(
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                dims=("batch", "x"),
            )
        },
        stat={},
        id=None,
    )
    for proc in pre:
        proc(sample)

    np.testing.assert_allclose(
        sample.members[tid].to_numpy(),
        np.asarray([[-1.224744, 0.0, 1.224744]], dtype=np.float32),
        atol=1e-5,
    )


def test_postprocessing_sample_stats_are_computed_after_preceding_ops(
    monkeypatch: pytest.MonkeyPatch,
):
    tid = MemberId("output")
    clip = Clip(input=tid, output=tid, min=0.0, max=1.0)
    op = ZeroMeanUnitVariance(
        input=tid,
        output=tid,
        mean=SampleMean(member_id=tid, preceding_ops=(clip,)),
        std=SampleStd(member_id=tid, preceding_ops=(clip,)),
    )
    model = SimpleNamespace(inputs=[object()], outputs=[object()])

    def fake_get_described_procs(tensor_descrs: Any, **kwargs: Any):
        return [] if tensor_descrs is model.inputs else [clip, op]

    monkeypatch.setattr(
        "bioimageio.core.proc_setup._get_described_procs", fake_get_described_procs
    )
    pre, post = setup_pre_and_postprocessing(
        model=model,  # pyright: ignore[reportArgumentType]
        dataset_for_initial_statistics=_iter_raises_if_consumed(),
    )
    sample = Sample(
        members={tid: Tensor(np.asarray([[0.0, 1.0, 10.0]]), dims=("batch", "x"))},
        stat={},
        id=None,
    )
    for proc in (*pre, *post):
        proc(sample)

    assert [type(proc) for proc in post] == [Clip, UpdateStats, ZeroMeanUnitVariance]
    np.testing.assert_allclose(
        sample.members[tid].to_numpy(),
        np.asarray([[-(2.0**0.5), 2.0**-0.5, 2.0**-0.5]]),
        atol=1e-5,
    )


def test_fixed_sample_stats_are_not_computed_on_the_fly(
    monkeypatch: pytest.MonkeyPatch,
):
    tid = MemberId("input")
    mean = SampleMean(member_id=tid)
    std = SampleStd(member_id=tid)
    op = ZeroMeanUnitVariance(input=tid, output=tid, mean=mean, std=std)

    model = SimpleNamespace(inputs=[object()], outputs=[object()])

    def fake_get_described_procs(tensor_descrs: Any, **kwargs: Any):
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

    def fake_get_described_procs(tensor_descrs: Any, **kwargs: Any):
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
