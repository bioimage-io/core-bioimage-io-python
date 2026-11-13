from __future__ import annotations

from itertools import product
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.base import compute  # pyright: ignore[reportUnknownVariableType]
from dask.callbacks import Callback
from dask.delayed import delayed  # pyright: ignore[reportUnknownVariableType]
from numpy.typing import NDArray
from xarray.testing import assert_allclose  # pyright: ignore[reportUnknownVariableType]

from bioimageio.core import stat_calculators
from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId
from bioimageio.core.proc_ops import (
    AddKnownDatasetStats,
    Clip,
    ScaleLinear,
    ScaleRange,
    UpdateStats,
)
from bioimageio.core.sample import Sample
from bioimageio.core.stat_calculators import (
    CrickQuantilesCalculator,
    InMemoryDatasetQuantilesCalculator,
    MeanCalculator,
    MeanQuantilesCalculator,
    MeanVarStdCalculator,
    NaiveSampleMeasureCalculator,
    SampleQuantilesCalculator,
    StatsCalculator,
    compute_dataset_measures,
    compute_measures,
    compute_sample_measures,
    get_measure_calculators,
)
from bioimageio.core.stat_measures import (
    DatasetMean,
    DatasetMeasure,
    DatasetQuantile,
    DatasetStd,
    DatasetVar,
    Measure,
    MeasureValue,
    SampleMean,
    SampleMeasure,
    SampleQuantile,
    SampleStd,
    SampleVar,
)
from bioimageio.core.tensor import Tensor

_DIMS = tuple(map(AxisId, ("batch", "channel", "y", "x")))
_MEMBER = MemberId("input")
_SAMPLE_TYPES = (SampleMean, SampleVar, SampleStd, SampleQuantile)
_DATASET_TYPES = (DatasetMean, DatasetVar, DatasetStd, DatasetQuantile)


def _sample(data: NDArray[np.float64], lazy: bool) -> Sample:
    tensor = Tensor(data, dims=_DIMS)
    if lazy:
        tensor = Tensor.from_xarray(
            cast(
                xr.DataArray,
                tensor.data.chunk(dict(zip(_DIMS, (1, 1, 2, 3)))),
            )
        )
    return Sample(id=None, members={_MEMBER: tensor}, stat={})


def _lazy_array(value: MeasureValue) -> da.Array:
    assert isinstance(value, Tensor)
    assert isinstance(value.data.data, da.Array)
    return value.data.data


def _no_execution(key: object, *args: object) -> None:
    pytest.fail(f"Statistics executed a task during graph construction: {key}")


@pytest.fixture(params=[MeanQuantilesCalculator, CrickQuantilesCalculator])
def dataset_quantiles_backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> bool:
    use_crick = request.param is CrickQuantilesCalculator
    if use_crick:
        _ = pytest.importorskip("crick")
    monkeypatch.setattr(stat_calculators, "DatasetQuantilesCalculator", request.param)
    return use_crick


def _expected_dataset_quantile(
    arrays: list[NDArray[np.float64]],
    axes: tuple[AxisId, ...] | None,
    q: float,
    use_crick: bool,
) -> NDArray[np.float64]:
    reduced = tuple(i for i, d in enumerate(_DIMS) if axes is None or d in axes)
    kept = tuple(i for i in range(len(_DIMS)) if i not in reduced)
    shape = tuple(arrays[0].shape[i] for i in kept)
    if not use_crick:
        counts = [np.prod([a.shape[i] for i in reduced]) for a in arrays]
        return np.asarray(
            sum(n * np.quantile(a, q, axis=reduced) for n, a in zip(counts, arrays))
            / sum(counts)
        )

    crick: Any = pytest.importorskip("crick")
    result = np.empty(shape)
    for idx in product(*map(range, shape)):
        digest = crick.TDigest()
        for array in arrays:
            digest.update(array.transpose((*kept, *reduced))[idx])
        result[idx] = digest.quantile(q)
    return result


def create_random_dataset(tid: MemberId, axes: tuple[AxisId, ...]):
    n = 3
    sizes = list(range(n, len(axes) + n))
    data = np.asarray(np.random.rand(*sizes))
    ds = [
        Sample(members={tid: Tensor(data[i : i + 1], dims=axes)}, stat={}, id=None)
        for i in range(n)
    ]
    return Tensor(data, dims=axes), ds


@pytest.mark.parametrize(
    "axes",
    [
        (AxisId("x"), AxisId("y")),
        (AxisId("channel"), AxisId("y")),
    ],
)
def test_sample_mean_var_std_calculator(axes: tuple[AxisId, ...] | None):
    tid = MemberId("tensor")
    d_axes = tuple(map(AxisId, ("batch", "channel", "x", "y")))
    data, ds = create_random_dataset(tid, d_axes)
    expected_mean = data[0:1].mean(axes)
    expected_var = data[0:1].var(axes)
    expected_std = data[0:1].std(axes)

    calc = MeanVarStdCalculator(tid, axes=axes)

    actual = calc.compute(ds[0])
    actual_mean = actual[SampleMean(member_id=tid, axes=axes)]
    actual_var = actual[SampleVar(member_id=tid, axes=axes)]
    actual_std = actual[SampleStd(member_id=tid, axes=axes)]

    assert_allclose(
        actual_mean if isinstance(actual_mean, (int, float)) else actual_mean.data,
        expected_mean.data,
    )
    assert_allclose(
        actual_var if isinstance(actual_var, (int, float)) else actual_var.data,
        expected_var.data,
    )
    assert_allclose(
        actual_std if isinstance(actual_std, (int, float)) else actual_std.data,
        expected_std.data,
    )


@pytest.mark.parametrize(
    "axes",
    [
        None,
        (AxisId("batch"), AxisId("channel"), AxisId("x"), AxisId("y")),
    ],
)
def test_dataset_mean_var_std_calculator(axes: tuple[AxisId, ...] | None):
    tid = MemberId("tensor")
    d_axes = tuple(map(AxisId, ("batch", "channel", "x", "y")))
    data, ds = create_random_dataset(tid, d_axes)
    expected_mean = data.mean(axes)
    expected_var = data.var(axes)
    expected_std = data.std(axes)

    calc = MeanVarStdCalculator(tid, axes=axes)
    for s in ds:
        calc.update(s)

    actual = calc.finalize()
    actual_mean = actual[DatasetMean(member_id=tid, axes=axes)]
    actual_var = actual[DatasetVar(member_id=tid, axes=axes)]
    actual_std = actual[DatasetStd(member_id=tid, axes=axes)]

    assert_allclose(
        actual_mean if isinstance(actual_mean, (int, float)) else actual_mean.data,
        expected_mean.data,
    )
    assert_allclose(
        actual_var if isinstance(actual_var, (int, float)) else actual_var.data,
        expected_var.data,
    )
    assert_allclose(
        actual_std if isinstance(actual_std, (int, float)) else actual_std.data,
        expected_std.data,
    )


@pytest.mark.parametrize("lazy", [False, True], ids=["numpy", "dask"])
@pytest.mark.parametrize("axes", [None, (AxisId("y"), AxisId("x"))])
@pytest.mark.parametrize("measure_type", _SAMPLE_TYPES)
@pytest.mark.parametrize("direct", [False, True], ids=["calculator", "measure"])
def test_sample_measure_evaluation_mode(
    lazy: bool,
    axes: tuple[AxisId, ...] | None,
    measure_type: type[SampleMeasure],
    direct: bool,
):
    data = np.arange(48, dtype="float64").reshape(1, 2, 4, 6)
    sample = _sample(data, lazy)
    kwargs: dict[str, Any] = {"member_id": _MEMBER, "axes": axes}
    if measure_type is SampleQuantile:
        kwargs["q"] = 0.25
    measure = measure_type(**kwargs)
    with Callback(pretask=_no_execution):
        if direct:
            result = measure.compute(sample)
        else:
            values = compute_sample_measures([measure], sample)
            for value in values.values():
                _ = _lazy_array(value)
            result = values[measure]

    reduced = None if axes is None else tuple(_DIMS.index(d) for d in axes)
    expected = (
        np.quantile(data, 0.25, axis=reduced)
        if isinstance(measure, SampleQuantile)
        else getattr(np, measure.name)(data, axis=reduced)
    )
    assert isinstance(result, Tensor)
    assert result.dims == tuple(d for d in _DIMS if axes is not None and d not in axes)
    if direct and not lazy:
        assert isinstance(result.data.data, np.ndarray)
        actual = result.to_numpy()
    else:
        actual = cast(NDArray[Any], _lazy_array(result).compute())
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("lazy", [False, True], ids=["numpy", "dask"])
@pytest.mark.parametrize("axes", [None, (AxisId("batch"), AxisId("y"), AxisId("x"))])
@pytest.mark.parametrize("measure_type", _DATASET_TYPES)
@pytest.mark.parametrize("combined", [False, True], ids=["dataset", "combined"])
def test_dataset_measures_are_lazy(
    lazy: bool,
    axes: tuple[AxisId, ...] | None,
    measure_type: type[DatasetMeasure],
    combined: bool,
    dataset_quantiles_backend: bool,
):
    arrays = [
        np.arange(48, dtype="float64").reshape(1, 2, 4, 6),
        np.arange(96, dtype="float64").reshape(2, 2, 4, 6) + 100,
    ]
    dataset = [_sample(a, lazy) for a in arrays]
    kwargs: dict[str, Any] = {"member_id": _MEMBER, "axes": axes}
    if measure_type is DatasetQuantile:
        kwargs["q"] = 0.25
    measure = measure_type(**kwargs)
    with Callback(pretask=_no_execution):
        values = (
            compute_measures([measure], iter(dataset))
            if combined
            else compute_dataset_measures([measure], iter(dataset))
        )
        for value in values.values():
            _ = _lazy_array(value)

    reduced = None if axes is None else tuple(_DIMS.index(d) for d in axes)
    expected = (
        _expected_dataset_quantile(arrays, axes, 0.25, dataset_quantiles_backend)
        if isinstance(measure, DatasetQuantile)
        else getattr(np, measure.name)(np.concatenate(arrays), axis=reduced)
    )
    result = values[measure]
    assert isinstance(result, Tensor)
    assert result.dims == tuple(d for d in _DIMS if axes is not None and d not in axes)
    np.testing.assert_allclose(
        cast(NDArray[Any], _lazy_array(result).compute()), expected
    )


@pytest.mark.parametrize("lazy", [False, True], ids=["numpy", "dask"])
@pytest.mark.parametrize(
    "axes",
    [None, (AxisId("batch"), AxisId("y"), AxisId("x")), (AxisId("batch"), AxisId("x"))],
)
def test_dataset_statistics_are_snapshots(
    lazy: bool,
    axes: tuple[AxisId, ...] | None,
    dataset_quantiles_backend: bool,
):
    arrays = [
        np.arange(48, dtype="float64").reshape(1, 2, 4, 6),
        np.arange(96, dtype="float64").reshape(2, 2, 4, 6) + 1000,
    ]
    measures = [
        DatasetMean(member_id=_MEMBER, axes=axes),
        DatasetVar(member_id=_MEMBER, axes=axes),
        DatasetStd(member_id=_MEMBER, axes=axes),
        DatasetQuantile(member_id=_MEMBER, axes=axes, q=0.25),
        DatasetQuantile(member_id=_MEMBER, axes=axes, q=0.75),
    ]
    calc = StatsCalculator(measures)
    with Callback(pretask=_no_execution):
        assert calc.finalize() == {}
        calc.update(_sample(arrays[0], lazy))
        first = calc.finalize()
        second_sample = _sample(arrays[1], lazy)
        second_sample.members[_MEMBER] = second_sample.members[_MEMBER].transpose(
            _DIMS[::-1]
        )
        calc.update(second_sample)
        second = calc.finalize()

    reduced = None if axes is None else tuple(_DIMS.index(d) for d in axes)
    # Compute the new snapshot first, then recompute both to detect mutated digests.
    for values, inputs in [(second, arrays), (first, arrays[:1]), (second, arrays)]:
        actual = cast(
            tuple[NDArray[Any], ...],
            compute(
                *(_lazy_array(values[m]) for m in measures), scheduler="synchronous"
            ),
        )
        for measure, value in zip(measures, actual):
            expected = (
                _expected_dataset_quantile(
                    inputs, axes, measure.q, dataset_quantiles_backend
                )
                if isinstance(measure, DatasetQuantile)
                else getattr(np, measure.name)(np.concatenate(inputs), axis=reduced)
            )
            np.testing.assert_allclose(value, expected)


@pytest.mark.parametrize("dataset", [False, True])
def test_mean_variance_std_share_one_calculator(dataset: bool):
    types = _DATASET_TYPES[:3] if dataset else _SAMPLE_TYPES[:3]
    for requested in [types[1:2], types[2:3], types]:
        measures = [cls(member_id=_MEMBER, axes=None) for cls in requested]
        sample_calcs, dataset_calcs = get_measure_calculators(measures)
        selected, other = (
            (dataset_calcs, sample_calcs) if dataset else (sample_calcs, dataset_calcs)
        )
        assert not other
        assert len(selected) == 1
        assert isinstance(selected[0], MeanVarStdCalculator)


@pytest.mark.parametrize(
    "measure",
    [
        SampleMean(member_id=_MEMBER),
        SampleVar(member_id=_MEMBER),
        SampleStd(member_id=_MEMBER),
        SampleQuantile(member_id=_MEMBER, q=0.25),
        DatasetMean(member_id=_MEMBER),
        DatasetVar(member_id=_MEMBER),
        DatasetStd(member_id=_MEMBER),
        DatasetQuantile(member_id=_MEMBER, q=0.25),
    ],
)
def test_measure_calculators_derive_preprocessing_provenance(
    measure: Measure,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        stat_calculators, "DatasetQuantilesCalculator", MeanQuantilesCalculator
    )
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    measure = measure.model_copy(update={"preceding_ops": (clip,)})
    data = np.arange(48, dtype="float64").reshape(1, 2, 4, 6)
    sample = _sample(data, lazy=True)
    sample_calculators, dataset_calculators = get_measure_calculators([measure])
    with Callback(pretask=_no_execution):
        values: dict[Measure, MeasureValue] = {}
        for calc in sample_calculators:
            values.update(calc.compute(sample).items())
        for calc in dataset_calculators:
            calc.update(sample)
            values.update(calc.finalize().items())

    raw_measure = measure.model_copy(update={"preceding_ops": ()})
    assert raw_measure in values
    assert all(not key.preceding_ops for key in values)

    clip(sample)
    sample_calculators, dataset_calculators = get_measure_calculators([measure])
    with Callback(pretask=_no_execution):
        values = {}
        for calc in sample_calculators:
            values.update(calc.compute(sample).items())
        for calc in dataset_calculators:
            calc.update(sample)
            values.update(calc.finalize().items())

    data = np.clip(data, 0.0, 1.0)
    expected = (
        np.quantile(data, 0.25)
        if measure.name == "quantile"
        else getattr(np, measure.name)(data)
    )
    actual = cast(NDArray[Any], _lazy_array(values[measure]).compute())
    np.testing.assert_allclose(actual, expected)
    assert all(key.preceding_ops == (clip,) for key in values)


def test_measure_calculators_share_configurations_across_provenance():
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    raw_mean = SampleMean(member_id=_MEMBER)
    clipped_std = SampleStd(member_id=_MEMBER, preceding_ops=(clip,))
    raw_quantile = DatasetQuantile(member_id=_MEMBER, q=0.25)
    clipped_quantile = DatasetQuantile(member_id=_MEMBER, q=0.75, preceding_ops=(clip,))

    sample_calcs, dataset_calcs = get_measure_calculators(
        [raw_mean, clipped_std, raw_quantile, clipped_quantile]
    )
    assert len(sample_calcs) == 1
    assert len(dataset_calcs) == 1
    sample = _sample(np.arange(48, dtype="float64").reshape(1, 2, 4, 6), lazy=True)
    sample_values: dict[SampleMeasure, MeasureValue] = {}
    for calc in sample_calcs:
        sample_values.update(calc.compute(sample).items())
    assert raw_mean in sample_values
    assert SampleStd(member_id=_MEMBER) in sample_values
    assert clipped_std not in sample_values
    clip(sample)
    sample_values = {}
    for calc in sample_calcs:
        sample_values.update(calc.compute(sample).items())
    assert clipped_std in sample_values
    assert raw_mean not in sample_values


@pytest.mark.parametrize(
    "backend",
    [
        MeanQuantilesCalculator,
        CrickQuantilesCalculator,
        InMemoryDatasetQuantilesCalculator,
    ],
)
def test_dataset_quantile_backends_preserve_provenance(
    backend: type[
        MeanQuantilesCalculator
        | CrickQuantilesCalculator
        | InMemoryDatasetQuantilesCalculator
    ],
    monkeypatch: pytest.MonkeyPatch,
):
    if backend is CrickQuantilesCalculator:
        _ = pytest.importorskip("crick")
    elif backend is MeanQuantilesCalculator:
        monkeypatch.setattr(stat_calculators, "DatasetQuantilesCalculator", backend)

    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    measures = [
        DatasetQuantile(member_id=_MEMBER, q=q, preceding_ops=(clip,))
        for q in (0.25, 0.75)
    ]
    _, calculators = get_measure_calculators(
        measures, datast_fits_in_memory=backend is InMemoryDatasetQuantilesCalculator
    )
    assert len(calculators) == 1
    calc = calculators[0]
    assert isinstance(
        calc,
        (
            MeanQuantilesCalculator,
            CrickQuantilesCalculator,
            InMemoryDatasetQuantilesCalculator,
        ),
    )
    sample = _sample(np.linspace(0.0, 1.0, 48).reshape(1, 2, 4, 6), lazy=True)
    clip(sample)
    with Callback(pretask=_no_execution):
        calc.update(sample)
        values = calc.finalize()

    assert set(values) == set(measures)
    lower, upper = (
        float(cast(NDArray[Any], _lazy_array(values[measure]).compute()))
        for measure in measures
    )
    assert lower < upper


@pytest.mark.parametrize(
    "kind", ["mean", "mean_var", "mean_quantile", "crick_quantile", "memory_quantile"]
)
def test_dataset_calculators_reject_mixed_sample_provenance(kind: str):
    if kind == "mean":
        calc = MeanCalculator(_MEMBER, axes=None)
    elif kind == "mean_var":
        calc = MeanVarStdCalculator(_MEMBER, axes=None)
    elif kind == "mean_quantile":
        calc = MeanQuantilesCalculator(_MEMBER, axes=None, qs=[0.5])
    elif kind == "crick_quantile":
        _ = pytest.importorskip("crick")
        calc = CrickQuantilesCalculator(_MEMBER, axes=None, qs=[0.5])
    else:
        calc = InMemoryDatasetQuantilesCalculator(_MEMBER, axes=None, qs=[0.5])

    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    other_clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=2.0)
    missing = Sample(id=None, members={}, stat={}, applied_ops=(other_clip,))
    with pytest.raises(KeyError):
        calc.update(missing)
    assert calc.finalize() == {}

    first = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    second = first.copy()
    clip(first)
    clip(second)
    other_member = MemberId("other")
    second.members[other_member] = second.members[_MEMBER]
    ScaleLinear(input=other_member, output=other_member, gain=2.0)(second)
    AddKnownDatasetStats({})(second)
    calc.update(first)
    calc.update(second)
    before: dict[DatasetMeasure, MeasureValue] = {}
    before.update(calc.finalize().items())
    assert before
    assert all(measure.preceding_ops == (clip,) for measure in before)

    mismatched = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    other_clip(mismatched)
    with pytest.raises(ValueError, match="different applied_ops"):
        calc.update(mismatched)
    after: dict[DatasetMeasure, MeasureValue] = {}
    after.update(calc.finalize().items())
    assert after.keys() == before.keys()
    for measure, value in after.items():
        np.testing.assert_allclose(
            cast(NDArray[Any], _lazy_array(value).compute()),
            cast(NDArray[Any], _lazy_array(before[measure]).compute()),
        )


def test_sample_calculators_use_current_sample_history():
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    sample = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    calculators = (
        MeanCalculator(_MEMBER, axes=None),
        MeanVarStdCalculator(_MEMBER, axes=None),
        SampleQuantilesCalculator(_MEMBER, axes=None, qs=[0.5]),
        NaiveSampleMeasureCalculator(
            _MEMBER, SampleMean(member_id=_MEMBER, preceding_ops=(clip,))
        ),
    )
    for calculator in calculators:
        assert all(not measure.preceding_ops for measure in calculator.compute(sample))
    clip(sample)
    AddKnownDatasetStats({})(sample)
    for calculator in calculators:
        assert all(
            measure.preceding_ops == (clip,) for measure in calculator.compute(sample)
        )

    mean_calc = MeanCalculator(_MEMBER, axes=None)
    assert set(mean_calc.compute_and_update(sample)) == {
        SampleMean(member_id=_MEMBER, preceding_ops=(clip,))
    }
    assert set(mean_calc.finalize()) == {
        DatasetMean(member_id=_MEMBER, preceding_ops=(clip,))
    }


def test_stats_calculator_preserves_provenance_and_uses_processed_samples():
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    raw_mean = DatasetMean(member_id=_MEMBER)
    clipped_mean = raw_mean.model_copy(update={"preceding_ops": (clip,)})
    sample_mean = SampleMean(member_id=_MEMBER, preceding_ops=(clip,))
    calc = StatsCalculator(
        [clipped_mean, sample_mean],
        {raw_mean: 7.0, clipped_mean: 8.0},
    )
    assert calc.finalize() == {raw_mean: 7.0, clipped_mean: 8.0}

    raw = _sample(np.full((1, 1, 2, 2), 4.0), lazy=True)
    processed = raw.copy()
    clip(processed)
    with Callback(pretask=_no_execution):
        actual = calc.update_and_get_all(processed)

    assert set(actual) == {raw_mean, clipped_mean, sample_mean}
    assert actual[raw_mean] == 7.0
    for measure in [clipped_mean, sample_mean]:
        computed = cast(NDArray[Any], _lazy_array(actual[measure]).compute())
        np.testing.assert_allclose(computed, 1.0)
    np.testing.assert_allclose(processed.members[_MEMBER].to_numpy(), 1.0)

    other = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=2.0)
    other(raw)
    with pytest.raises(ValueError, match="preceding_ops"):
        StatsCalculator([clipped_mean]).update(raw)


@pytest.mark.parametrize(
    "method",
    ["update", "update_and_get_all", "skip_update_and_get_all"],
)
def test_unbound_measures_reject_applied_ops(method: str):
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    mean = DatasetMean(member_id=_MEMBER)
    calc = StatsCalculator([mean], {mean: 42.0})
    sample = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    clip(sample)

    with pytest.raises(ValueError, match="preceding_ops"):
        getattr(calc, method)(sample)

    assert calc.sample_count == 0
    assert calc.finalize() == {mean: 42.0}


def test_partial_preprocessing_mismatch_is_rejected():
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    raw_mean = DatasetMean(member_id=_MEMBER)
    clipped_mean = raw_mean.model_copy(update={"preceding_ops": (clip,)})
    calc = StatsCalculator([raw_mean, clipped_mean])
    sample = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    clip(sample)

    with pytest.raises(ValueError, match="preceding_ops"):
        calc.update(sample)
    assert calc.sample_count == 0
    assert calc.finalize() == {}


def test_applied_ops_are_checked_for_each_member():
    other_member = MemberId("other")
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    clipped_mean = SampleMean(member_id=_MEMBER, preceding_ops=(clip,))
    other_mean = SampleMean(member_id=other_member)
    sample = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    sample.members[other_member] = sample.members[_MEMBER]
    clip(sample)

    values = StatsCalculator([clipped_mean, other_mean]).update_and_get_all(sample)
    for measure, expected in [(clipped_mean, 1.0), (other_mean, 3.0)]:
        computed = cast(NDArray[Any], _lazy_array(values[measure]).compute())
        np.testing.assert_allclose(computed, expected)


def test_initial_stats_with_different_provenance_are_not_relabelled():
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    raw_mean = DatasetMean(member_id=_MEMBER)
    clipped_mean = raw_mean.model_copy(update={"preceding_ops": (clip,)})
    calc = StatsCalculator([clipped_mean], {raw_mean: 42.0})

    assert not calc.has_dataset_measures
    assert calc.finalize() == {raw_mean: 42.0}


def test_update_stats_uses_initial_values_with_matching_provenance():
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    mean = DatasetMean(member_id=_MEMBER)
    processed_mean = mean.model_copy(update={"preceding_ops": (clip,)})
    calc = StatsCalculator([processed_mean], {processed_mean: 42.0})
    op = UpdateStats(calc)
    sample = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)

    clip(sample)
    op(sample)

    assert sample.stat == {processed_mean: 42.0}
    assert calc.sample_count == 0


def test_stats_calculator_checks_each_sample_in_an_iterable():
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    measure = DatasetMean(member_id=_MEMBER, preceding_ops=(clip,))
    first = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    second = first.copy()
    clip(first)

    with pytest.raises(ValueError, match="preceding_ops"):
        StatsCalculator([measure]).update(iter([first, second]))


def test_compute_measures_reuses_already_applied_processing():
    scale = ScaleLinear(input=_MEMBER, output=_MEMBER, gain=2.0, offset=0.0)
    mean = DatasetMean(member_id=_MEMBER, preceding_ops=(scale,))
    sample = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    scale(sample)

    value = compute_dataset_measures([mean], [sample])[mean]
    np.testing.assert_allclose(cast(NDArray[Any], _lazy_array(value).compute()), 6.0)
    assert sample.applied_ops == (scale,)


def test_compute_measures_applies_only_missing_processing_and_rejects_mismatch():
    first = ScaleLinear(input=_MEMBER, output=_MEMBER, gain=2.0, offset=0.0)
    second = ScaleLinear(input=_MEMBER, output=_MEMBER, gain=3.0, offset=0.0)
    measure = DatasetMean(member_id=_MEMBER, preceding_ops=(first, second))
    sample = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    first(sample)
    value = compute_dataset_measures([measure], [sample])[measure]
    np.testing.assert_allclose(cast(NDArray[Any], _lazy_array(value).compute()), 18.0)
    assert sample.applied_ops == (first,)

    mismatched = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    second(mismatched)
    with pytest.raises(ValueError, match="preceding_ops"):
        _ = compute_dataset_measures([measure], [mismatched])
    with pytest.raises(ValueError, match="preceding_ops"):
        _ = compute_dataset_measures([DatasetMean(member_id=_MEMBER)], [mismatched])


def test_update_stats_checks_history_even_with_initial_values():
    clip = Clip(input=_MEMBER, output=_MEMBER, min=0.0, max=1.0)
    mean = DatasetMean(member_id=_MEMBER, preceding_ops=(clip,))
    updater = UpdateStats(StatsCalculator([mean], {mean: 42.0}))
    sample = _sample(np.full((1, 1, 2, 2), 3.0), lazy=True)
    with pytest.raises(ValueError, match="preceding_ops"):
        updater(sample)
    assert sample.applied_ops == ()
    assert sample.stat == {}


@pytest.mark.parametrize("scheduler", ["synchronous", "threads"])
@pytest.mark.parametrize("delayed_blocks", [False, True], ids=["arrays", "inference"])
def test_quantiles_are_shared_by_multiple_blocks(
    monkeypatch: pytest.MonkeyPatch, scheduler: str, delayed_blocks: bool
):
    _ = pytest.importorskip("crick")
    calls: list[str] = []
    original_update = stat_calculators._update_crick_digests  # pyright: ignore[reportPrivateUsage]
    original_quantiles = stat_calculators._get_crick_quantiles  # pyright: ignore[reportPrivateUsage]

    def load():
        calls.append("load")
        return np.arange(48, dtype="float64").reshape(1, 2, 4, 6)

    def update(*args: Any):
        calls.append("update")
        return original_update(*args)

    def quantiles(*args: Any):
        calls.append("quantiles")
        return original_quantiles(*args)

    def forward(inputs: list[NDArray[Any]]) -> NDArray[Any]:
        return inputs[0] + 1

    monkeypatch.setattr(stat_calculators, "_update_crick_digests", update)
    monkeypatch.setattr(stat_calculators, "_get_crick_quantiles", quantiles)
    monkeypatch.setattr(
        stat_calculators, "DatasetQuantilesCalculator", CrickQuantilesCalculator
    )
    data = da.from_delayed(delayed(load)(), shape=(1, 2, 4, 6), dtype="float64")
    sample = Sample(id=None, members={_MEMBER: Tensor(data, _DIMS)}, stat={})
    lower_measure = DatasetQuantile(member_id=_MEMBER, axes=None, q=0.25)
    upper_measure = DatasetQuantile(member_id=_MEMBER, axes=None, q=0.75)
    with Callback(pretask=_no_execution):
        UpdateStats(StatsCalculator([lower_measure, upper_measure]))(sample)
        ScaleRange(
            input=_MEMBER,
            output=_MEMBER,
            _lower=lower_measure,
            _upper=upper_measure,
            eps=0,
        )(sample)
        lower = _lazy_array(sample.stat[lower_measure])
        upper = _lazy_array(sample.stat[upper_measure])
        normalized = _lazy_array(sample.members[_MEMBER])
        blocks = [cast(da.Array, normalized[..., i : i + 2]) for i in range(0, 6, 2)]
        outputs = blocks
        if delayed_blocks:
            outputs = [
                cast(
                    da.Array,
                    da.from_delayed(
                        delayed(forward, pure=False)([block]),
                        shape=block.shape,
                        dtype="float64",
                    ),
                )
                for block in blocks
            ]
    assert calls == []

    actual = cast(
        tuple[NDArray[Any], ...], compute(*outputs, lower, upper, scheduler=scheduler)
    )
    expected = (np.arange(48, dtype="float64").reshape(1, 2, 4, 6) - actual[-2]) / (
        actual[-1] - actual[-2]
    )
    if delayed_blocks:
        expected += 1
    for i, block in enumerate(actual[:-2]):
        np.testing.assert_allclose(block, expected[..., 2 * i : 2 * i + 2])
    assert calls.count("load") == 1
    assert calls.count("update") == 1
    assert calls.count("quantiles") == 1


def test_crick_rejects_incompatible_unreduced_dimensions():
    _ = pytest.importorskip("crick")
    calc = CrickQuantilesCalculator(
        _MEMBER, (AxisId("batch"), AxisId("y"), AxisId("x")), [0.5]
    )
    with Callback(pretask=_no_execution):
        calc.update(_sample(np.zeros((1, 2, 4, 6)), True))
        with pytest.raises(ValueError, match="unreduced dimensions"):
            calc.update(_sample(np.zeros((1, 3, 4, 6)), True))
