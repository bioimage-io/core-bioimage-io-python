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
from bioimageio.core.proc_ops import ScaleRange, UpdateStats
from bioimageio.core.sample import Sample
from bioimageio.core.stat_calculators import (
    CrickQuantilesCalculator,
    MeanQuantilesCalculator,
    MeanVarStdCalculator,
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
            cast(xr.DataArray, tensor.data.chunk(dict(zip(_DIMS, (1, 1, 2, 3)))))
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
            lower_quantile=lower_measure,
            upper_quantile=upper_measure,
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
