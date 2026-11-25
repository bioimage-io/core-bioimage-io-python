from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from bioimageio.core._ops_cellpose import CellposeFlowDynamics
from bioimageio.core._ops_stardist import (
    StardistPostprocessing2D,
    StardistPostprocessing3D,
)
from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId
from bioimageio.core.io import load_stat, save_stat, serialize_stat
from bioimageio.core.proc_ops import (
    AddKnownDatasetStats,
    Binarize,
    Clip,
    CustomProcessing,
    EnsureDtype,
    FixedZeroMeanUnitVariance,
    ScaleLinear,
    ScaleMeanVariance,
    ScaleRange,
    Sigmoid,
    Softmax,
    UpdateStats,
    ZeroMeanUnitVariance,
    deserialize_processing,
    serialize_processing,
)
from bioimageio.core.sample import Sample
from bioimageio.core.stat_calculators import StatsCalculator, compute_dataset_measures
from bioimageio.core.stat_measures import (
    DatasetMean,
    SampleMean,
    SampleQuantile,
    SampleStd,
    Stat,
)
from bioimageio.core.tensor import Array, Tensor

ArrayBackend = Literal["numpy", "dask"]
ARRAY_BACKENDS = pytest.mark.parametrize("backend", ["numpy", "dask"])


def test_serialized_measure_replays_preprocessing_with_measure_dependencies():
    member = MemberId("input")
    prior = Clip(input=member, output=member, min=0.0, max=10.0)
    percentiles = Clip(
        input=member,
        output=member,
        min=SampleQuantile(
            member_id=member,
            q=0.25,
            axes=(AxisId("x"),),
            method="linear",
            preceding_ops=(prior,),
        ),
        max=SampleQuantile(
            member_id=member,
            q=0.75,
            axes=(AxisId("x"),),
            method="linear",
            preceding_ops=(prior,),
        ),
    )
    measure = DatasetMean(member_id=member, preceding_ops=(prior, percentiles))
    loaded = load_stat(serialize_stat({measure: 1.0}))
    assert set(loaded) == {measure}
    loaded_measure = next(iter(loaded))
    assert isinstance(loaded_measure, DatasetMean)
    sample = Sample(
        id=None,
        members={member: Tensor(np.asarray([[0.0, 1.0, 100.0]]), dims=("batch", "x"))},
        stat={},
    )

    result = compute_dataset_measures([loaded_measure], [sample])[loaded_measure]
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.to_numpy(), 7.0 / 3.0)


def test_serialized_measure_preserves_reference_member_dependencies():
    member, ref = MemberId("input"), MemberId("reference")
    input_clip = Clip(input=member, output=member, min=0.0, max=10.0)
    ref_clip = Clip(input=ref, output=ref, min=0.0, max=2.0)
    axes = (AxisId("x"),)
    op = ScaleMeanVariance(
        input=member,
        output=member,
        reference_tensor=ref,
        axes=axes,
        _mean=SampleMean(member_id=member, axes=axes, preceding_ops=(input_clip,)),
        _std=SampleStd(member_id=member, axes=axes, preceding_ops=(input_clip,)),
        _ref_mean=SampleMean(member_id=ref, axes=axes, preceding_ops=(ref_clip,)),
        _ref_std=SampleStd(member_id=ref, axes=axes, preceding_ops=(ref_clip,)),
    )
    measure = DatasetMean(member_id=member, preceding_ops=(input_clip, op))
    loaded = load_stat(serialize_stat({measure: 1.0}))
    assert set(loaded) == {measure}
    loaded_measure = next(iter(loaded))
    assert isinstance(loaded_measure, DatasetMean)
    sample = Sample(
        id=None,
        members={
            member: Tensor(np.asarray([[0.0, 2.0, 100.0]]), dims=("batch", "x")),
            ref: Tensor(np.asarray([[0.0, 5.0, 10.0]]), dims=("batch", "x")),
        },
        stat={},
    )

    result = compute_dataset_measures([loaded_measure], [sample])[loaded_measure]
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.to_numpy(), 4.0 / 3.0)


def test_serialized_processing_requires_complete_tagged_fields():
    member = MemberId("input")
    op = Clip(
        input=member,
        output=member,
        min=SampleQuantile(member_id=member, q=0.25),
        max=10.0,
    )
    data = serialize_processing(op)
    _ = data.pop("min")
    with pytest.raises(ValueError, match="missing.*min"):
        _ = deserialize_processing(data)

    with pytest.raises(ValidationError, match="union_tag_invalid"):
        _ = deserialize_processing({"type": "unknown"})
    with pytest.raises(ValidationError, match="union_tag_not_found"):
        _ = deserialize_processing({"input": "input"})

    range_data = serialize_processing(
        ScaleRange(
            member,
            member,
            _lower=SampleQuantile(member_id=member, q=0.25),
            _upper=SampleQuantile(member_id=member, q=0.75),
        )
    )
    _ = range_data.pop("_upper")
    with pytest.raises(ValueError, match="missing.*_upper"):
        _ = deserialize_processing(range_data)


def test_serialized_processing_round_trips_all_supported_types():
    member, output = MemberId("input"), MemberId("output")
    axis = AxisId("channel")
    lower = SampleQuantile(member_id=member, q=0.25)
    upper = SampleQuantile(member_id=member, q=0.75)
    ops = (
        Binarize(member, output, threshold=(0.2, 0.8), axis=axis),
        Clip(member, output, min=lower, max=upper),
        CellposeFlowDynamics(labels_id=output, do_3D=True, output_dtype="uint32"),
        EnsureDtype(member, output, dtype="float32"),
        FixedZeroMeanUnitVariance(member, output, mean=2.0, std=3.0),
        FixedZeroMeanUnitVariance(
            member, output, mean=(1.0, 2.0), std=(3.0, 4.0), axis=axis
        ),
        ScaleLinear(member, output, gain=2.0, offset=3.0),
        ScaleLinear(member, output, gain=(1.0, 2.0), offset=3.0, axis=axis),
        ScaleMeanVariance(member, output),
        ScaleRange(member, output),
        ScaleRange(member, output, _lower=lower, _upper=upper),
        Sigmoid(member, output),
        StardistPostprocessing2D(member, output, (2, 3), 0.5, 0.4, ((1, 2), (3, 4)), 8),
        StardistPostprocessing3D(
            member,
            output,
            (1, 2, 3),
            0.5,
            0.4,
            ((1, 2), (3, 4), (5, 6)),
            8,
            (1.0, 2.0, 3.0),
            overlap_label=7,
        ),
        Softmax(member, output, axis=axis),
        ZeroMeanUnitVariance(
            member, output, SampleMean(member_id=member), SampleStd(member_id=member)
        ),
    )
    for op in ops:
        payload = json.loads(json.dumps(serialize_processing(op)))
        restored = deserialize_processing(payload)
        assert type(restored) is type(op)
        assert restored == op
        assert hash(restored) == hash(op)


def test_custom_processing_cannot_be_serialized():
    member = MemberId("input")
    op = CustomProcessing(member, member, custom_factory=lambda: lambda x: x, kwargs={})
    with pytest.raises(TypeError, match="cannot be serialized"):
        _ = serialize_processing(op)
    with pytest.raises(PydanticSerializationError, match="cannot be serialized"):
        _ = DatasetMean(member_id=member, preceding_ops=(op,)).model_dump(mode="json")


def test_serialized_statistics_operators_preserve_configuration_not_calculator_state():
    member = MemberId("input")
    measure = DatasetMean(member_id=member)
    sample = Sample(
        id=None, members={member: Tensor(np.asarray([1.0, 3.0]), dims=("x",))}, stat={}
    )
    known = AddKnownDatasetStats({measure: Tensor(np.asarray([2.0, 4.0]), dims=("x",))})
    known(sample)
    calculator = StatsCalculator([measure], dataset_fits_in_memory=True)
    updater = UpdateStats(calculator, keep_updating_initial_dataset_stats=True)
    before_update = serialize_processing(updater)
    updater(sample)
    assert calculator.sample_count == 1
    known_data = json.loads(json.dumps(serialize_processing(known)))
    updater_data = json.loads(json.dumps(serialize_processing(updater)))
    assert updater_data == before_update
    assert updater_data["stats_calculator"] == {
        "measures": [measure.model_dump(mode="json")],
        "dataset_fits_in_memory": True,
    }

    restored_known = deserialize_processing(known_data)
    restored_updater = deserialize_processing(updater_data)
    assert isinstance(restored_known, AddKnownDatasetStats)
    known_value = restored_known.dataset_stats[measure]
    assert isinstance(known_value, Tensor)
    np.testing.assert_array_equal(known_value.to_numpy(), [2.0, 4.0])
    assert isinstance(restored_updater, UpdateStats)
    assert restored_updater == updater
    assert restored_updater.stats_calculator.measures == (measure,)
    assert restored_updater.stats_calculator.dataset_fits_in_memory
    assert serialize_processing(restored_updater) == updater_data
    with pytest.raises(RuntimeError, match="reference and cannot run"):
        restored_updater(sample)
    assert sample.applied_ops == (known, updater)


def test_serialized_statistics_operators_reject_invalid_payloads():
    member = MemberId("input")
    updater_data = serialize_processing(UpdateStats(StatsCalculator([])))
    updater_data["stats_calculator"]["unexpected_state"] = 1
    with pytest.raises(ValidationError, match="unexpected_state"):
        _ = deserialize_processing(updater_data)

    sample_measure = SampleMean(member_id=member)
    with pytest.raises(ValueError, match="requires dataset measures"):
        _ = deserialize_processing(
            {
                "type": "AddKnownDatasetStats",
                "dataset_stats": serialize_stat({sample_measure: 1.0}),
            }
        )


def test_unregistered_operator_subclass_cannot_be_serialized():
    class CustomSigmoid(Sigmoid):
        pass

    member = MemberId("input")
    with pytest.raises(TypeError, match="CustomSigmoid cannot be serialized"):
        _ = serialize_processing(CustomSigmoid(member, member))


def test_serialized_measure_rejects_obsolete_applied_ops_field():
    with pytest.raises(ValidationError, match="applied_ops"):
        _ = DatasetMean.model_validate({"member_id": "input", "applied_ops": []})


@pytest.mark.parametrize("kind", ["range", "normalize"])
def test_serialized_measure_preserves_processing_measure_fields(kind: str):
    member = MemberId("input")
    prior = Clip(input=member, output=member, min=0.0, max=10.0)
    axes = (AxisId("x"),)
    if kind == "range":
        op = ScaleRange(
            input=member,
            output=member,
            _lower=SampleQuantile(
                member_id=member, q=0.25, axes=axes, preceding_ops=(prior,)
            ),
            _upper=SampleQuantile(
                member_id=member, q=0.75, axes=axes, preceding_ops=(prior,)
            ),
        )
    else:
        op = ZeroMeanUnitVariance(
            input=member,
            output=member,
            mean=SampleMean(member_id=member, axes=axes, preceding_ops=(prior,)),
            std=SampleStd(member_id=member, axes=axes, preceding_ops=(prior,)),
        )
    measure = DatasetMean(member_id=member, preceding_ops=(prior, op))
    assert set(load_stat(serialize_stat({measure: 1.0}))) == {measure}


def _array(backend: ArrayBackend, data: NDArray[Any]) -> Array:
    return (
        cast(da.Array, da.from_array(data, chunks="auto"))
        if backend == "dask"
        else data
    )


def _make_stat_with_tensor_value(backend: ArrayBackend) -> Stat:
    member_id = MemberId("test_tensor")
    mean_measure = DatasetMean(
        member_id=member_id,
        axes=(AxisId("batch"),),
        preceding_ops=(Clip(input=member_id, output=member_id, min=0, max=1),),
    )
    quantile_measure = SampleQuantile(member_id=member_id, q=0.5)

    tensor_value = Tensor(
        _array(backend, np.arange(6, dtype=np.float32).reshape(2, 3)),
        dims=(AxisId("x"), AxisId("y")),
    )
    float_value = 3.14

    return {mean_measure: tensor_value, quantile_measure: float_value}


@ARRAY_BACKENDS
def test_serialize_stat_with_tensor_value(backend: ArrayBackend):
    stat = _make_stat_with_tensor_value(backend)

    serialized = serialize_stat(stat)
    # should be plain JSON-compatible data, e.g. round-trippable through json
    json_str = json.dumps(serialized)
    assert json.loads(json_str) == serialized

    loaded = load_stat(serialized)
    assert loaded.keys() == stat.keys()
    for measure, expected_value in stat.items():
        actual_value = loaded[measure]
        if isinstance(expected_value, Tensor):
            assert isinstance(actual_value, Tensor)
            xr.testing.assert_allclose(expected_value.data, actual_value.data)
        else:
            assert actual_value == expected_value


@ARRAY_BACKENDS
def test_save_and_load_stat_with_tensor_value(backend: ArrayBackend, tmp_path: Path):
    stat = _make_stat_with_tensor_value(backend)
    path = tmp_path / "stat.json"

    save_stat(stat, path)
    loaded = load_stat(path)

    assert loaded.keys() == stat.keys()
    for measure, expected_value in stat.items():
        actual_value = loaded[measure]
        if isinstance(expected_value, Tensor):
            assert isinstance(actual_value, Tensor)
            xr.testing.assert_allclose(expected_value.data, actual_value.data)
        else:
            assert actual_value == expected_value
