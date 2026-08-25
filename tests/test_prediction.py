from collections.abc import Mapping
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from typing_extensions import assert_never

from bioimageio.core import (
    AxisId,
    MemberId,
    PredictionPipeline,
    Sample,
    Tensor,
    create_prediction_pipeline,
    load_model,
    predict,
)
from bioimageio.core.digest_spec import (
    get_test_input_sample,
    get_test_output_sample,
    transpose_sample_for_model,
)
from bioimageio.spec import AnyModelDescr


def _assert_equal_samples(expected: Sample, actual: Sample):
    assert expected.id == actual.id
    assert expected.members == actual.members


class Prep(NamedTuple):
    model: AnyModelDescr
    prediction_pipeline: PredictionPipeline
    input_sample: Sample
    output_sample: Sample


@pytest.fixture(scope="module")
def prep(any_model: str):
    model = load_model(any_model, perform_io_checks=False)
    input_sample = get_test_input_sample(model)
    output_sample = get_test_output_sample(model)
    return Prep(model, create_prediction_pipeline(model), input_sample, output_sample)


def test_predict_with_pipeline(prep: Prep):
    out = predict(
        model=prep.prediction_pipeline,
        inputs=prep.input_sample,
    )
    _assert_equal_samples(out, prep.output_sample)


@pytest.mark.parametrize("tensor_input", ["numpy", "xarray"])
def test_predict_with_model_description(
    tensor_input: Literal["numpy", "xarray"], prep: Prep
):
    if tensor_input == "xarray":
        ipt = {m: t.data for m, t in prep.input_sample.members.items()}
        assert all(isinstance(v, xr.DataArray) for v in ipt.values())
    elif tensor_input == "numpy":
        ipt = {m: t.data.data for m, t in prep.input_sample.members.items()}
        assert all(isinstance(v, np.ndarray) for v in ipt.values())
    else:
        assert_never(tensor_input)

    out = predict(
        model=prep.model,
        inputs=ipt,
        sample_id=prep.input_sample.id,
        skip_preprocessing=False,
        skip_postprocessing=False,
    )
    _assert_equal_samples(out, prep.output_sample)


@pytest.mark.parametrize("with_procs", [True, False])
def test_predict_with_blocking(with_procs: bool, prep: Prep):
    try:
        out = predict(
            model=prep.prediction_pipeline,
            inputs=prep.input_sample,
            blocksize_parameter=3,
            sample_id=prep.input_sample.id,
            skip_preprocessing=with_procs,
            skip_postprocessing=with_procs,
        )
    except NotImplementedError as e:
        pytest.skip(str(e))

    if with_procs:
        _assert_equal_samples(out, prep.output_sample)
    else:
        assert isinstance(out, Sample)


def test_predict_with_fixed_blocking(prep: Prep):
    block_along = list(prep.input_sample.members)
    input_block_shape: Mapping[MemberId, Mapping[AxisId, int]] = {
        ba: {
            "x": min(  # pyright: ignore[reportAssignmentType]
                128, prep.input_sample.members[ba].tagged_shape[AxisId("x")]
            ),
            AxisId("y"): min(
                128, prep.input_sample.members[ba].tagged_shape[AxisId("y")]
            ),
        }
        for ba in block_along
    }
    try:
        out = predict(
            model=prep.prediction_pipeline,
            inputs=prep.input_sample,
            input_block_shape=input_block_shape,
            sample_id=prep.input_sample.id,
        )
    except NotImplementedError as e:
        pytest.skip(str(e))

    _assert_equal_samples(out, prep.output_sample)


def test_predict_save_output(prep: Prep, tmp_path: Path):
    save_path = tmp_path / "{member_id}_{sample_id}.tiff"
    out = predict(
        model=prep.prediction_pipeline,
        inputs=prep.input_sample,
        save_output_path=save_path,
    )
    _assert_equal_samples(out, prep.output_sample)
    assert save_path.parent.exists()


def test_predict_z_as_batch(unet2d_nuclei_broad_model: str):
    model = load_model(
        unet2d_nuclei_broad_model, perform_io_checks=False, format_version="latest"
    )
    assert [a.type for a in model.inputs[0].axes] == [
        "batch",
        "channel",
        "space",
        "space",
    ], "expected 2d model"

    input_sample = get_test_input_sample(model)

    data = input_sample.members[MemberId("raw")].to_numpy()[:, :, None]
    data = np.concatenate([data, data], axis=2)  # add a second z-slice
    input_sample = Sample(
        id=input_sample.id,
        members={
            MemberId("raw"): Tensor(data, dims=["batch", "channel", "z", "y", "x"])
        },
        stat=input_sample.stat,
    )
    input_sample = transpose_sample_for_model(input_sample, model)
    out = predict(model=unet2d_nuclei_broad_model, inputs=input_sample)
    out = out.unstack_batch_multi_index()
    pred = out.members[MemberId("probability")]
    assert "z" in pred.dims
    assert pred.tagged_shape[AxisId("batch")] == 1, "expected 1 batch slice in output"
    assert pred.to_numpy().shape[0] == 1, "expected 1 batch slice in output"
    assert pred.tagged_shape[AxisId("z")] == 2, "expected 2 z-slices in output"
    assert pred.to_numpy().shape[1] == 2, "expected 2 z-slices in output"
    assert pred.tagged_shape[AxisId("channel")] == 1, (
        "expected 1 channel slice in output"
    )
    assert pred.to_numpy().shape[2] == 1, "expected 1 channel slice in output"


def test_transpose_sample_for_model_stacks_extra_z_to_batch(
    unet2d_nuclei_broad_model: str,
):
    model = load_model(
        unet2d_nuclei_broad_model, perform_io_checks=False, format_version="latest"
    )
    input_sample = get_test_input_sample(model)

    data = input_sample.members[MemberId("raw")].to_numpy()[:, :, None]
    data = np.concatenate([data, data], axis=2)
    sample_with_z = Sample(
        id=input_sample.id,
        members={
            MemberId("raw"): Tensor(data, dims=["batch", "channel", "z", "y", "x"])
        },
        stat=input_sample.stat,
    )

    transposed = transpose_sample_for_model(sample_with_z, model)
    raw = transposed.members[MemberId("raw")]
    assert raw.dims == (AxisId("batch"), AxisId("channel"), AxisId("y"), AxisId("x"))
    assert raw.sizes[AxisId("batch")] == 2
    assert isinstance(raw.data.indexes[AxisId("batch")], pd.MultiIndex)
