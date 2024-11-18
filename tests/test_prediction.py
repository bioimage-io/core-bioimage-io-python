from pathlib import Path
from typing import Literal, Mapping, NamedTuple

import numpy as np
import pytest
import xarray as xr
from typing_extensions import assert_never

from bioimageio.core import (
    AxisId,
    MemberId,
    PredictionPipeline,
    Sample,
    create_prediction_pipeline,
    load_model,
    predict,
)
from bioimageio.core.digest_spec import get_test_inputs, get_test_outputs
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
    input_sample = get_test_inputs(model)
    output_sample = get_test_outputs(model)
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
    save_path = tmp_path / "{member_id}_{sample_id}.h5"
    out = predict(
        model=prep.prediction_pipeline,
        inputs=prep.input_sample,
        save_output_path=save_path,
    )
    _assert_equal_samples(out, prep.output_sample)
    assert save_path.parent.exists()
