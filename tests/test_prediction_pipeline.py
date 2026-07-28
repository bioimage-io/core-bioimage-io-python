from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from bioimageio.core import Sample
from bioimageio.core._resource_tests import evaluate_mismatched_elements
from bioimageio.core.common import SupportedWeightsFormat
from bioimageio.spec import load_description
from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr04
from bioimageio.spec.model.v0_5 import ModelDescr


def _alter_sample(sample: Sample, offset: float) -> Sample:
    # add 1 to all values to get a different sample with the same shape and axes
    return Sample(
        id=f"{sample.id}_altered",
        members={m: t + offset for m, t in sample.members.items()},
        stat=sample.stat,
    )


def _test_prediction_pipeline(
    model_package: Path, weights_format: SupportedWeightsFormat
):
    from bioimageio.core._prediction_pipeline import create_prediction_pipeline
    from bioimageio.core.digest_spec import (
        get_test_input_sample,
        get_test_output_sample,
    )

    bio_model = load_description(model_package)
    assert isinstance(bio_model, (ModelDescr, ModelDescr04)), (
        bio_model.validation_summary.format()
    )

    pp = create_prediction_pipeline(
        bioimageio_model=bio_model, weight_format=weights_format, devices=["cpu", "cpu"]
    )

    inputs = get_test_input_sample(bio_model)

    # test in a multi-threaded setting
    multiple_inputs = [inputs, _alter_sample(inputs, offset=100.0)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        multiple_outputs = list(
            executor.map(
                partial(
                    pp.predict_sample_without_blocking,
                    skip_input_padding=True,
                    skip_output_cropping=True,
                ),
                multiple_inputs,
            )
        )

    outputs = multiple_outputs[0]

    expected_outputs = get_test_output_sample(bio_model)
    assert len(outputs.shape) == len(expected_outputs.shape)
    for m in expected_outputs.members:
        out = outputs.members[m]
        assert out is not None
        exp = expected_outputs.members[m]
        mismatched_ppm, msg, error_msg = evaluate_mismatched_elements(
            out, exp, rtol=0.01, atol=0.1, name=m
        )
        if error_msg is not None:
            raise AssertionError(error_msg)
        elif mismatched_ppm > 50_000:
            raise AssertionError(msg)


def test_prediction_pipeline_torch(any_torch_model: Path):
    _test_prediction_pipeline(any_torch_model, "pytorch_state_dict")


def test_prediction_pipeline_torchscript(any_torchscript_model: Path):
    _test_prediction_pipeline(any_torchscript_model, "torchscript")


def test_prediction_pipeline_onnx(any_onnx_model: Path):
    _test_prediction_pipeline(any_onnx_model, "onnx")


def test_prediction_pipeline_tensorflow(any_tensorflow_model: Path):
    _test_prediction_pipeline(any_tensorflow_model, "tensorflow_saved_model_bundle")


def test_prediction_pipeline_keras(any_keras_model: Path):
    _test_prediction_pipeline(any_keras_model, "keras_hdf5")
