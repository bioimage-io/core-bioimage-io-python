from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from bioimageio.core import AxisId, Sample
from bioimageio.core._resource_tests import evaluate_mismatched_elements
from bioimageio.core.common import SupportedWeightsFormat
from bioimageio.spec import load_description
from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr04
from bioimageio.spec.model.v0_5 import ModelDescr


def _alter_sample(sample: Sample, offset: float) -> Sample:
    # add an offset to get a different sample with the same shape and axes
    return Sample(
        id=f"{sample.id}_altered",
        members={m: t + offset for m, t in sample.members.items()},
        stat=sample.stat.copy(),
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

    inputs = get_test_input_sample(bio_model)
    multiple_inputs = [inputs, _alter_sample(inputs, offset=100.0)]

    with create_prediction_pipeline(
        bioimageio_model=bio_model,
        weight_format=weights_format,
        devices=["cpu", "cpu"],
        dataset_for_initial_statistics=[inputs],
    ) as pp:
        # test in a multi-threaded setting
        def run(sample: Sample) -> Sample:
            # Evaluate lazy inference in the worker while the adapter is loaded.
            return pp.predict_without_blocking(
                sample, skip_input_padding=True, skip_output_cropping=True
            ).as_numpy()

        sequential_outputs = [run(sample) for sample in multiple_inputs]
        with ThreadPoolExecutor(max_workers=3) as executor:
            multiple_outputs = list(executor.map(run, multiple_inputs))

        for actual, expected in zip(multiple_outputs, sequential_outputs):
            assert actual.id == expected.id
            assert actual.shape == expected.shape
            for member in expected.members:
                np.testing.assert_allclose(
                    actual.members[member].to_numpy(),
                    expected.members[member].to_numpy(),
                    rtol=0.01,
                    atol=0.1,
                )

        outputs = multiple_outputs[0]

        expected_outputs = get_test_output_sample(bio_model)
        assert len(outputs.shape) == len(expected_outputs.shape)
        for m in expected_outputs.members:
            out = outputs.members[m]
            assert out is not None
            exp = expected_outputs.members[m]
            mismatched_ppm, msg, error_msg = evaluate_mismatched_elements(
                out, exp, rtol=0.01, atol=0.1, mismatched_tol=50_000, name=m
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


def test_blockwise_prediction_with_stacked_dims(unet2d_nuclei_broad_model: Path):
    from bioimageio.core import load_model_description
    from bioimageio.core._prediction_pipeline import create_prediction_pipeline
    from bioimageio.core.axis import AxisInfo, AxisSize
    from bioimageio.core.digest_spec import (
        get_test_input_sample,
        get_test_output_sample,
    )

    bio_model = load_model_description(
        unet2d_nuclei_broad_model, format_version="latest"
    )
    initial_statistics_sample = get_test_input_sample(bio_model)
    with create_prediction_pipeline(
        bioimageio_model=bio_model,
        weight_format="pytorch_state_dict",
        devices=["cpu", "cpu"],
        dataset_for_initial_statistics=[initial_statistics_sample],
    ) as pp:
        inputs = get_test_input_sample(
            bio_model,
            repetitions=[
                (
                    2,
                    AxisInfo(
                        AxisId("extra_dim"), type="index", size=AxisSize(min=2, max=2)
                    ),
                    [1.0, 0.8],
                )
            ],
        )
        outputs = pp.predict_with_blocking(inputs)[{AxisId("extra_dim"): 0}]

        expected_outputs = get_test_output_sample(bio_model)
        assert len(outputs.shape) == len(expected_outputs.shape)
        for m in expected_outputs.members:
            out = outputs.members[m]
            assert out is not None
            exp = expected_outputs.members[m]
            mismatched_ppm, msg, error_msg = evaluate_mismatched_elements(
                out, exp, rtol=0.01, atol=0.1, mismatched_tol=50_000, name=m
            )
            if error_msg is not None:
                raise AssertionError(error_msg)
            elif mismatched_ppm > 50_000:
                raise AssertionError(msg)
