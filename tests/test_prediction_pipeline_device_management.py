from pathlib import Path

from numpy.testing import assert_array_almost_equal

from bioimageio.core.utils.testing import skip_on
from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr04
from bioimageio.spec.model.v0_5 import ModelDescr, WeightsFormat


class TooFewDevicesException(Exception):
    pass


def _test_device_management(model_package: Path, weight_format: WeightsFormat):
    import torch

    from bioimageio.core import load_description
    from bioimageio.core._prediction_pipeline import create_prediction_pipeline
    from bioimageio.core.digest_spec import get_test_inputs, get_test_outputs

    if torch.cuda.device_count() == 0:
        raise TooFewDevicesException("Need at least one cuda device for this test")

    bio_model = load_description(model_package)
    assert isinstance(bio_model, (ModelDescr, ModelDescr04))
    pred_pipe = create_prediction_pipeline(
        bioimageio_model=bio_model, weight_format=weight_format, devices=["cuda:0"]
    )

    inputs = get_test_inputs(bio_model)
    with pred_pipe as pp:
        outputs = pp.predict_sample_without_blocking(inputs)

    expected_outputs = get_test_outputs(bio_model)

    assert len(outputs.shape) == len(expected_outputs.shape)
    for m in expected_outputs.members:
        out = outputs.members[m].data
        assert out is not None
        exp = expected_outputs.members[m].data
        assert_array_almost_equal(out, exp, decimal=4)

    # repeat inference with context manager to test load/predict/unload/load/predict
    with pred_pipe as pp:
        outputs = pp.predict_sample_without_blocking(inputs)

    assert len(outputs.shape) == len(expected_outputs.shape)
    for m in expected_outputs.members:
        out = outputs.members[m].data
        assert out is not None
        exp = expected_outputs.members[m].data
        assert_array_almost_equal(out, exp, decimal=4)


@skip_on(
    TooFewDevicesException, reason="Too few devices"
)  # pyright: ignore[reportArgumentType]
def test_device_management_torch(any_torch_model: Path):
    _test_device_management(any_torch_model, "pytorch_state_dict")


@skip_on(
    TooFewDevicesException, reason="Too few devices"
)  # pyright: ignore[reportArgumentType]
def test_device_management_torchscript(any_torchscript_model: Path):
    _test_device_management(any_torchscript_model, "torchscript")


@skip_on(
    TooFewDevicesException, reason="Too few devices"
)  # pyright: ignore[reportArgumentType]
def test_device_management_onnx(any_onnx_model: Path):
    _test_device_management(any_onnx_model, "onnx")


@skip_on(
    TooFewDevicesException, reason="Too few devices"
)  # pyright: ignore[reportArgumentType]
def test_device_management_tensorflow(any_tensorflow_model: Path):
    _test_device_management(any_tensorflow_model, "tensorflow_saved_model_bundle")


@skip_on(
    TooFewDevicesException, reason="Too few devices"
)  # pyright: ignore[reportArgumentType]
def test_device_management_keras(any_keras_model: Path):
    _test_device_management(any_keras_model, "keras_hdf5")
