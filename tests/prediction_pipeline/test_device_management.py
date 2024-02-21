import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

from bioimageio.core import load_description
from bioimageio.core._internal.pytest_utils import skip_on
from bioimageio.core.resource_io.nodes import Model


class TooFewDevicesException(Exception):
    pass


def _test_device_management(model_package, weight_format):
    import torch

    if torch.cuda.device_count() == 0:
        raise TooFewDevicesException("Need at least one cuda device for this test")

    from bioimageio.core.prediction_pipeline import create_prediction_pipeline

    bio_model = load_description(model_package)
    assert isinstance(bio_model, Model)
    pred_pipe = create_prediction_pipeline(bioimageio_model=bio_model, weight_format=weight_format, devices=["cuda:0"])

    inputs = [
        xr.DataArray(np.load(str(test_tensor)), dims=tuple(spec.axes))
        for test_tensor, spec in zip(bio_model.test_inputs, bio_model.inputs)
    ]
    with pred_pipe as pp:
        outputs = pp.forward(*inputs)

    assert isinstance(outputs, list)

    expected_outputs = [
        xr.DataArray(np.load(str(test_tensor)), dims=tuple(spec.axes))
        for test_tensor, spec in zip(bio_model.test_outputs, bio_model.outputs)
    ]

    assert len(outputs) == len(expected_outputs)
    for out, exp in zip(outputs, expected_outputs):
        assert_array_almost_equal(out, exp, decimal=4)

    # repeat inference with context manager to test load/unload/load/forward
    with pred_pipe as pp:
        outputs = pp.forward(*inputs)

    assert len(outputs) == len(expected_outputs)
    for out, exp in zip(outputs, expected_outputs):
        assert_array_almost_equal(out, exp, decimal=4)


@skip_on(TooFewDevicesException, reason="Too few devices")
def test_device_management_torch(any_torch_model):
    _test_device_management(any_torch_model, "pytorch_state_dict")


@skip_on(TooFewDevicesException, reason="Too few devices")
def test_device_management_torchscript(any_torchscript_model):
    _test_device_management(any_torchscript_model, "torchscript")


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch for device discovery")
@skip_on(TooFewDevicesException, reason="Too few devices")
def test_device_management_onnx(any_onnx_model):
    _test_device_management(any_onnx_model, "onnx")


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch for device discovery")
@skip_on(TooFewDevicesException, reason="Too few devices")
def test_device_management_tensorflow(any_tensorflow_model):
    _test_device_management(any_tensorflow_model, "tensorflow_saved_model_bundle")


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch for device discovery")
@skip_on(TooFewDevicesException, reason="Too few devices")
def test_device_management_keras(any_keras_model):
    _test_device_management(any_keras_model, "keras_hdf5")
