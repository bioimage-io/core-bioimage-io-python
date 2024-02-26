from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_array_almost_equal

from bioimageio.core.utils.testing import skip_on
from bioimageio.spec import load_description
from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr04
from bioimageio.spec.model.v0_5 import ModelDescr, WeightsFormat
from bioimageio.spec.utils import load_array


class TooFewDevicesException(Exception):
    pass


def _test_device_management(model_package: Path, weight_format: WeightsFormat):
    import torch

    if torch.cuda.device_count() == 0:
        raise TooFewDevicesException("Need at least one cuda device for this test")

    from bioimageio.core.prediction_pipeline import create_prediction_pipeline

    bio_model = load_description(model_package)
    assert isinstance(bio_model, (ModelDescr, ModelDescr04))
    pred_pipe = create_prediction_pipeline(bioimageio_model=bio_model, weight_format=weight_format, devices=["cuda:0"])

    if isinstance(bio_model, ModelDescr04):
        inputs = [
            xr.DataArray(np.load(str(test_tensor)), dims=tuple(spec.axes))
            for test_tensor, spec in zip(bio_model.test_inputs, bio_model.inputs)
        ]
    else:
        inputs = [
            xr.DataArray(load_array(ipt.test_tensor), dims=tuple(a.id for a in ipt.axes)) for ipt in bio_model.inputs
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


@skip_on(TooFewDevicesException, reason="Too few devices")
def test_device_management_onnx(any_onnx_model):
    _test_device_management(any_onnx_model, "onnx")


@skip_on(TooFewDevicesException, reason="Too few devices")
def test_device_management_tensorflow(any_tensorflow_model):
    _test_device_management(any_tensorflow_model, "tensorflow_saved_model_bundle")


@skip_on(TooFewDevicesException, reason="Too few devices")
def test_device_management_keras(any_keras_model):
    _test_device_management(any_keras_model, "keras_hdf5")
