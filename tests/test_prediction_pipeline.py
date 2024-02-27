from pathlib import Path
import numpy as np
import xarray as xr
from numpy.testing import assert_array_almost_equal

from bioimageio.spec import load_description
from bioimageio.spec.model.v0_5 import WeightsFormat, ModelDescr
from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr04


def _test_prediction_pipeline(model_package: Path, weights_format: WeightsFormat):
    from bioimageio.core.prediction_pipeline import create_prediction_pipeline

    bio_model = load_description(model_package)
    assert isinstance(bio_model, (ModelDescr, ModelDescr04))
    pp = create_prediction_pipeline(bioimageio_model=bio_model, weight_format=weights_format)

    if isinstance(bio_model, ModelDescr04):
        inputs = [
            xr.DataArray(np.load(str(test_tensor)), dims=tuple(spec.axes))
            for test_tensor, spec in zip(bio_model.test_inputs, bio_model.inputs)
        ]
    else:

    outputs = pp.forward(*inputs)
    assert isinstance(outputs, list)

    expected_outputs = [
        xr.DataArray(np.load(str(test_tensor)), dims=tuple(spec.axes))
        for test_tensor, spec in zip(bio_model.test_outputs, bio_model.outputs)
    ]
    assert len(outputs) == len(expected_outputs)

    for out, exp in zip(outputs, expected_outputs):
        assert_array_almost_equal(out, exp, decimal=4)


def test_prediction_pipeline_torch(any_torch_model):
    _test_prediction_pipeline(any_torch_model, "pytorch_state_dict")


def test_prediction_pipeline_torchscript(any_torchscript_model):
    _test_prediction_pipeline(any_torchscript_model, "torchscript")


def test_prediction_pipeline_onnx(any_onnx_model):
    _test_prediction_pipeline(any_onnx_model, "onnx")


def test_prediction_pipeline_tensorflow(any_tensorflow_model):
    _test_prediction_pipeline(any_tensorflow_model, "tensorflow_saved_model_bundle")


def test_prediction_pipeline_keras(any_keras_model):
    _test_prediction_pipeline(any_keras_model, "keras_hdf5")
