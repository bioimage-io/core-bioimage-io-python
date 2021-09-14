import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

from bioimageio.core import load_resource_description
from bioimageio.core.resource_io.nodes import Model


def _test_prediction_pipeline(model_package, weight_format):
    from bioimageio.core.prediction_pipeline import create_prediction_pipeline

    bio_model = load_resource_description(model_package)
    assert isinstance(bio_model, Model)
    pp = create_prediction_pipeline(bioimageio_model=bio_model, weight_format=weight_format)

    input_tensors = [np.load(str(ipt)) for ipt in bio_model.test_inputs]
    tagged_data = [
        xr.DataArray(ipt_tensor, dims=tuple(ipt.axes)) for ipt_tensor, ipt in zip(input_tensors, bio_model.inputs)
    ]
    outputs = pp.forward(*tagged_data)
    assert isinstance(outputs, list)

    expected_outputs = [np.load(str(opt)) for opt in bio_model.test_outputs]
    assert len(outputs) == len(expected_outputs)

    for out, exp in zip(outputs, expected_outputs):
        assert_array_almost_equal(out, exp, decimal=4)


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
def test_prediction_pipeline_torch(any_torch_model):
    _test_prediction_pipeline(any_torch_model, "pytorch_state_dict")


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
def test_prediction_pipeline_torchscript(any_torchsccript_model):
    _test_prediction_pipeline(any_torchsccript_model, "pytorch_script")


@pytest.mark.skipif(pytest.skip_onnx, reason="requires onnx")
def test_prediction_pipeline_onnx(any_onnx_model):
    _test_prediction_pipeline(any_onnx_model, "onnx")


@pytest.mark.skipif(pytest.skip_tensorflow or pytest.tf_major_version != 1, reason="requires tensorflow 1")
def test_prediction_pipeline_tensorflow(any_tensorflow1_model):
    _test_prediction_pipeline(any_tensorflow1_model, "tensorflow_saved_model_bundle")


@pytest.mark.skipif(pytest.skip_tensorflow or pytest.tf_major_version != 2, reason="requires tensorflow 2")
def test_prediction_pipeline_tensorflow(any_tensorflow2_model):
    _test_prediction_pipeline(any_tensorflow2_model, "tensorflow_saved_model_bundle")


@pytest.mark.skipif(pytest.skip_keras, reason="requires keras")
def test_prediction_pipeline_keras(any_keras_model):
    _test_prediction_pipeline(any_keras_model, "keras_hdf5")
