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
    assert len(input_tensors) == 1
    tagged_data = [
        xr.DataArray(ipt_tensor, dims=tuple(ipt.axes)) for ipt_tensor, ipt in zip(input_tensors, bio_model.inputs)
    ]
    output = pp.forward(*tagged_data)

    expected_outputs = [np.load(opt) for opt in bio_model.test_outputs]
    assert len(expected_outputs) == 1
    expected_output = expected_outputs[0]

    assert_array_almost_equal(output, expected_output, decimal=4)


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
def test_prediction_pipeline_torch(unet2d_nuclei_broad_model):
    _test_prediction_pipeline(unet2d_nuclei_broad_model, "pytorch_state_dict")


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
def test_prediction_pipeline_torchscript(unet2d_nuclei_broad_model):
    _test_prediction_pipeline(unet2d_nuclei_broad_model, "pytorch_script")


@pytest.mark.skipif(pytest.skip_onnx, reason="requires onnx")
def test_prediction_pipeline_onnx(unet2d_nuclei_broad_model):
    _test_prediction_pipeline(unet2d_nuclei_broad_model, "onnx")


@pytest.mark.skipif(pytest.skip_tf or pytest.tf_major_version != 1, reason="requires tensorflow1")
def test_prediction_pipeline_tensorflow(FruNet_model):
    _test_prediction_pipeline(FruNet_model, "tensorflow_saved_model_bundle")


@pytest.mark.skipif(pytest.skip_keras, reason="requires keras")
def test_prediction_pipeline_keras(FruNet_model):
    _test_prediction_pipeline(FruNet_model, "keras_hdf5")
