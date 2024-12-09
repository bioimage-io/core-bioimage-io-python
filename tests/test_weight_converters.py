# type: ignore  # TODO enable type checking
import os
import zipfile
from pathlib import Path

import pytest

from bioimageio.core.weight_converters import (
    Pytorch2Onnx,
    Tensorflow2Bundled,
)
from bioimageio.spec import load_description
from bioimageio.spec.model import v0_5


def test_torchscript_converter(any_torch_model, tmp_path):
    from bioimageio.core.weight_converters.pytorch_to_torchscript import convert

    bio_model = load_description(any_torch_model)
    out_path = tmp_path / "weights.pt"
    ret_val = convert(bio_model, out_path)
    assert out_path.exists()
    assert isinstance(ret_val, v0_5.TorchscriptWeightsDescr)
    assert ret_val.source == out_path


def test_onnx_converter(convert_to_onnx, tmp_path):
    from bioimageio.core.weight_converters.pytorch_to_onnx import convert

    bio_model = load_description(convert_to_onnx)
    out_path = tmp_path / "weights.onnx"
    opset_version = 15
    ret_val = convert(
        model_descr=bio_model,
        output_path=out_path,
        test_decimal=3,
        opset_version=opset_version,
    )
    assert os.path.exists(out_path)
    assert isinstance(ret_val, v0_5.OnnxWeightsDescr)
    assert ret_val.opset_version == opset_version
    assert ret_val.source == out_path


def test_tensorflow_converter(any_keras_model: Path, tmp_path: Path):
    model = load_description(any_keras_model)
    out_path = tmp_path / "weights.h5"
    util = Tensorflow2Bundled()
    ret_val = util.convert(model, out_path)
    assert out_path.exists()
    assert isinstance(ret_val, v0_5.TensorflowSavedModelBundleWeightsDescr)
    assert ret_val.source == out_path


@pytest.mark.skip()
def test_tensorflow_converter_zipped(any_keras_model: Path, tmp_path: Path):
    out_path = tmp_path / "weights.zip"
    model = load_description(any_keras_model)
    util = Tensorflow2Bundled()
    ret_val = util.convert(model, out_path)

    assert out_path.exists()
    assert isinstance(ret_val, v0_5.TensorflowSavedModelBundleWeightsDescr)

    expected_names = {"saved_model.pb", "variables/variables.index"}
    with zipfile.ZipFile(out_path, "r") as f:
        names = set([name for name in f.namelist()])
    assert len(expected_names - names) == 0
