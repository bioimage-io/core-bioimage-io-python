# type: ignore  # TODO enable type checking
import os

from bioimageio.spec import load_description
from bioimageio.spec.model import v0_5

from bioimageio.core.weight_converter.torch._onnx import convert_weights_to_onnx


def test_onnx_converter(convert_to_onnx, tmp_path):
    bio_model = load_description(convert_to_onnx)
    out_path = tmp_path / "weights.onnx"
    opset_version = 15
    ret_val = convert_weights_to_onnx(
        model_spec=bio_model,
        output_path=out_path,
        test_decimal=3,
        opset_version=opset_version,
    )
    assert os.path.exists(out_path)
    assert isinstance(ret_val, v0_5.OnnxWeightsDescr)
    assert ret_val.opset_version == opset_version
    assert ret_val.source == out_path
