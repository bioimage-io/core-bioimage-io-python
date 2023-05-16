import os
import pytest


# todo: test with 'any_torch_model'
def test_onnx_converter(convert_to_onnx, tmp_path):
    from bioimageio.core.weight_converter.torch.onnx import convert_weights_to_onnx

    out_path = tmp_path / "weights.onnx"
    ret_val = convert_weights_to_onnx(convert_to_onnx, out_path, test_decimal=3)
    assert os.path.exists(out_path)
    if not pytest.skip_onnx:
        assert ret_val == 0  # check for correctness is done in converter and returns 0 if it passes
