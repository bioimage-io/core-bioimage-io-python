import os
import pytest


@pytest.mark.skipif(pytest.skip_torch, reason="requires pytorch")
def test_torchscript_converter(any_torch_model, tmp_path):
    from bioimageio.core.weight_converter.torch import convert_weights_to_pytorch_script

    out_path = tmp_path / "weights.pt"
    ret_val = convert_weights_to_pytorch_script(any_torch_model, out_path)
    assert os.path.exists(out_path)
    assert ret_val == 0  # check for correctness is done in converter and returns 0 if it passes
