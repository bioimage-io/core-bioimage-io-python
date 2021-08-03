import os
import pytest

try:
    import torch
except ImportError:
    torch = None

try:
    import onnxruntime as rt
except ImportError:
    rt = None


@pytest.mark.skipif(torch is None, reason="requires pytorch")
@pytest.mark.skipif(rt is None, reason="requires onnx")
def test_torchscript_converter(unet2d_nuclei_broad_model, tmp_path):
    from bioimageio.core.weight_converter.torch.onnx import convert_weights_to_onnx

    out_path = tmp_path / "weights.onnx"
    convert_weights_to_onnx(unet2d_nuclei_broad_model, out_path)
    assert os.path.exists(out_path)
    # TODO check that weights can be loaded and results agree if we have onnx runtime
