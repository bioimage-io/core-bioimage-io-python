import os
import pytest


@pytest.mark.skipif(pytest.skip_torch, reason="requires pytorch")
def test_torchscript_converter(unet2d_nuclei_broad_model, tmp_path):
    from bioimageio.core.weight_converter.torch.onnx import convert_weights_to_onnx

    out_path = tmp_path / "weights.onnx"
    convert_weights_to_onnx(unet2d_nuclei_broad_model, out_path)
    assert os.path.exists(out_path)

    # TODO check that weights can be loaded and results agree if we have onnx runtime
    if not pytest.skip_onnx:
        pass
