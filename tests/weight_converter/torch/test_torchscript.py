import os
import pytest
try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="requires pytorch")
def test_torchscript_converter(unet2d_nuclei_broad_model, tmp_path):
    from bioimageio.core.weight_converter.torch import convert_weights_to_pytorch_script
    out_path = tmp_path / "weights.pt"
    convert_weights_to_pytorch_script(unet2d_nuclei_broad_model, out_path)
    assert os.path.exists(out_path)
    # TODO check results for correctness
