# type: ignore  # TODO enable type checking
import pytest
from bioimageio.spec import load_description
from bioimageio.spec.model import v0_5

from bioimageio.core.weight_converter.torch._torchscript import (
    convert_weights_to_torchscript,
)


@pytest.mark.skip()
def test_torchscript_converter(any_torch_model, tmp_path):
    bio_model = load_description(any_torch_model)
    out_path = tmp_path / "weights.pt"
    ret_val = convert_weights_to_torchscript(bio_model, out_path)
    assert out_path.exists()
    assert isinstance(ret_val, v0_5.TorchscriptWeightsDescr)
    assert ret_val.source == out_path
