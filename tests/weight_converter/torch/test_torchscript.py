def test_torchscript_converter(any_torch_model, tmp_path):
    from bioimageio.core.weight_converter.torch import convert_weights_to_torchscript

    out_path = tmp_path / "weights.pt"
    ret_val = convert_weights_to_torchscript(any_torch_model, out_path)
    assert out_path.exists()
    assert ret_val == 0  # check for correctness is done in converter and returns 0 if it passes
