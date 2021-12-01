def test_tensorflow_converter(any_keras_model, tmp_path):
    from bioimageio.core.weight_converter.keras import convert_weights_to_tensorflow_saved_model_bundle

    out_path = tmp_path / "weights"
    ret_val = convert_weights_to_tensorflow_saved_model_bundle(any_keras_model, out_path)
    assert out_path.exists()
    assert (out_path / "variables").exists()
    assert (out_path / "saved_model.pb").exists()
    assert ret_val == 0  # check for correctness is done in converter and returns 0 if it passes


def test_tensorflow_converter_zipped(any_keras_model, tmp_path):
    from bioimageio.core.weight_converter.keras import convert_weights_to_tensorflow_saved_model_bundle

    out_path = tmp_path / "weights.zip"
    ret_val = convert_weights_to_tensorflow_saved_model_bundle(any_keras_model, out_path)
    assert out_path.exists()
    assert ret_val == 0  # check for correctness is done in converter and returns 0 if it passes
