import zipfile
from pathlib import Path

from bioimageio.spec import load_description
from bioimageio.spec.model.v0_5 import ModelDescr


def test_tensorflow_converter(any_keras_model: Path, tmp_path: Path):
    from bioimageio.core.weight_converter.keras import convert_weights_to_tensorflow_saved_model_bundle

    out_path = tmp_path / "weights"
    model = load_description(any_keras_model)
    assert isinstance(model, ModelDescr), model.validation_summary.format()
    ret_val = convert_weights_to_tensorflow_saved_model_bundle(model, out_path)
    assert out_path.exists()
    assert (out_path / "variables").exists()
    assert (out_path / "saved_model.pb").exists()
    assert ret_val == 0  # check for correctness is done in converter and returns 0 if it passes


def test_tensorflow_converter_zipped(any_keras_model: Path, tmp_path: Path):
    from bioimageio.core.weight_converter.keras import convert_weights_to_tensorflow_saved_model_bundle

    out_path = tmp_path / "weights.zip"
    model = load_description(any_keras_model)
    assert isinstance(model, ModelDescr), model.validation_summary.format()
    ret_val = convert_weights_to_tensorflow_saved_model_bundle(model, out_path)
    assert out_path.exists()
    assert ret_val == 0  # check for correctness is done in converter and returns 0 if it passes

    # make sure that the zip package was created correctly
    expected_names = {"saved_model.pb", "variables/variables.index"}
    with zipfile.ZipFile(out_path, "r") as f:
        names = set([name for name in f.namelist()])
    assert len(expected_names - names) == 0
