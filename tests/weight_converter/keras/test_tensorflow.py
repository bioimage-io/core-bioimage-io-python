# type: ignore  # TODO enable type checking
import zipfile
from pathlib import Path

import pytest
from bioimageio.spec import load_description
from bioimageio.spec.model import v0_5

from bioimageio.core.weight_converter.keras._tensorflow import (
    convert_weights_to_tensorflow_saved_model_bundle,
)


@pytest.mark.skip()
def test_tensorflow_converter(any_keras_model: Path, tmp_path: Path):
    model = load_description(any_keras_model)
    out_path = tmp_path / "weights.h5"
    ret_val = convert_weights_to_tensorflow_saved_model_bundle(model, out_path)
    assert out_path.exists()
    assert isinstance(ret_val, v0_5.TensorflowSavedModelBundleWeightsDescr)
    assert ret_val.source == out_path


@pytest.mark.skip()
def test_tensorflow_converter_zipped(any_keras_model: Path, tmp_path: Path):
    out_path = tmp_path / "weights.zip"
    model = load_description(any_keras_model)
    ret_val = convert_weights_to_tensorflow_saved_model_bundle(model, out_path)

    assert out_path.exists()
    assert isinstance(ret_val, v0_5.TensorflowSavedModelBundleWeightsDescr)

    expected_names = {"saved_model.pb", "variables/variables.index"}
    with zipfile.ZipFile(out_path, "r") as f:
        names = set([name for name in f.namelist()])
    assert len(expected_names - names) == 0
