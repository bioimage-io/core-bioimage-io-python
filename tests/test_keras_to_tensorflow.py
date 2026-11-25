import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

_ = pytest.importorskip("tensorflow")

from bioimageio.core.weight_converters import keras_to_tensorflow as converter
from bioimageio.spec.model import v0_5

# pyright: reportPrivateUsage=false


@pytest.fixture
def model():
    return v0_5.ModelDescr.model_construct(
        inputs=[v0_5.InputTensorDescr.model_construct(id=v0_5.TensorId("input"))],
        outputs=[v0_5.OutputTensorDescr.model_construct(id=v0_5.TensorId("output"))],
        weights=SimpleNamespace(
            keras_hdf5=SimpleNamespace(get_reader=Mock(), tensorflow_version=None)
        ),
    )


@pytest.mark.parametrize("version", ["1.15.0", "2.15.0"])
@pytest.mark.parametrize("suffix", ["", ".zip"])
def test_conversion_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: v0_5.ModelDescr,
    version: str,
    suffix: str,
):
    monkeypatch.setattr(converter, "tensorflow", SimpleNamespace(__version__=version))
    unzipped = tmp_path / "weights.h5"
    ensure = Mock(return_value=unzipped)
    monkeypatch.setattr(converter, "ensure_unzipped", ensure)
    tf1, tf2 = Mock(), Mock()
    monkeypatch.setattr(converter, "_convert_tf1", tf1)
    monkeypatch.setattr(converter, "_convert_tf2", tf2)
    target = tmp_path / ("saved" + suffix)
    factory, other = (tf1, tf2) if version.startswith("1") else (tf2, tf1)
    assert converter.convert(model, target) is factory.return_value
    factory.assert_called_once_with(
        unzipped,
        tmp_path / "saved",
        *(["input", "output"] if version.startswith("1") else []),
    )
    other.assert_not_called()
    assert (
        ensure.call_args_list[0].args[0]
        is cast(Any, model.weights.keras_hdf5).get_reader.return_value
    )


@pytest.mark.parametrize("directory", [False, True])
def test_conversion_rejects_existing_output_and_missing_weights(
    tmp_path: Path, directory: bool
):
    model = v0_5.ModelDescr.model_construct(weights=SimpleNamespace(keras_hdf5=None))
    existing = tmp_path / ("existing" if directory else "existing.zip")
    if directory:
        existing.mkdir()
    else:
        _ = existing.write_bytes(b"keep me")
    with pytest.raises(ValueError, match="must not exist"):
        _ = converter.convert(model, existing)
    if directory:
        assert existing.is_dir()
    else:
        assert existing.read_bytes() == b"keep me"
    with pytest.raises(ValueError, match="Missing Keras Hdf5 weights"):
        _ = converter.convert(model, tmp_path / "new.zip")


def test_conversion_rejects_tensorflow_major_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: v0_5.ModelDescr
):
    monkeypatch.setattr(converter, "tensorflow", SimpleNamespace(__version__="2.15.0"))
    cast(Any, model.weights.keras_hdf5).tensorflow_version = v0_5.Version("1.15.0")
    with pytest.raises(RuntimeError, match="major versions of model 1 is not 2"):
        _ = converter.convert(model, tmp_path / "saved")


@pytest.mark.parametrize("multiple", ["inputs", "outputs"])
def test_tf1_rejects_multiple_tensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: v0_5.ModelDescr,
    multiple: str,
):
    monkeypatch.setattr(converter, "tensorflow", SimpleNamespace(__version__="1.15.0"))
    monkeypatch.setattr(
        converter, "ensure_unzipped", Mock(return_value=tmp_path / "weights")
    )
    tensors = getattr(model, multiple)
    tensors.append(tensors[0])
    with pytest.raises(NotImplementedError, match="multiple inputs or outputs"):
        _ = converter.convert(model, tmp_path / "saved")


def test_tf1_legacy_tensor_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: v0_5.ModelDescr
):
    monkeypatch.setattr(converter, "tensorflow", SimpleNamespace(__version__="1.15.0"))
    monkeypatch.setattr(
        converter, "ensure_unzipped", Mock(return_value=tmp_path / "weights")
    )
    model = model.model_copy(
        update={
            "inputs": [SimpleNamespace(name="old_input")],
            "outputs": [SimpleNamespace(name="old_output")],
        }
    )
    convert = Mock()
    monkeypatch.setattr(converter, "_convert_tf1", convert)
    assert converter.convert(model, tmp_path / "saved") is convert.return_value
    assert convert.call_args.args[2:] == ("old_input", "old_output")


def write_saved_model(path: Path):
    path.mkdir()
    _ = (path / "saved_model.pb").write_bytes(b"graph")
    (path / "variables").mkdir()
    _ = (path / "variables" / "variables.index").write_bytes(b"index")


def assert_bundle(
    descriptor: v0_5.TensorflowSavedModelBundleWeightsDescr, target: Path
):
    assert descriptor.source == target.with_suffix(".zip").absolute()
    assert descriptor.parent == "keras_hdf5"
    assert not target.exists()
    with ZipFile(target.with_suffix(".zip")) as archive:
        assert set(archive.namelist()) == {
            "saved_model.pb",
            "variables/variables.index",
        }
        assert archive.read("variables/variables.index") == b"index"


def test_tf2_export_and_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    model = Mock()
    model.export.side_effect = write_saved_model
    load = Mock(return_value=model)
    monkeypatch.setattr(
        converter, "keras", SimpleNamespace(models=SimpleNamespace(load_model=load))
    )
    monkeypatch.setattr(converter, "tensorflow", SimpleNamespace(__version__="2.15.0"))
    target, source = tmp_path / "saved", tmp_path / "weights.h5"
    descriptor = converter._convert_tf2(source, target)
    load.assert_called_once_with(source)
    model.export.assert_called_once_with(target)
    assert_bundle(descriptor, target)
    assert str(descriptor.tensorflow_version) == "2.15.0"


@pytest.mark.parametrize("suffix", ["", ".zip"])
def test_convert_creates_archive_at_requested_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: v0_5.ModelDescr, suffix: str
):
    keras_model = Mock()
    keras_model.export.side_effect = write_saved_model
    monkeypatch.setattr(
        converter,
        "keras",
        SimpleNamespace(
            models=SimpleNamespace(load_model=Mock(return_value=keras_model))
        ),
    )
    monkeypatch.setattr(converter, "tensorflow", SimpleNamespace(__version__="2.15.0"))
    monkeypatch.setattr(
        converter, "ensure_unzipped", Mock(return_value=tmp_path / "weights.h5")
    )
    cast(Any, model.weights.keras_hdf5).tensorflow_version = v0_5.Version("2.15.0")
    descriptor = converter.convert(model, tmp_path / ("saved" + suffix))
    assert_bundle(descriptor, tmp_path / "saved")
    with pytest.raises(ValueError, match="must not exist"):
        _ = converter.convert(model, tmp_path / ("saved" + suffix))
    assert_bundle(descriptor, tmp_path / "saved")


def test_tf1_signature_and_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    target = tmp_path / "saved"
    model, session, signature = Mock(), Mock(), Mock()
    builder = Mock()
    builder.save.side_effect = lambda: write_saved_model(target)
    builder_factory, predict_signature = (
        Mock(return_value=builder),
        Mock(return_value=signature),
    )
    monkeypatch.setattr(
        converter,
        "keras",
        SimpleNamespace(
            models=SimpleNamespace(load_model=Mock(return_value=model)),
            backend=SimpleNamespace(get_session=Mock(return_value=session)),
        ),
    )
    monkeypatch.setattr(
        converter,
        "tensorflow",
        SimpleNamespace(
            __version__="1.15.0",
            saved_model=SimpleNamespace(
                builder=SimpleNamespace(SavedModelBuilder=builder_factory),
                signature_def_utils=SimpleNamespace(
                    predict_signature_def=predict_signature
                ),
                signature_constants=SimpleNamespace(
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY="default"
                ),
                tag_constants=SimpleNamespace(SERVING="serve"),
            ),
        ),
    )
    descriptor = converter._convert_tf1(
        tmp_path / "weights.h5", target, "input", "output"
    )
    predict_signature.assert_called_once_with(
        inputs={"input": model.input}, outputs={"output": model.output}
    )
    builder_factory.assert_called_once_with(target)
    builder.add_meta_graph_and_variables.assert_called_once_with(
        session, ["serve"], signature_def_map={"default": signature}
    )
    assert_bundle(descriptor, target)


def test_bundle_preserved_if_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    target = tmp_path / "saved"
    write_saved_model(target)
    monkeypatch.setattr(shutil, "rmtree", Mock(side_effect=PermissionError("busy")))
    path = converter._zip_model_bundle(target)
    with ZipFile(path) as archive:
        assert archive.read("saved_model.pb") == b"graph"
    assert target.exists()
    assert "was not removed after compression" in capsys.readouterr().out
