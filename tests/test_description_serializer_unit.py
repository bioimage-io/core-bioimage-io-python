import base64
import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from bioimageio.core import _description_serializer as module
from bioimageio.core._description_serializer import DescriptionSerializer
from bioimageio.spec import InvalidDescr, ValidationContext
from bioimageio.spec.model import v0_4, v0_5


def package_bytes(files: dict[str, bytes]) -> bytes:
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


@pytest.mark.parametrize("size", [1, 3000])
def test_string_encoding_padding_and_hash(monkeypatch: pytest.MonkeyPatch, size: int):
    raw = bytes(range(256)) * size
    serialize = Mock(return_value=raw)
    monkeypatch.setattr(DescriptionSerializer, "serialize", serialize)
    model = v0_5.ModelDescr.model_construct()
    encoded, digest = DescriptionSerializer.serialize_to_string_and_hash(model)
    assert base64.b64decode(encoded) == raw
    assert len(encoded) == max(
        DescriptionSerializer.MIN_LENGTH, len(base64.b64encode(raw))
    )
    assert digest == hashlib.sha256(raw).hexdigest()
    assert DescriptionSerializer.serialize_to_string(model) == encoded
    deserialize = Mock(return_value=model)
    monkeypatch.setattr(DescriptionSerializer, "deserialize", deserialize)
    assert DescriptionSerializer.deserialize_from_string(encoded) is model
    deserialize.assert_called_once_with(raw)


@pytest.mark.parametrize("invalid", [False, True])
def test_deserialization_checks_description(
    monkeypatch: pytest.MonkeyPatch, invalid: bool
):
    descr: Any = (
        Mock(spec=InvalidDescr) if invalid else v0_5.ModelDescr.model_construct()
    )
    if invalid:
        descr.get_reason.return_value = "broken metadata"
    load = Mock(return_value=descr)
    monkeypatch.setattr(module, "load_description", load)
    raw = package_bytes({"rdf.yaml": b"type: model"})
    if invalid:
        with pytest.raises(
            TypeError, match="invalid serialized model package: broken metadata"
        ):
            _ = DescriptionSerializer.deserialize(raw)
    else:
        assert DescriptionSerializer.deserialize(raw) is descr
    assert load.call_args.kwargs == {"perform_io_checks": False}
    assert load.call_args.args[0].read("rdf.yaml") == b"type: model"


@pytest.mark.parametrize("existing", [None, b"class Net: pass", b"conflict"])
def test_relative_architecture_inclusion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, existing: bytes | None
):
    content = b"class Net: pass"
    _ = (tmp_path / "network.py").write_bytes(content)
    with ValidationContext(root=tmp_path, perform_io_checks=False):
        architecture = v0_4.CallableFromFile("network.py:Net")
    model = v0_4.ModelDescr.model_construct(
        weights=SimpleNamespace(
            pytorch_state_dict=SimpleNamespace(
                architecture=architecture,
                architecture_sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    )
    raw = package_bytes({} if existing is None else {"network.py": existing})
    save = Mock(return_value=BytesIO(raw))
    monkeypatch.setattr(module, "save_bioimageio_package_to_stream", save)
    if existing == b"conflict":
        with pytest.raises(ValueError, match="Conflicting package file"):
            _ = DescriptionSerializer.serialize(model)
    else:
        with ZipFile(BytesIO(DescriptionSerializer.serialize(model))) as archive:
            assert archive.namelist().count("network.py") == 1
            assert archive.read("network.py") == content
    save.assert_called_once_with(model, local_files_only=True)


@pytest.mark.parametrize("old", [False, True])
def test_serialize_without_architecture(monkeypatch: pytest.MonkeyPatch, old: bool):
    raw = package_bytes({"rdf.yaml": b"type: model"})
    monkeypatch.setattr(
        module, "save_bioimageio_package_to_stream", Mock(return_value=BytesIO(raw))
    )
    model = (v0_4.ModelDescr if old else v0_5.ModelDescr).model_construct(
        weights=SimpleNamespace(pytorch_state_dict=None)
    )
    assert DescriptionSerializer.serialize(model) == raw


@pytest.mark.parametrize("from_url", [False, True])
def test_nonrelative_architecture_is_left_to_package_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, from_url: bool
):
    with ValidationContext(root=tmp_path, perform_io_checks=False):
        architecture = (
            v0_4.CallableFromFile("https://example.org/network.py:Net")
            if from_url
            else v0_4.CallableFromDepencency("package.network.Net")
        )
    model = v0_4.ModelDescr.model_construct(
        weights=SimpleNamespace(
            pytorch_state_dict=SimpleNamespace(architecture=architecture)
        )
    )
    raw = package_bytes({})
    monkeypatch.setattr(
        module, "save_bioimageio_package_to_stream", Mock(return_value=BytesIO(raw))
    )
    assert DescriptionSerializer.serialize(model) == raw
