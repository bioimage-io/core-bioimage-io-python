import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest


def test_description_serializer_roundtrip(unet2d_nuclei_broad_model: str):
    from bioimageio.core._description_serializer import DescriptionSerializer
    from bioimageio.core._utils.type_guards import is_dict, is_list
    from bioimageio.spec import load_model_description
    from bioimageio.spec._internal.node import Node
    from bioimageio.spec.common import FileDescr, RelativeFilePath

    rd1 = load_model_description(unet2d_nuclei_broad_model, perform_io_checks=False)

    serialized = DescriptionSerializer.serialize_to_string(rd1)
    assert len(serialized) > 2083

    rd2 = DescriptionSerializer.deserialize_from_string(serialized)

    def compare(f1: Any, f2: Any):
        if is_list(f1):
            assert is_list(f2)
            assert len(f1) == len(f2)
            for v1, v2 in zip(f1, f2):
                compare(v1, v2)
        elif is_dict(f1):
            assert is_dict(f2)
            assert len(f1) == len(f2)
            for k, v1 in f1.items():
                v2 = f2[k]
                compare(v1, v2)
        elif isinstance(f1, FileDescr) and isinstance(f1.source, RelativeFilePath):
            assert isinstance(f2, FileDescr)
            assert f1.download().read() == f2.download().read()
            assert f1.sha256 == f2.sha256
        elif isinstance(f1, Node):
            assert isinstance(f2, Node)
            assert f1.__class__ == f2.__class__
            for k, v in f1:
                compare(v, getattr(f2, k))
        else:
            assert str(f1) == str(f2), (f1, f2)

    compare(rd1, rd2)


def test_description_serializer_includes_v0_4_relative_architecture(
    unet2d_nuclei_broad_model_old: str,
):
    from bioimageio.core._description_serializer import DescriptionSerializer
    from bioimageio.spec import load_model_description
    from bioimageio.spec.common import FileDescr
    from bioimageio.spec.model import v0_4

    model = load_model_description(
        unet2d_nuclei_broad_model_old, perform_io_checks=False
    )
    assert isinstance(model, v0_4.ModelDescr)
    weights = model.weights.pytorch_state_dict
    assert weights is not None
    assert isinstance(weights.architecture, v0_4.CallableFromFile)
    architecture = weights.architecture.source_file
    source = FileDescr(
        source=architecture, sha256=weights.architecture_sha256
    ).get_reader()

    with ZipFile(BytesIO(DescriptionSerializer.serialize(model))) as package:
        assert package.read(str(architecture)) == source.read()


@pytest.mark.parametrize(
    "architecture_path", ["unet2d.py", "nested/unet2d.py", "../unet2d.py"]
)
def test_description_serializer_roundtrip_local_v0_4_architecture(
    unet2d_nuclei_broad_model_old: str, tmp_path: Path, architecture_path: str
):
    from bioimageio.core._description_serializer import DescriptionSerializer
    from bioimageio.spec import ValidationContext, load_model_description
    from bioimageio.spec.common import FileDescr
    from bioimageio.spec.model import v0_4

    model = load_model_description(
        unet2d_nuclei_broad_model_old, perform_io_checks=False
    )
    assert isinstance(model, v0_4.ModelDescr)
    weights = model.weights.pytorch_state_dict
    assert weights is not None
    root = tmp_path / "model"
    root.mkdir()
    path = root / architecture_path
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = b"class MyModel: pass\n"
    path.write_bytes(contents)
    with ValidationContext(root=root, perform_io_checks=False):
        architecture = v0_4.CallableFromFile(f"{architecture_path}:MyModel")
    weights = weights.model_copy(
        update={
            "architecture": architecture,
            "architecture_sha256": hashlib.sha256(contents).hexdigest(),
        }
    )
    model = model.model_copy(
        update={
            "weights": model.weights.model_copy(update={"pytorch_state_dict": weights})
        }
    )

    if architecture_path.startswith("../"):
        with pytest.raises(ValueError, match="Unsafe architecture path"):
            DescriptionSerializer.serialize(model)
        return

    serialized = DescriptionSerializer.serialize(model)
    with ZipFile(BytesIO(serialized)) as package:
        assert package.namelist().count(architecture_path) == 1
        assert package.read(architecture_path) == contents

    restored = DescriptionSerializer.deserialize(serialized)
    assert isinstance(restored, v0_4.ModelDescr)
    restored_weights = restored.weights.pytorch_state_dict
    assert restored_weights is not None
    assert isinstance(restored_weights.architecture, v0_4.CallableFromFile)
    reader = FileDescr(source=restored_weights.architecture.source_file).get_reader()
    assert reader.read() == contents
