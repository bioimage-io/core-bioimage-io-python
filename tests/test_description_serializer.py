from io import BytesIO
from typing import Any
from zipfile import ZipFile


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
