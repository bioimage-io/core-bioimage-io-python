import dataclasses
from pathlib import Path

import pytest
from bioimageio.spec.shared import raw_nodes
from bioimageio.spec.shared.raw_nodes import RawNode

from bioimageio.core._internal import validation_visitors
from bioimageio.core._internal.validation_visitors import Sha256NodeChecker
from bioimageio.core.resource_io import nodes


def test_resolve_import_path(tmpdir):
    tmpdir = Path(tmpdir)
    manifest_path = tmpdir / "manifest.yaml"
    manifest_path.touch()
    source_file = Path("my_mod.py")
    (tmpdir / str(source_file)).write_text("class Foo: pass", encoding="utf8")
    node = raw_nodes.ImportableSourceFile(source_file=source_file, callable_name="Foo")
    uri_transformed = validation_visitors.UriNodeTransformer(root_path=tmpdir).transform(node)
    source_transformed = validation_visitors.SourceNodeTransformer().transform(uri_transformed)
    assert isinstance(source_transformed, nodes.ImportedSource), type(source_transformed)
    Foo = source_transformed.factory
    assert Foo.__name__ == "Foo", Foo.__name__
    assert isinstance(Foo, type), type(Foo)


def test_resolve_directory_uri(tmpdir):
    node = raw_nodes.URI(Path(tmpdir).as_uri())
    uri_transformed = validation_visitors.UriNodeTransformer(root_path=Path(tmpdir)).transform(node)
    assert uri_transformed == Path(tmpdir)


def test_uri_available():
    pass  # todo


def test_all_uris_available():
    from bioimageio.core._internal.validation_visitors import all_sources_available

    not_available = {
        "uri": raw_nodes.URI(scheme="file", path="non_existing_file_in/non_existing_dir/ftw"),
        "uri_exists": raw_nodes.URI(scheme="file", path="."),
    }
    assert not all_sources_available(not_available)


def test_uri_node_transformer_is_ok_with_abs_path():
    from bioimageio.core._internal.validation_visitors import UriNodeTransformer

    # note: the call of .absolute() is required to add the drive letter for windows paths, which are relative otherwise
    tree = {"rel_path": Path("something/relative"), "abs_path": Path("/something/absolute").absolute()}
    assert not tree["rel_path"].is_absolute()
    assert tree["abs_path"].is_absolute()

    root = Path("/root").absolute()
    print(root)

    tree = UriNodeTransformer(root_path=root).transform(tree)
    assert tree["rel_path"].is_absolute()
    assert tree["rel_path"] == Path("/root/something/relative").absolute()
    assert tree["abs_path"].is_absolute()
    assert tree["abs_path"] == Path("/something/absolute").absolute()


def test_sha256_checker(tmpdir):
    root = Path(tmpdir)
    src1 = root / "meh.txt"
    src2 = root / "muh.txt"
    src1.write_text(src1.stem, encoding="utf-8")
    src2.write_text(src2.stem, encoding="utf-8")

    @dataclasses.dataclass
    class TestNode(RawNode):
        source: Path = src1
        sha256: str = "f65255094d7773ed8dd417badc9fc045c1f80fdc5b2d25172b031ce6933e039a"
        my_src: Path = src2
        my_src_sha256: str = "8cf5844c38045aa19aae00d689002549d308de07a777c2ea34355d65283255ac"

    checker = Sha256NodeChecker(root_path=root)
    checker.visit(TestNode())

    with pytest.raises(ValueError):
        checker.visit(TestNode(my_src_sha256="nope"))
