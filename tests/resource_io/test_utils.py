from pathlib import Path

from bioimageio.core.resource_io import nodes, utils
from bioimageio.spec.shared import raw_nodes


def test_resolve_import_path(tmpdir):
    tmpdir = Path(tmpdir)
    manifest_path = tmpdir / "manifest.yaml"
    manifest_path.touch()
    source_file = nodes.URI(path="my_mod.py")
    (tmpdir / str(source_file)).write_text("class Foo: pass", encoding="utf8")
    node = raw_nodes.ImportableSourceFile(source_file=source_file, callable_name="Foo")
    uri_transformed = utils.UriNodeTransformer(root_path=tmpdir).transform(node)
    source_transformed = utils.SourceNodeTransformer().transform(uri_transformed)
    assert isinstance(source_transformed, nodes.ImportedSource)
    Foo = source_transformed.factory
    assert Foo.__name__ == "Foo"
    assert isinstance(Foo, type)


def test_resolve_directory_uri(tmpdir):
    node = raw_nodes.URI(Path(tmpdir).as_uri())
    uri_transformed = utils.UriNodeTransformer(root_path=Path(tmpdir)).transform(node)
    assert uri_transformed == Path(tmpdir)


def test_uri_available():
    pass  # todo


def test_all_uris_available():
    from bioimageio.core.resource_io.utils import all_uris_available

    not_available = {
        "uri": raw_nodes.URI(path="non_existing_file_in/non_existing_dir/ftw"),
        "uri_exists": raw_nodes.URI(path="."),
    }
    assert not all_uris_available(not_available)


def test_uri_node_transformer_is_ok_with_abs_path():
    from bioimageio.core.resource_io.utils import UriNodeTransformer

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
