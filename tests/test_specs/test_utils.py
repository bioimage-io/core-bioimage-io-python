from pathlib import Path

from pybio.spec.nodes import ImportablePath
from pybio.spec.utils import URITransformer, SourceTransformer, ImportedSource


def test_resolve_import_path(tmpdir, cache_path):
    tmpdir = Path(tmpdir)
    manifest_path = tmpdir / "manifest.yaml"
    manifest_path.touch()
    filepath = tmpdir / "my_mod.py"
    filepath.write_text("class Foo: pass", encoding="utf8")
    node = ImportablePath(filepath=filepath, callable_name="Foo")
    uri_transformed = URITransformer(root_path=tmpdir, cache_path=cache_path).transform(node)
    source_transformed = SourceTransformer().transform(uri_transformed)
    assert isinstance(source_transformed, ImportedSource)
    Foo = source_transformed.callable_
    assert Foo.__name__ == "Foo"
    assert isinstance(Foo, type)
