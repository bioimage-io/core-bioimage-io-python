from pybio.spec.nodes import ImportablePath
from pybio.spec.utils import _resolve_import


def test_resolve_import_path(tmpdir):
    filepath = tmpdir / "my_mod.py"
    filepath.write_text("class Foo: pass", encoding="utf8")
    Foo = _resolve_import(ImportablePath(filepath=filepath, callable_name="Foo"))
    assert Foo.__name__ == "Foo"
    assert isinstance(Foo, type)
