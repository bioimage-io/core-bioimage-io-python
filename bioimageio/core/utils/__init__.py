from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info < (3, 9):

    def files(package_name: str):
        assert package_name == "bioimageio.core"
        return Path(__file__).parent.parent

else:
    from importlib.resources import files as files


# TODO: import helpers
# @singledispatch
# def import_callable(node: type, /) -> Callable[..., Any]:
#     raise TypeError(type(node))


# @import_callable.register
# def import_from_dependency(node: CallableFromDepencency) -> Callable[..., Any]:
#     module = importlib.import_module(node.module_name)
#     c = getattr(module, node.callable_name)
#     if not callable(c):
#         raise ValueError(f"{node} (imported: {c}) is not callable")

#     return c


# @import_callable.register
# def import_from_file04(node: CallableFromFile, **kwargs: Unpack[HashKwargs]):
#     return _import_from_file_impl(node.file, node.callable_name, **kwargs)


# @import_callable.register
# def import_from_file05(node: CallableFromFile05, **kwargs: Unpack[HashKwargs]):
#     return _import_from_file_impl(node.source_file, node.callable_name, **kwargs)


# def _import_from_file_impl(source: FileSource, callable_name: str, **kwargs: Unpack[HashKwargs]):
#     local_file = download(source, **kwargs)
#     module_name = local_file.path.stem
#     importlib_spec = importlib.util.spec_from_file_location(module_name, local_file.path)
#     if importlib_spec is None:
#         raise ImportError(f"Failed to import {module_name} from {source}.")

#     dep = importlib.util.module_from_spec(importlib_spec)
#     importlib_spec.loader.exec_module(dep)  # type: ignore  # todo: possible to use "loader.load_module"?
#     return getattr(dep, callable_name)
