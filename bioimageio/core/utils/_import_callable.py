from __future__ import annotations

import importlib.util
from functools import singledispatch
from typing import Any, Callable

from typing_extensions import Unpack

from bioimageio.spec._internal.io_utils import HashKwargs, download
from bioimageio.spec.common import FileSource
from bioimageio.spec.model.v0_4 import CallableFromDepencency, CallableFromFile
from bioimageio.spec.model.v0_5 import ArchitectureFromFileDescr, ArchitectureFromLibraryDescr


@singledispatch
def import_callable(node: type, /) -> Callable[..., Any]:
    raise TypeError(type(node))


@import_callable.register
def import_from_dependency04(node: CallableFromDepencency) -> Callable[..., Any]:
    module = importlib.import_module(node.module_name)
    c = getattr(module, node.callable_name)
    if not callable(c):
        raise ValueError(f"{node} (imported: {c}) is not callable")

    return c


@import_callable.register
def import_from_dependency05(node: ArchitectureFromLibraryDescr) -> Callable[..., Any]:
    module = importlib.import_module(node.import_from)
    c = getattr(module, node.callable)
    if not callable(c):
        raise ValueError(f"{node} (imported: {c}) is not callable")

    return c


@import_callable.register
def import_from_file04(node: CallableFromFile, **kwargs: Unpack[HashKwargs]):
    return _import_from_file_impl(node.file, node.callable_name, **kwargs)


@import_callable.register
def import_from_file05(node: ArchitectureFromFileDescr, **kwargs: Unpack[HashKwargs]):
    return _import_from_file_impl(node.source, node.callable, sha256=node.sha256)


def _import_from_file_impl(source: FileSource, callable_name: str, **kwargs: Unpack[HashKwargs]):
    local_file = download(source, **kwargs)
    module_name = local_file.path.stem
    importlib_spec = importlib.util.spec_from_file_location(module_name, local_file.path)
    if importlib_spec is None:
        raise ImportError(f"Failed to import {module_name} from {source}.")

    dep = importlib.util.module_from_spec(importlib_spec)
    importlib_spec.loader.exec_module(dep)  # type: ignore  # todo: possible to use "loader.load_module"?
    return getattr(dep, callable_name)
