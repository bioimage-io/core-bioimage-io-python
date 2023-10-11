# todo: cleanup __init__: move stuff to util submodules or elsewhere
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from contextlib import AbstractContextManager
from functools import singledispatch
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import AnyUrl, HttpUrl

from bioimageio.core.io import FileSource, download
from bioimageio.spec.model.v0_4 import CallableFromDepencency
from bioimageio.spec.model.v0_4 import CallableFromFile as CallableFromFile04
from bioimageio.spec.model.v0_5 import CallableFromFile as CallableFromFile05
from bioimageio.spec.model.v0_5 import Sha256

if sys.version_info < (3, 9):

    def files(package_name: str):
        assert package_name == "bioimageio.core"
        return Path(__file__).parent.parent

else:
    from importlib.resources import files as files


def get_parent_url(url: HttpUrl) -> HttpUrl:
    parsed = urlsplit(str(url))
    return AnyUrl(
        urlunsplit((parsed.scheme, parsed.netloc, "/".join(parsed.path.split("/")[:-1]), parsed.query, parsed.fragment))
    )


def get_sha256(path: os.PathLike[str]) -> str:
    """from https://stackoverflow.com/a/44873382"""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(path, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])

    return h.hexdigest()


class TemporaryInsertionIntoPythonPath(AbstractContextManager[None]):
    def __init__(self, path: Path):
        super().__init__()
        self.path = str(path)

    def __enter__(self):
        super().__enter__()
        sys.path.insert(0, self.path)

    def __exit__(
        self,
        __exc_type: "type[BaseException] | None",
        __exc_value: "BaseException | None",
        __traceback: "TracebackType | None",
    ) -> "bool | None":
        assert sys.path[0] == self.path
        _ = sys.path.pop(0)
        return super().__exit__(__exc_type, __exc_value, __traceback)


@singledispatch
def import_callable(node: type, /) -> Callable[..., Any]:
    raise TypeError(type(node))


@import_callable.register
def import_from_dependency(node: CallableFromDepencency) -> Callable[..., Any]:
    module = importlib.import_module(node.module_name)
    c = getattr(module, node.callable_name)
    if not callable(c):
        raise ValueError(f"{node} (imported: {c}) is not callable")

    return c


@import_callable.register
def import_from_file04(node: CallableFromFile04, sha256: Optional[Sha256] = None):
    return _import_from_file_impl(node.file, node.callable_name, sha256)


@import_callable.register
def import_from_file05(node: CallableFromFile05, sha256: Optional[Sha256] = None):
    return _import_from_file_impl(node.source_file, node.callable_name, sha256)


def _import_from_file_impl(source: FileSource, callable_name: str, sha256: Optional[Sha256]):
    local_file = download(source, known_hash=None if sha256 is None else f"sha256:{sha256}")
    module_name = local_file.path.stem
    importlib_spec = importlib.util.spec_from_file_location(module_name, local_file.path)
    if importlib_spec is None:
        raise ImportError(f"Failed to import {module_name} from {source}.")

    dep = importlib.util.module_from_spec(importlib_spec)
    importlib_spec.loader.exec_module(dep)  # type: ignore  # todo: possible to use "loader.load_module"?
    return getattr(dep, callable_name)
