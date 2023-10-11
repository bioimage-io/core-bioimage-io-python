import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass, replace
from functools import singledispatchmethod
from pathlib import Path, PosixPath, PurePath
from typing import Any, Hashable, List, Optional, Tuple, Type, TypedDict, Union

import requests
from pydantic import AnyUrl, DirectoryPath
from pydantic.fields import FieldInfo
from typing_extensions import NotRequired

from bioimageio.core.utils import get_sha256
from bioimageio.spec._internal.base_nodes import Node
from bioimageio.spec._internal.constants import IN_PACKAGE_MESSAGE, KW_ONLY, SLOTS
from bioimageio.spec._internal.types import Sha256
from bioimageio.spec.summary import ErrorEntry, Loc, WarningEntry


class VisitorKwargs(TypedDict):
    info: NotRequired[FieldInfo]


@dataclass(frozen=True, **SLOTS, **KW_ONLY)
class Memo:
    loc: Loc = ()
    info: Optional[FieldInfo] = None
    parent_nodes: Tuple[Node, ...] = ()


class ValidationVisitor:
    def __init__(self) -> None:
        super().__init__()
        self.errors: List[ErrorEntry] = []
        self.warnings: List[WarningEntry] = []

    def visit(self, obj: Any, /, memo: Memo = Memo()):
        self._traverse(obj, memo=memo)

    @singledispatchmethod
    def _traverse(self, obj: type, /, memo: Memo):
        pass

    @_traverse.register
    def _traverse_node(self, node: Node, memo: Memo):
        for k, v in node:
            self.visit(
                v,
                replace(memo, loc=memo.loc + (k,), info=node.model_fields[k], parent_nodes=memo.parent_nodes + (node,)),
            )

    @_traverse.register
    def _traverse_list(self, lst: list, memo: Memo):  # type: ignore
        e: Any
        for i, e in enumerate(lst):  # type: ignore
            self.visit(e, replace(memo, loc=memo.loc + (i,)))

    @_traverse.register
    def _traverse_tuple(self, tup: tuple, memo: Memo):  # type: ignore
        e: Any
        for i, e in enumerate(tup):  # type: ignore
            self.visit(e, replace(memo, loc=memo.loc + (i,)))

    @_traverse.register
    def _traverse_dict(self, dict_: dict, memo: Memo):  # type: ignore
        v: Any
        for k, v in dict_.items():  # type: ignore
            self.visit(v, replace(memo, loc=memo.loc + (k,)))


class _NoSha:
    pass


class SourceValidator(ValidationVisitor):
    def __init__(self, root: Union[DirectoryPath, AnyUrl]) -> None:
        super().__init__()
        self.root = root

    def visit(self, obj: Any, /, memo: Memo = Memo()):
        self._visit_impl(obj, memo=memo)
        return super().visit(obj, memo)

    @singledispatchmethod
    def _visit_impl(self, obj: type, /, memo: Memo):
        pass

    @_visit_impl.register
    def _visit_path(self, path: PurePath, memo: Memo):
        if Path(path).exists():
            sha256: Union[None, Sha256, Type[_NoSha]] = _NoSha

            for parent in memo.parent_nodes:
                if "sha256" in parent.model_fields:
                    sha256: Optional[Sha256] = parent.sha256  # type: ignore
                    break

            if sha256 is _NoSha:
                return

            actual_sha256 = get_sha256(path)
            if sha256 is None:
                self.warnings.append(
                    WarningEntry(
                        loc=memo.loc,
                        msg=(
                            f"Cannot validate file integrity (`sha256` not specified). "
                            f"File {path} has SHA-256: {actual_sha256}"
                        ),
                        type="unknown_hash",
                    )
                )
            elif actual_sha256 != sha256:
                self.errors.append(
                    ErrorEntry(
                        loc=memo.loc,
                        msg=f"SHA-256 mismatch: actual ({actual_sha256}) != specified ({sha256})",
                        type="hash_mismatch",
                    )
                )
        else:
            msg = f"{path} not found"
            if (
                memo.info
                and isinstance(memo.info.description, str)
                and memo.info.description.startswith(IN_PACKAGE_MESSAGE)
            ):
                self.errors.append(ErrorEntry(loc=memo.loc, msg=msg, type="file_not_found"))
            else:
                self.warnings.append(WarningEntry(loc=memo.loc, msg=msg, type="file_not_found"))

    @_visit_impl.register
    def _visit_url(self, url: AnyUrl, memo: Memo):
        if url.scheme not in ("http", "https"):
            self.errors.append(ErrorEntry(loc=memo.loc, msg=f"invalid http(s) URL: {url}", type="url_scheme"))
        else:
            response = requests.head(str(url))
            if response.status_code != 200:
                self.errors.append(ErrorEntry(loc=memo.loc, msg=response.reason, type="url_unavailable"))
