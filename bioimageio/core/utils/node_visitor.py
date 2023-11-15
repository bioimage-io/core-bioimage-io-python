from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import singledispatchmethod
from pathlib import Path, PurePath
from typing import Any, List, Optional, Tuple, Union

import requests
from pydantic import AnyUrl, DirectoryPath
from pydantic.fields import FieldInfo

from bioimageio.core.utils import get_sha256
from bioimageio.spec._internal.base_nodes import Node
from bioimageio.spec._internal.constants import IN_PACKAGE_MESSAGE, KW_ONLY, SLOTS
from bioimageio.spec._internal.types import Sha256
from bioimageio.spec.summary import ErrorEntry, Loc, WarningEntry


@dataclass(frozen=True, **SLOTS, **KW_ONLY)
class Memo:
    loc: Loc = ()
    info: Optional[FieldInfo] = None
    parent_nodes: Tuple[Node, ...] = ()


class NodeVisitor:
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


class ValidationVisitor(NodeVisitor, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.errors: List[ErrorEntry] = []
        self.warnings: List[WarningEntry] = []

    def visit(self, obj: Any, /, memo: Memo = Memo()):
        self.validate(obj, memo=memo)
        return super().visit(obj, memo)

    @singledispatchmethod
    @abstractmethod
    def validate(self, obj: type, /, memo: Memo):
        ...
