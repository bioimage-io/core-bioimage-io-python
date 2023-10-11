import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass, replace
from functools import singledispatchmethod
from pathlib import Path, PosixPath, PurePath
from types import ModuleType
from typing import Any, Hashable, List, Optional, Tuple, TypedDict, Union

from pydantic import AnyUrl, DirectoryPath
from pydantic.fields import FieldInfo
from typing_extensions import NotRequired

from bioimageio.spec._internal.base_nodes import Node
from bioimageio.spec._internal.constants import ALERT_TYPE, IN_PACKAGE_MESSAGE, KW_ONLY, SLOTS
from bioimageio.spec.description import ResourceDescription
from bioimageio.spec.summary import ErrorEntry, Loc, WarningEntry


class VisitorKwargs(TypedDict):
    info: NotRequired[FieldInfo]


@dataclass(frozen=True, **SLOTS, **KW_ONLY)
class Note:
    loc: Loc = ()
    info: Optional[FieldInfo] = None


class ValidationVisitor:
    def __init__(self) -> None:
        super().__init__()
        self.errors: List[ErrorEntry] = []
        self.warnings: List[WarningEntry] = []

    @singledispatchmethod
    def visit(self, obj: type, /, note: Note = Note()):
        pass

    @visit.register
    def _visit_node(self, node: Node, note: Note = Note()):
        for k, v in node:
            self.visit(v, replace(note, loc=note.loc + (k,), info=node.model_fields[k]))

    @visit.register
    def _visit_list(self, lst: list, note: Note = Note()):  # type: ignore
        for i, e in enumerate(lst):  # type: ignore
            self.visit(e, replace(note, loc=note.loc + (i,)))

    @visit.register
    def _visit_tuple(self, tup: tuple, note: Note = Note()):  # type: ignore
        for i, e in enumerate(tup):  # type: ignore
            self.visit(e, replace(note, loc=note.loc + (i,)))

    @visit.register
    def _visit_dict(self, dict_: dict, note: Note = Note()):  # type: ignore
        for k, v in dict_.items():  # type: ignore
            self.visit(v, replace(note, loc=note.loc + (k,)))


class SourceValidator(ValidationVisitor):
    def __init__(self, root: Union[DirectoryPath, AnyUrl]) -> None:
        super().__init__()
        self.root = root

    def _visit_path(self, path: PurePath, note: Note):
        if not Path(path).exists():
            msg = f"{path} not found"
            if (
                note.info
                and isinstance(note.info.description, str)
                and note.info.description.startswith(IN_PACKAGE_MESSAGE)
            ):
                self.errors.append(ErrorEntry(loc=note.loc, msg=msg, type="file-not-found"))
            else:
                self.warnings.append(WarningEntry(loc=note.loc, msg=msg, type="file-not-found"))


#             # info.description.startswith(IN_PACKAGE_MESSAGE)
#         if not source_available(leaf, self.root_path):
#             raise FileNotFoundError(leaf)

#     def visit_URI(self, node: raw_nodes.URI):
#         self._visit_source(node)

#     def visit_PosixPath(self, leaf: PosixPath):
#         self._visit_source(leaf)

#     def visit_WindowsPath(self, leaf: pathlib.WindowsPath):
#         self._visit_source(leaf)

#     def generic_visit(self, node):
#         """Called if no explicit visitor function exists for a node."""

#         if isinstance(node, raw_nodes.RawNode):
#             for field, value in iter_fields(node):
#                 if field != "root_path":  # do not visit root_path, as it might be an incomplete (non-available) URL
#                     self.visit(value)
#         else:
#             super().generic_visit(node)


# def get_sha256(path: os.PathLike) -> str:
#     """from https://stackoverflow.com/a/44873382"""
#     h = hashlib.sha256()
#     b = bytearray(128 * 1024)
#     mv = memoryview(b)
#     with open(path, "rb", buffering=0) as f:
#         for n in iter(lambda: f.readinto(mv), 0):
#             h.update(mv[:n])

#     return h.hexdigest()


# class Sha256NodeChecker(NodeVisitor):
#     """Check integrity of the source-like field for every sha256-like field encountered"""

#     def __init__(self, *, root_path: os.PathLike):
#         self.root_path = root_path if isinstance(root_path, raw_nodes.URI) else pathlib.Path(root_path).resolve()

#     def generic_visit(self, node):
#         if isinstance(node, raw_nodes.RawNode):
#             for sha_field, expected in ((k, v) for (k, v) in iter_fields(node) if "sha256" in k and v is not missing):
#                 if sha_field == "sha256":
#                     source_name = "source"
#                     if not hasattr(node, "source") and hasattr(node, "uri"):
#                         source_name = "uri"

#                 elif sha_field.endswith("_sha256"):
#                     source_name = sha_field[: -len("_sha256")]
#                 else:
#                     raise NotImplementedError(f"Don't know how to check integrity with {sha_field}")

#                 if not hasattr(node, source_name):
#                     raise ValueError(
#                         f"Node {node} expected to have '{source_name}' field associated with '{sha_field}'"
#                     )

#                 source_node = getattr(node, source_name)
#                 if isinstance(source_node, ImportedSource):
#                     continue  # test is run after loading. Warning issued in resource_tests._test_resource_integrity

#                 source = get_resolved_source_path(source_node, root_path=self.root_path)
#                 actual = get_sha256(source)

#                 if not isinstance(expected, str):
#                     raise TypeError(f"Expected '{sha_field}' to hold string, not {type(expected)}")

#                 if actual != expected:
#                     if actual[:6] != expected[:6]:
#                         actual = actual[:6] + "..."
#                         expected = expected[:6] + "..."

#                     raise ValueError(
#                         f"Determined {actual} for {source_name}={source}, but expected {sha_field}={expected}"
#                     )

#         super().generic_visit(node)


# class SourceNodeTransformer(NodeTransformer):
#     """
#     Imports all source callables
#     note: Requires previous transformation by UriNodeTransformer
#     """

#     class TemporaryInsertionIntoPythonPath:
#         def __init__(self, path: str):
#             self.path = path

#         def __enter__(self):
#             sys.path.insert(0, self.path)

#         def __exit__(self, exc_type, exc_value, traceback):
#             sys.path.remove(self.path)

#     def transform_LocalImportableModule(self, node: raw_nodes.LocalImportableModule) -> nodes.ImportedSource:
#         with self.TemporaryInsertionIntoPythonPath(str(node.root_path)):
#             module = importlib.import_module(node.module_name)

#         return nodes.ImportedSource(factory=getattr(module, node.callable_name))

#     @staticmethod
#     def transform_ResolvedImportableSourceFile(node: raw_nodes.ResolvedImportableSourceFile) -> nodes.ImportedSource:
#         module_path = resolve_source(node.source_file)
#         module_name = f"module_from_source.{module_path.stem}"
#         importlib_spec = importlib.util.spec_from_file_location(module_name, module_path)
#         assert importlib_spec is not None
#         dep = importlib.util.module_from_spec(importlib_spec)
#         importlib_spec.loader.exec_module(dep)  # type: ignore  # todo: possible to use "loader.load_module"?
#         return nodes.ImportedSource(factory=getattr(dep, node.callable_name))


# class RawNodeTypeTransformer(NodeTransformer):
#     def __init__(self, nodes_module: ModuleType):
#         super().__init__()
#         self.nodes = nodes_module

#     def generic_transformer(self, node: GenericRawNode) -> GenericResolvedNode:
#         if isinstance(node, raw_nodes.RawNode):
#             resolved_data = {
#                 field.name: self.transform(getattr(node, field.name)) for field in dataclasses.fields(node)
#             }
#             resolved_node_type: typing.Type[GenericResolvedNode] = getattr(self.nodes, node.__class__.__name__)
#             return resolved_node_type(**resolved_data)  # type: ignore
#         else:
#             return super().generic_transformer(node)


# def all_sources_available(
#     node: typing.Union[GenericNode, list, tuple, dict], root_path: os.PathLike = pathlib.Path()
# ) -> bool:
#     try:
#         SourceNodeChecker(root_path=root_path).visit(node)
#     except FileNotFoundError:
#         return False
#     else:
#         return True
