import dataclasses
import importlib.util
import os
import pathlib
import sys
import typing
from types import ModuleType

from bioimageio.spec.shared import raw_nodes
from bioimageio.spec.shared.utils import (
    GenericRawNode,
    GenericRawRD,
    GenericResolvedNode,
    NodeTransformer,
    NodeVisitor,
    UriNodeTransformer,
    resolve_source,
    source_available,
)
from . import nodes

GenericNode = typing.Union[GenericRawNode, GenericResolvedNode]


def iter_fields(node: GenericNode):
    for field in dataclasses.fields(node):
        yield field.name, getattr(node, field.name)


class SourceNodeChecker(NodeVisitor):
    """raises FileNotFoundError for unavailable URIs and paths"""

    def __init__(self, *, root_path: os.PathLike):
        self.root_path = pathlib.Path(root_path).resolve()

    def _visit_source(self, leaf: typing.Union[pathlib.Path, raw_nodes.URI]):
        if not source_available(leaf, self.root_path):
            raise FileNotFoundError(leaf)

    def visit_URI(self, node: raw_nodes.URI):
        self._visit_source(node)

    def visit_PosixPath(self, leaf: pathlib.PosixPath):
        self._visit_source(leaf)

    def visit_WindowsPath(self, leaf: pathlib.WindowsPath):
        self._visit_source(leaf)


class SourceNodeTransformer(NodeTransformer):
    """
    Imports all source callables
    note: Requires previous transformation by UriNodeTransformer
    """

    class TemporaryInsertionIntoPythonPath:
        def __init__(self, path: str):
            self.path = path

        def __enter__(self):
            sys.path.insert(0, self.path)

        def __exit__(self, exc_type, exc_value, traceback):
            sys.path.remove(self.path)

    def transform_LocalImportableModule(self, node: raw_nodes.LocalImportableModule) -> nodes.ImportedSource:
        with self.TemporaryInsertionIntoPythonPath(str(node.root_path)):
            module = importlib.import_module(node.module_name)

        return nodes.ImportedSource(factory=getattr(module, node.callable_name))

    @staticmethod
    def transform_ImportableModule(node):
        raise RuntimeError(
            "Encountered raw_nodes.ImportableModule in _SourceNodeTransformer. Apply _UriNodeTransformer first!"
        )

    @staticmethod
    def transform_ResolvedImportableSourceFile(node: raw_nodes.ResolvedImportableSourceFile) -> nodes.ImportedSource:
        module_path = resolve_source(node.source_file)
        module_name = f"module_from_source.{module_path.stem}"
        importlib_spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert importlib_spec is not None
        dep = importlib.util.module_from_spec(importlib_spec)
        importlib_spec.loader.exec_module(dep)  # type: ignore  # todo: possible to use "loader.load_module"?
        return nodes.ImportedSource(factory=getattr(dep, node.callable_name))

    @staticmethod
    def transform_ImportablePath(node):
        raise RuntimeError(
            "Encountered raw_nodes.ImportableSourceFile in _SourceNodeTransformer. Apply _UriNodeTransformer first!"
        )


class RawNodeTypeTransformer(NodeTransformer):
    def __init__(self, nodes_module: ModuleType):
        super().__init__()
        self.nodes = nodes_module

    def generic_transformer(self, node: GenericRawNode) -> GenericResolvedNode:
        if isinstance(node, raw_nodes.RawNode):
            resolved_data = {
                field.name: self.transform(getattr(node, field.name)) for field in dataclasses.fields(node)
            }
            resolved_node_type: typing.Type[GenericResolvedNode] = getattr(self.nodes, node.__class__.__name__)
            return resolved_node_type(**resolved_data)  # type: ignore
        else:
            return super().generic_transformer(node)


def all_sources_available(
    node: typing.Union[GenericNode, list, tuple, dict], root_path: os.PathLike = pathlib.Path()
) -> bool:
    try:
        SourceNodeChecker(root_path=root_path).visit(node)
    except FileNotFoundError:
        return False
    else:
        return True


def resolve_raw_resource_description(raw_rd: GenericRawRD, nodes_module: typing.Any) -> GenericResolvedNode:
    """resolve all uris and sources"""
    rd = UriNodeTransformer(root_path=raw_rd.root_path).transform(raw_rd)
    rd = SourceNodeTransformer().transform(rd)
    rd = RawNodeTypeTransformer(nodes_module).transform(rd)
    return rd
