import dataclasses
import importlib.util
import os
import pathlib
import sys
import typing
from types import ModuleType

from marshmallow import missing

from bioimageio.spec.shared import raw_nodes, resolve_source, source_available
from bioimageio.spec.shared.node_transformer import (
    GenericRawNode,
    GenericResolvedNode,
    NodeTransformer,
    NodeVisitor,
    UriNodeTransformer,
)
from . import nodes

GenericNode = typing.Union[GenericRawNode, GenericResolvedNode]


def iter_fields(node: GenericNode):
    for field in dataclasses.fields(node):
        yield field.name, getattr(node, field.name)


class SourceNodeChecker(NodeVisitor):
    """raises FileNotFoundError for unavailable URIs and paths"""

    def __init__(self, *, root_path: os.PathLike):
        self.root_path = root_path if isinstance(root_path, raw_nodes.URI) else pathlib.Path(root_path).resolve()

    def _visit_source(self, leaf: typing.Union[pathlib.Path, raw_nodes.URI]):
        if not source_available(leaf, self.root_path):
            raise FileNotFoundError(leaf)

    def visit_URI(self, node: raw_nodes.URI):
        self._visit_source(node)

    def visit_PosixPath(self, leaf: pathlib.PosixPath):
        self._visit_source(leaf)

    def visit_WindowsPath(self, leaf: pathlib.WindowsPath):
        self._visit_source(leaf)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""

        if isinstance(node, raw_nodes.RawNode):
            for field, value in iter_fields(node):
                if field != "root_path":  # do not visit root_path, as it might be an incomplete (non-available) URL
                    self.visit(value)
        else:
            super().generic_visit(node)


class CallableNodeTransformer(NodeTransformer):
    """
    Import all callables
    note: Requires previous transformation by UriNodeTransformer
    """

    class TemporaryInsertionIntoPythonPath:
        def __init__(self, path: str):
            self.path = path

        def __enter__(self):
            sys.path.insert(0, self.path)

        def __exit__(self, exc_type, exc_value, traceback):
            sys.path.remove(self.path)

    def transform_LocalCallableFromModule(self, node: raw_nodes.LocalCallableFromModule) -> nodes.ImportedCallable:
        with self.TemporaryInsertionIntoPythonPath(str(node.root_path)):
            module = importlib.import_module(node.module_name)

        return nodes.ImportedCallable(call=getattr(module, node.callable_name))

    @staticmethod
    def transform_ResolvedCallableFromSourceFile(
        node: raw_nodes.ResolvedCallableFromSourceFile,
    ) -> nodes.ImportedCallable:
        module_path = resolve_source(node.source_file)
        module_name = f"module_from_source.{module_path.stem}"
        importlib_spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert importlib_spec is not None
        dep = importlib.util.module_from_spec(importlib_spec)
        importlib_spec.loader.exec_module(dep)  # type: ignore  # todo: possible to use "loader.load_module"?
        return nodes.ImportedCallable(call=getattr(dep, node.callable_name))


class RawNodeTypeTransformer(NodeTransformer):
    def __init__(self, nodes_module: ModuleType):
        super().__init__()
        self.nodes = nodes_module

    def generic_transformer(self, node: GenericRawNode) -> GenericResolvedNode:
        if isinstance(node, raw_nodes.RawNode):
            resolved_data = {
                field.name: self.transform(getattr(node, field.name))
                for field in dataclasses.fields(node)
                if getattr(node, field.name) is not missing  # exclude missing fields to respect for node defaults
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


def resolve_raw_node(
    raw_rd: GenericRawNode,
    nodes_module: typing.Any,
    uri_only_if_in_package: bool = True,
    root_path: typing.Optional[pathlib.Path] = None,
) -> GenericResolvedNode:
    """resolve all uris and paths (that are included when packaging)"""
    rd = UriNodeTransformer(
        root_path=root_path or raw_rd.root_path, uri_only_if_in_package=uri_only_if_in_package
    ).transform(raw_rd)
    rd = CallableNodeTransformer().transform(rd)
    rd = RawNodeTypeTransformer(nodes_module).transform(rd)
    return rd
