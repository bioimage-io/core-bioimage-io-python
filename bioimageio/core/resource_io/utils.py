import dataclasses
import importlib.util
import logging
import os
import pathlib
import sys
import typing
import warnings
from functools import singledispatch
from types import ModuleType
from urllib.request import url2pathname, urlretrieve

import requests
from marshmallow import ValidationError
from tqdm import tqdm

from bioimageio.spec.shared import fields, raw_nodes
from bioimageio.spec.shared.common import BIOIMAGEIO_CACHE_PATH
from bioimageio.spec.shared.utils import GenericRawNode, GenericRawRD, NodeTransformer, NodeVisitor
from . import nodes

GenericResolvedNode = typing.TypeVar("GenericResolvedNode", bound=nodes.Node)
GenericNode = typing.Union[GenericRawNode, GenericResolvedNode]


def iter_fields(node: GenericNode):
    for field in dataclasses.fields(node):
        yield field.name, getattr(node, field.name)


class UriNodeChecker(NodeVisitor):
    """raises FileNotFoundError for unavailable URIs and paths"""

    def __init__(self, *, root_path: os.PathLike):
        self.root_path = pathlib.Path(root_path).resolve()

    def visit_URI(self, node: raw_nodes.URI):
        if not uri_available(node, self.root_path):
            raise FileNotFoundError(node)

    def _visit_Path(self, leaf: pathlib.Path):
        if not leaf.exists():
            raise FileNotFoundError(leaf)

    def visit_PosixPath(self, leaf: pathlib.PosixPath):
        self._visit_Path(leaf)

    def visit_WindowsPath(self, leaf: pathlib.WindowsPath):
        self._visit_Path(leaf)


class UriNodeTransformer(NodeTransformer):
    def __init__(self, *, root_path: os.PathLike):
        self.root_path = pathlib.Path(root_path).resolve()

    def transform_URI(self, node: raw_nodes.URI) -> pathlib.Path:
        local_path = resolve_uri(node, root_path=self.root_path)
        return local_path

    def transform_ImportableSourceFile(
        self, node: raw_nodes.ImportableSourceFile
    ) -> nodes.ResolvedImportableSourceFile:
        return nodes.ResolvedImportableSourceFile(
            source_file=resolve_uri(node.source_file, self.root_path), callable_name=node.callable_name
        )

    def transform_ImportableModule(self, node: raw_nodes.ImportableModule) -> nodes.LocalImportableModule:
        return nodes.LocalImportableModule(**dataclasses.asdict(node), root_path=self.root_path)

    def _transform_Path(self, leaf: pathlib.Path):
        return self.root_path / leaf

    def transform_PosixPath(self, leaf: pathlib.PosixPath) -> pathlib.Path:
        return self._transform_Path(leaf)

    def transform_WindowsPath(self, leaf: pathlib.WindowsPath) -> pathlib.Path:
        return self._transform_Path(leaf)


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

    def transform_LocalImportableModule(self, node: nodes.LocalImportableModule) -> nodes.ImportedSource:
        with self.TemporaryInsertionIntoPythonPath(str(node.root_path)):
            module = importlib.import_module(node.module_name)

        return nodes.ImportedSource(factory=getattr(module, node.callable_name))

    @staticmethod
    def transform_ImportableModule(node):
        raise RuntimeError(
            "Encountered raw_nodes.ImportableModule in _SourceNodeTransformer. Apply _UriNodeTransformer first!"
        )

    @staticmethod
    def transform_ResolvedImportableSourceFile(node: nodes.ResolvedImportableSourceFile) -> nodes.ImportedSource:
        module_path = resolve_uri(node.source_file)
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


@singledispatch
def resolve_uri(uri, root_path: os.PathLike = pathlib.Path()):
    raise TypeError(type(uri))


@resolve_uri.register
def _resolve_uri_uri_node(uri: raw_nodes.URI, root_path: os.PathLike = pathlib.Path()) -> pathlib.Path:
    assert isinstance(uri, (raw_nodes.URI, nodes.URI))
    path_or_remote_uri = resolve_local_uri(uri, root_path)
    if isinstance(path_or_remote_uri, raw_nodes.URI):
        local_path = _download_uri_to_local_path(path_or_remote_uri)
    elif isinstance(path_or_remote_uri, pathlib.Path):
        local_path = path_or_remote_uri
    else:
        raise TypeError(path_or_remote_uri)

    return local_path


@resolve_uri.register
def _resolve_uri_str(uri: str, root_path: os.PathLike = pathlib.Path()) -> pathlib.Path:
    return resolve_uri(fields.URI().deserialize(uri), root_path)


@resolve_uri.register
def _resolve_uri_path(uri: pathlib.Path, root_path: os.PathLike = pathlib.Path()) -> pathlib.Path:
    if not uri.is_absolute():
        uri = pathlib.Path(root_path).absolute() / uri

    return resolve_uri(uri.as_uri(), root_path)


@resolve_uri.register
def _resolve_uri_resolved_importable_path(
    uri: nodes.ResolvedImportableSourceFile, root_path: os.PathLike = pathlib.Path()
) -> nodes.ResolvedImportableSourceFile:
    return nodes.ResolvedImportableSourceFile(
        callable_name=uri.callable_name, source_file=resolve_uri(uri.source_file, root_path)
    )


@resolve_uri.register
def _resolve_uri_importable_path(
    uri: raw_nodes.ImportableSourceFile, root_path: os.PathLike = pathlib.Path()
) -> nodes.ResolvedImportableSourceFile:
    return nodes.ResolvedImportableSourceFile(
        callable_name=uri.callable_name, source_file=resolve_uri(uri.source_file, root_path)
    )


@resolve_uri.register
def _resolve_uri_list(uri: list, root_path: os.PathLike = pathlib.Path()) -> typing.List[pathlib.Path]:
    return [resolve_uri(el, root_path) for el in uri]


def resolve_local_uri(
    uri: typing.Union[str, os.PathLike, raw_nodes.URI], root_path: os.PathLike
) -> typing.Union[pathlib.Path, raw_nodes.URI]:
    if isinstance(uri, os.PathLike) or isinstance(uri, str):
        if isinstance(uri, str):
            try:  # uri as relative path from cwd
                is_path_cwd = pathlib.Path(uri).exists()
            except OSError:
                is_path_cwd = False

            try:  # uri as relative path from root_path
                path_from_root = pathlib.Path(root_path) / uri
                is_path_rp = (path_from_root).exists()
            except OSError:
                is_path_rp = False
            else:
                if not is_path_cwd and is_path_rp:
                    uri = path_from_root

            is_path = is_path_cwd or is_path_rp
        else:
            is_path = True

        if is_path:
            return pathlib.Path(uri)

    if isinstance(uri, str):
        uri = fields.URI().deserialize(uri)

    assert isinstance(uri, raw_nodes.URI), uri
    if not uri.scheme:  # relative path
        if uri.authority or uri.query or uri.fragment:
            raise ValidationError(f"Invalid Path/URI: {uri}")

        local_path_or_remote_uri: typing.Union[pathlib.Path, raw_nodes.URI] = pathlib.Path(root_path) / uri.path
    elif uri.scheme == "file":
        if uri.authority or uri.query or uri.fragment:
            raise NotImplementedError(uri)

        local_path_or_remote_uri = pathlib.Path(url2pathname(uri.path))
    elif uri.scheme in ("https", "https"):
        local_path_or_remote_uri = uri
    else:
        raise ValueError(f"Unknown uri scheme {uri.scheme}")

    return local_path_or_remote_uri


def uri_available(uri: raw_nodes.URI, root_path: pathlib.Path) -> bool:
    local_path_or_remote_uri = resolve_local_uri(uri, root_path)
    if isinstance(local_path_or_remote_uri, raw_nodes.URI):
        response = requests.head(str(local_path_or_remote_uri))
        available = response.status_code == 200
    elif isinstance(local_path_or_remote_uri, pathlib.Path):
        available = local_path_or_remote_uri.exists()
    else:
        raise TypeError(local_path_or_remote_uri)

    return available


def all_uris_available(
    node: typing.Union[GenericNode, list, tuple, dict], root_path: os.PathLike = pathlib.Path()
) -> bool:
    try:
        UriNodeChecker(root_path=root_path).visit(node)
    except FileNotFoundError:
        return False
    else:
        return True


def download_uri_to_local_path(uri: typing.Union[raw_nodes.URI, str]) -> pathlib.Path:
    return resolve_uri(uri)


def _download_uri_to_local_path(uri: raw_nodes.URI) -> pathlib.Path:
    local_path = BIOIMAGEIO_CACHE_PATH / uri.scheme / uri.authority / uri.path.strip("/") / uri.query
    if local_path.exists():
        warnings.warn(f"found cached {local_path}. Skipping download of {uri}.")
    else:
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # download with tqdm adapted from:
            # https://github.com/shaypal5/tqdl/blob/189f7fd07f265d29af796bee28e0893e1396d237/tqdl/core.py
            # Streaming, so we can iterate over the response.
            r = requests.get(str(uri), stream=True)
            # Total size in bytes.
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            t = tqdm(total=total_size, unit="iB", unit_scale=True, desc=local_path.name)
            with local_path.open("wb") as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                raise RuntimeError("Download does not have expected size.")
        except Exception as e:
            raise RuntimeError(f"Failed to download {uri} ({e})")

    return local_path


def resolve_raw_resource_description(raw_rd: GenericRawRD, nodes_module: typing.Any) -> GenericResolvedNode:
    """resolve all uris and sources"""
    rd = UriNodeTransformer(root_path=raw_rd.root_path).transform(raw_rd)
    rd = SourceNodeTransformer().transform(rd)
    rd = RawNodeTypeTransformer(nodes_module).transform(rd)
    return rd
