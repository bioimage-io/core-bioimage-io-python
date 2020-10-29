import dataclasses
import importlib.util
import logging
import os
import pathlib
import subprocess
import sys
import uuid
from dataclasses import fields
from typing import Any, Callable, Dict, TypeVar, Union
from urllib.parse import ParseResult, urlunparse
from urllib.request import url2pathname, urlretrieve

import yaml

from pybio.core.cache import cache_path
from . import nodes, schema
from .exceptions import InvalidDoiException, PyBioValidationException


def iter_fields(node: nodes.Node):
    for field in fields(node):
        yield field.name, getattr(node, field.name)


class NodeVisitor:
    def visit(self, node: Any) -> None:
        method = "visit_" + node.__class__.__name__

        visitor = getattr(self, method, self.generic_visit)

        visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        if isinstance(node, nodes.Node):
            for field, value in iter_fields(node):
                self.visit(value)
        elif isinstance(node, list):
            [self.visit(subnode) for subnode in node]
        elif isinstance(node, dict):
            [self.visit(subnode) for subnode in node.values()]
        elif isinstance(node, tuple):
            assert not any(
                isinstance(subnode, nodes.Node) or isinstance(subnode, list) or isinstance(subnode, dict)
                for subnode in node
            )


GenericNode = TypeVar("GenericNode")


class NodeTransformer:
    def transform(self, node: GenericNode) -> GenericNode:
        method = "transform_" + node.__class__.__name__

        transformer = getattr(self, method, self.generic_transformer)

        return transformer(node)

    def generic_transformer(self, node: Any) -> Any:
        if isinstance(node, nodes.Node):
            return dataclasses.replace(
                node, **{field.name: self.transform(getattr(node, field.name)) for field in fields(node)}
            )
        else:
            return node

    def transform_list(self, node: list) -> list:
        return [self.transform(subnode) for subnode in node]


@dataclasses.dataclass
class ImportedSource:
    factory: Callable

    def __call__(self, *args, **kwargs):
        return self.factory(*args, **kwargs)


def get_instance(node: Union[nodes.SpecWithKwargs, nodes.BaseSpec, nodes.WithImportableSource], **kwargs):
    if isinstance(node, nodes.SpecWithKwargs):
        joined_spec_kwargs = dict(node.kwargs)
        joined_spec_kwargs.update(kwargs)
        return get_instance(node.spec, **joined_spec_kwargs)
    elif isinstance(node, nodes.ModelSpec):
        return get_instance(node.model, **kwargs)
    elif isinstance(node, nodes.WithImportableSource):
        if not isinstance(node.source, ImportedSource):
            raise ValueError(
                f"Encountered unexpected node.source type {type(node.source)}. `get_instance` requires URITransformer and SourceTransformer to be applied beforehand."
            )

        joined_kwargs = dict(node.kwargs)
        joined_kwargs.update(kwargs)
        return node.source.factory(**joined_kwargs)
    else:
        raise TypeError(node)


def train(model: nodes.Model, **kwargs) -> Any:
    raise NotImplementedError
    # resolve magic kwargs
    available_magic_kwargs = {"pybio_model": model}
    enchanted_kwargs = {
        req: available_magic_kwargs[req] for req in model.spec.config["pybio"]["training"].get("required_kwargs", {})
    }
    enchanted_kwargs.update(kwargs)

    return get_instance(model.spec.config["pybio"]["training"], **enchanted_kwargs)


def resolve_uri(uri_node: nodes.URI, cache_path: pathlib.Path, root_path: pathlib.Path) -> pathlib.Path:
    if uri_node.scheme == "":  # relative path
        if uri_node.netloc:
            raise PyBioValidationException(f"Invalid Path/URI: {uri_node}")

        local_path = root_path / uri_node.path
    elif uri_node.scheme == "file":
        if uri_node.netloc or uri_node.query:
            raise NotImplementedError(uri_node)

        local_path = pathlib.Path(url2pathname(uri_node.path))
    elif uri_node.netloc == "github.com":
        orga, repo_name, blob_releases_archive, commit_id, *in_repo_path = uri_node.path.strip("/").split("/")
        if blob_releases_archive == "releases":
            local_path = _download_uri_node_to_local_path(uri_node, cache_path)
        elif blob_releases_archive == "archive":
            raise NotImplementedError("unpacking of github archive not implemented")
            # local_path = _download_uri_node_to_local_path(uri_node, cache_path)
        elif blob_releases_archive == "blob":
            in_repo_path = "/".join(in_repo_path)
            cached_repo_path = cache_path / orga / repo_name / commit_id
            local_path = cached_repo_path / in_repo_path
            if not local_path.exists():
                cached_repo_path = str(cached_repo_path.resolve())
                subprocess.call(
                    ["git", "clone", f"{uri_node.scheme}://{uri_node.netloc}/{orga}/{repo_name}.git", cached_repo_path]
                )
                # -C <working_dir> available in git 1.8.5+
                # https://github.com/git/git/blob/5fd09df3937f54c5cfda4f1087f5d99433cce527/Documentation/RelNotes/1.8.5.txt#L115-L116
                subprocess.call(["git", "-C", cached_repo_path, "checkout", "--force", commit_id])
        else:
            raise NotImplementedError(f"unkown github url format: {uri_node} with '{blob_releases_archive}'")
    elif uri_node.scheme == "https":
        local_path = _download_uri_node_to_local_path(uri_node, cache_path)
    else:
        raise ValueError(f"Unknown uri scheme {uri_node.scheme}")

    return local_path


def _download_uri_node_to_local_path(uri_node: nodes.URI, cache_path: pathlib.Path) -> pathlib.Path:
    local_path = cache_path / uri_node.scheme / uri_node.netloc / uri_node.path.strip("/") / uri_node.query
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url_str = urlunparse([uri_node.scheme, uri_node.netloc, uri_node.path, "", uri_node.query, ""])
        try:
            urlretrieve(url_str, str(local_path))
        except Exception:
            logging.getLogger("download").error("Failed to download %s", uri_node)
            raise

    return local_path


def resolve_doi(uri: ParseResult) -> ParseResult:
    if uri.scheme.lower() != "doi" and not uri.netloc.endswith("doi.org"):
        return uri

    doi = uri.path.strip("/")

    url = "https://doi.org/api/handles/" + doi
    r = requests.get(url).json()
    response_code = r["responseCode"]
    if response_code != 1:
        raise InvalidDoiException(f"Could not resolve doi {doi} (responseCode={response_code})")

    val = min(r["values"], key=lambda v: v["index"])

    assert val["type"] == "URL"  # todo: handle other types
    assert val["data"]["format"] == "string"
    return urlparse(val["data"]["value"])


@dataclasses.dataclass
class LocalImportableModule(nodes.ImportableModule):
    python_path: pathlib.Path


@dataclasses.dataclass
class ResolvedImportablePath(nodes.ImportablePath):
    pass


class URITransformer(NodeTransformer):
    def __init__(self, root_path: pathlib.Path, cache_path: pathlib.Path):
        self.root_path = root_path
        self.cache_path = cache_path
        self.python_path = self._guess_python_path_from_local_spec_path(root_path=root_path)

    def transform_SpecURI(self, node: nodes.SpecURI) -> Any:
        local_path = resolve_uri(node, root_path=self.root_path, cache_path=self.cache_path)
        with local_path.open() as f:
            data = yaml.safe_load(f)

        resolved_node = node.spec_schema.load(data)
        subtransformer = self.__class__(root_path=local_path.parent, cache_path=self.cache_path)
        return subtransformer.transform(resolved_node)

    def transform_URI(self, node: nodes.URI) -> pathlib.Path:
        local_path = resolve_uri(node, root_path=self.root_path, cache_path=self.cache_path)
        return local_path
        # if local_path.is_dir():
        #     return local_path
        # else:
        #     with local_path.open(mode="rb") as f:
        #         return io.BytesIO(f.read())

    def transform_ImportablePath(self, node: nodes.ImportablePath) -> ResolvedImportablePath:
        return ResolvedImportablePath(
            filepath=(self.root_path / node.filepath).resolve(), callable_name=node.callable_name
        )

    def transform_ImportableModule(self, node: nodes.ImportableModule) -> LocalImportableModule:
        return LocalImportableModule(**dataclasses.asdict(node), python_path=self.python_path)

    def _transform_Path(self, leaf: pathlib.Path):
        assert not leaf.is_absolute()
        return self.root_path / leaf

    def transform_PosixPath(self, leaf: pathlib.PosixPath) -> pathlib.Path:
        return self._transform_Path(leaf)

    def transform_WindowsPath(self, leaf: pathlib.WindowsPath) -> pathlib.Path:
        return self._transform_Path(leaf)

    def _guess_python_path_from_local_spec_path(self, root_path: pathlib.Path):
        def potential_paths():
            yield root_path
            yield from root_path.parents

        for path in potential_paths():
            if (path / "manifest.yaml").exists() or (path / "manifest.yml").exists():
                return path.resolve()

        return root_path


class SourceTransformer(NodeTransformer):
    """
    Imports all source callables
    note: Requires previous transformation by URITransformer
    """

    class TemporaryInsertionIntoPythonPath:
        def __init__(self, path: str):
            self.path = path

        def __enter__(self):
            sys.path.insert(0, self.path)

        def __exit__(self, exc_type, exc_value, traceback):
            sys.path.remove(self.path)

    def transform_LocalImportableModule(self, node: LocalImportableModule) -> ImportedSource:
        with self.TemporaryInsertionIntoPythonPath(str(node.python_path)):
            module = importlib.import_module(node.module_name)

        return ImportedSource(factory=getattr(module, node.callable_name))

    def transform_ImportableModule(self, node):
        raise RuntimeError("Encountered nodes.ImportableModule in SourceTransformer. Apply URITransformer first!")

    def transform_ResolvedImportablePath(self, node: ResolvedImportablePath) -> ImportedSource:
        importlib_spec = importlib.util.spec_from_file_location(f"user_imports.{uuid.uuid4().hex}", node.filepath)
        dep = importlib.util.module_from_spec(importlib_spec)
        importlib_spec.loader.exec_module(dep)
        return ImportedSource(factory=getattr(dep, node.callable_name))

    def transform_ImportablePath(self, node):
        raise RuntimeError("Encountered nodes.ImportablePath in SourceTransformer. Apply URITransformer first!")


class MagicKwargsTransformer(NodeTransformer):
    def __init__(self, **magic_kwargs):
        super().__init__()
        self.magic_kwargs = magic_kwargs

    def transform_Kwargs(self, node: nodes.Kwargs) -> nodes.Kwargs:
        for key, value in self.magic_kwargs.items():
            if key in node and node[key] is None:
                node[key] = value

        return node


def load_spec_and_kwargs(
    uri: str,
    kwargs: Dict[str, Any] = None,
    *,
    root_path: pathlib.Path = pathlib.Path("."),
    cache_path: pathlib.Path = pathlib.Path(os.getenv("PYBIO_CACHE_PATH", "pybio_cache")),
    **spec_kwargs,
) -> nodes.Model:
    cache_path = cache_path.resolve()
    root_path = root_path.resolve()
    assert root_path.exists(), root_path

    data = {"spec": uri, "kwargs": kwargs or {}, **spec_kwargs}
    last_dot = uri.rfind(".")
    second_last_dot = uri[:last_dot].rfind(".")
    spec_suffix = uri[second_last_dot + 1 : last_dot]
    if spec_suffix == "model":
        tree = schema.Model().load(data)
    elif spec_suffix in ("transformation", "reader", "sampler"):
        raise NotImplementedError(spec_suffix)
    else:
        raise ValueError(f"Invalid spec suffix: {spec_suffix}")

    local_spec_path = resolve_uri(uri_node=tree.spec, root_path=root_path, cache_path=cache_path)

    tree = URITransformer(root_path=local_spec_path.parent, cache_path=cache_path).transform(tree)
    tree = SourceTransformer().transform(tree)
    tree = MagicKwargsTransformer(cache_path=cache_path).transform(tree)
    return tree


def load_model_config(uri: Union[pathlib.Path, str]) -> nodes.Model:
    if isinstance(uri, pathlib.Path):
        uri = uri.as_uri()

    ret = load_spec_and_kwargs(uri)
    assert isinstance(ret, nodes.Model)
    return ret


def cache_uri(uri_str: str, sha256: str) -> pathlib.Path:
    file_node = schema.File().load({"source": uri_str, "sha256": sha256})
    uri_transformer = URITransformer(root_path=cache_path, cache_path=cache_path)
    file_node = uri_transformer.transform(file_node)
    file_node = SourceTransformer().transform(file_node)
    # todo: check hash
    return file_node.source
