import dataclasses
import importlib.util
import io
import pathlib
import subprocess
import sys
import uuid
from dataclasses import fields
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from urllib.parse import ParseResult, urlunparse
from urllib.request import urlretrieve

import yaml

from . import nodes, schema
from .exceptions import (
    InvalidDoiException,
    PyBioMissingKwargException,
    PyBioTransformationOrderException,
    PyBioValidationException,
)


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
    callable_: Callable

    def __call__(self, *args, **kwargs):
        return self.callable_(*args, **kwargs)


def get_instance(node: Union[nodes.SpecWithKwargs, nodes.WithImportableSource], **kwargs):
    if isinstance(node, nodes.SpecWithKwargs):
        joined_spec_kwargs = dict(node.kwargs)
        joined_spec_kwargs.update(kwargs)
        if isinstance(node, nodes.Reader):
            if node.transformations:
                assert "transformations" not in joined_spec_kwargs
                joined_spec_kwargs["transformations"] = [get_instance(t) for t in node.transformations]

        if isinstance(node, nodes.Sampler):
            if node.readers:
                assert "readers" not in joined_spec_kwargs
                joined_spec_kwargs["readers"] = [get_instance(r) for r in node.readers]

        return get_instance(node.spec, **joined_spec_kwargs)
    elif isinstance(node, nodes.WithImportableSource):
        if not isinstance(node.source, ImportedSource):
            raise PyBioTransformationOrderException(
                f"Encountered unexpected node.source type {type(node.source)}. `get_instance` requires URITransformer and SourceTransformer to be applied beforehand."
            )

        joined_kwargs = dict(node.optional_kwargs)
        joined_kwargs.update(kwargs)
        if isinstance(node, nodes.ReaderSpec):
            assert "outputs" not in joined_kwargs
            joined_kwargs["outputs"] = node.outputs

        missing_kwargs = [req for req in node.required_kwargs if req not in joined_kwargs]
        if missing_kwargs:
            raise PyBioMissingKwargException(
                f"{node.__class__.__name__} missing required kwargs: {missing_kwargs}\n{node.__class__.__name__}={node}"
            )

        return node.source.callable_(**joined_kwargs)
    else:
        raise TypeError(node)


def train(model: nodes.Model, **kwargs) -> Any:
    # resolve magic kwargs
    available_magic_kwargs = {"pybio_model": model}
    enchanted_kwargs = {req: available_magic_kwargs[req] for req in model.spec.training.required_kwargs}
    enchanted_kwargs.update(kwargs)

    return get_instance(model.spec.training, **enchanted_kwargs)


def resolve_uri(
    uri_node: nodes.URI, cache_path: pathlib.Path, root_path: Optional[pathlib.Path] = None
) -> pathlib.Path:
    if (
        uri_node.scheme == "" or len(uri_node.scheme) == 1
    ):  # Guess that scheme is not a scheme, but a windows path drive letter instead for uri.scheme == 1
        if uri_node.netloc:
            raise PyBioValidationException(f"Invalid Path/URI: {uri_node}")

        if root_path is None:
            local_path = pathlib.Path(uri_node.path)
        else:
            local_path = root_path / uri_node.path
    elif uri_node.scheme == "file":
        raise NotImplementedError
        # things to keep in mind when implementing this:
        # - problems with absolute paths on windows:
        #   >>> assert Path(urlparse(WindowsPath().absolute().as_uri()).path).exists() fails
        # - relative paths are invalid URIs
    elif uri_node.netloc == "github.com":
        orga, repo_name, blob, commit_id, *in_repo_path = uri_node.path.strip("/").split("/")
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
    elif uri_node.scheme == "https":
        local_path = cache_path / uri_node.scheme / uri_node.netloc / uri_node.path.strip("/")
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            url_str = urlunparse([uri_node.scheme, uri_node.netloc, uri_node.path, "", "", ""])
            urlretrieve(url_str, str(local_path))
    else:
        raise ValueError(f"Unknown uri scheme {uri_node.scheme}")

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

    def transform_URI(self, node: nodes.URI) -> io.BytesIO:
        local_path = resolve_uri(node, root_path=self.root_path, cache_path=self.cache_path)
        with local_path.open(mode="rb") as f:
            return io.BytesIO(f.read())

    def transform_ImportablePath(self, node: nodes.ImportablePath) -> ResolvedImportablePath:
        return ResolvedImportablePath(
            filepath=(self.root_path / node.filepath).resolve(), callable_name=node.callable_name
        )

    def transform_ImportableModule(self, node: nodes.ImportableModule) -> LocalImportableModule:
        return LocalImportableModule(**dataclasses.asdict(node), python_path=self.python_path)

    def _guess_python_path_from_local_spec_path(self, root_path: pathlib.Path):
        def potential_paths():
            yield root_path
            yield from root_path.parents

        for path in potential_paths():
            if (path / "manifest.yaml").exists() or (path / "manifest.yml").exists():
                return path.resolve()

        raise ValueError("Missing manifest.yaml")


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

        return ImportedSource(callable_=getattr(module, node.callable_name))

    def transform_ImportableModule(self, node):
        raise RuntimeError("Encountered nodes.ImportableModule in SourceTransformer. Apply URITransformer first!")

    def transform_ResolvedImportablePath(self, node: ResolvedImportablePath) -> ImportedSource:
        importlib_spec = importlib.util.spec_from_file_location(f"user_imports.{uuid.uuid4().hex}", node.filepath)
        dep = importlib.util.module_from_spec(importlib_spec)
        importlib_spec.loader.exec_module(dep)
        return ImportedSource(callable_=getattr(dep, node.callable_name))

    def transform_ImportablePath(self, node):
        raise RuntimeError("Encountered nodes.ImportablePath in SourceTransformer. Apply URITransformer first!")


def load_spec_and_kwargs(
    uri: str,
    kwargs: Dict[str, Any] = None,
    *,
    root_path: pathlib.Path = pathlib.Path("."),
    cache_path: pathlib.Path,
    **spec_kwargs,
) -> Union[nodes.Model, nodes.Transformation, nodes.Reader, nodes.Sampler]:
    cache_path = cache_path.resolve()
    root_path = root_path.resolve()
    assert root_path.exists(), root_path

    data = {"spec": str(root_path / uri), "kwargs": kwargs or {}, **spec_kwargs}
    last_dot = uri.rfind(".")
    second_last_dot = uri[:last_dot].rfind(".")
    spec_suffix = uri[second_last_dot + 1 : last_dot]
    if spec_suffix == "model":
        tree = schema.Model().load(data)
    elif spec_suffix == "transformation":
        tree = schema.Transformation().load(data)
    elif spec_suffix == "reader":
        tree = schema.Reader().load(data)
    elif spec_suffix == "sampler":
        tree = schema.Sampler().load(data)
    else:
        raise ValueError(f"Invalid spec suffix: {spec_suffix}")

    local_spec_path = resolve_uri(uri_node=tree.spec, root_path=root_path, cache_path=cache_path)

    uri_transformer = URITransformer(root_path=local_spec_path.parent, cache_path=cache_path)
    local_tree = uri_transformer.transform(tree)
    local_tree_with_sources = SourceTransformer().transform(local_tree)
    return local_tree_with_sources


def load_model(*args, **kwargs) -> nodes.Model:
    ret = load_spec_and_kwargs(*args, **kwargs)
    assert isinstance(ret, nodes.Model)
    return ret
