import importlib
import pathlib
import yaml
import typing
import contextlib
from dataclasses import fields
from typing import Any, Dict, Optional, Union
from . import spec_types
from . import schema

from .spec_types import ModelSpec, TransformationSpec, ReaderSpec, SamplerSpec, Importable, Node
from urllib.parse import ParseResult


def iter_fields(node: Node):
    for field in fields(node):
        yield field.name, getattr(node, field.name)


class NodeVisitor:
    def visit(self, node: Any) -> None:
        method = 'visit_' + node.__class__.__name__

        visitor = getattr(self, method, self.generic_visit)

        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        if isinstance(node, Node):
            for field, value in iter_fields(node):
                self.visit(value)


class NodeTransformer(NodeVisitor):
    class Transform:
        def __init__(self, value):
            self.value = value

    def generic_visit(self, node):
        if isinstance(node, Node):
            for field, value in iter_fields(node):
                op = self.visit(value)
                if isinstance(op, self.Transform):
                    setattr(node, field, op.value)


def _resolve_import(importable: Importable):
    if isinstance(importable, Importable.Module):
        module = importlib.import_module(importable.module_name)
        return getattr(module, importable.callable_name)
    elif isinstance(importable, Importable.Path):
        raise NotImplementedError()

    raise NotImplementedError(f"Can't resolve import for type {type(importable)}")


def get_instance(spec, **kwargs) -> Any:
    joined_kwargs = dict(spec.spec.optional_kwargs)
    joined_kwargs.update(spec.kwargs)
    joined_kwargs.update(kwargs)
    cls = _resolve_import(spec.spec.source)
    return cls(**joined_kwargs)


def train(model: ModelSpec, kwargs: Dict[str, Any] = None) -> Any:
    if kwargs is None:
        kwargs = {}

    print("HEY", model, model.spec)
    complete_kwargs = dict(model.spec.training.optional_kwargs)
    complete_kwargs.update(kwargs)

    mspec = "model_spec"
    if mspec not in complete_kwargs and mspec in model.spec.training.required_kwargs:
        complete_kwargs[mspec] = model

    train_cls = _resolve_import(model.spec.training.source)
    return train_cls(**complete_kwargs)


def resolve_local_path(path_str: str, parent_path: Optional[str] = None) -> pathlib.Path:
    local_path = pathlib.Path(path_str)
    if local_path.is_absolute():
        return local_path.resolve()
    elif parent_path:
        so_far = context["spec_path"]
        return (so_far[-1].parent / local_path).resolve()
    else:
        return local_path.absolute().resolve()


def resolve_uri(uri, parent_path=None):
    if uri.scheme == "" or len(uri.scheme) == 1:  # Guess that scheme is not a scheme, but a windows path drive letter instead for uri.scheme == 1
        if uri.netloc:
            raise PyBioValidationException(f"Invalid Path/URI: {uri}")
        spec_path = resolve_local_path(value, self.context)
    elif uri.scheme == "file":
        raise NotImplementedError
        # things to keep in mind when implementing this:
        # - problems with absolute paths on windows:
        #   >>> assert Path(urlparse(WindowsPath().absolute().as_uri()).path).exists() fails
        # - relative paths are invalid URIs
    elif uri.netloc == "github.com":
        orga, repo_name, blob, commit_id, *in_repo_path = uri.path.strip("/").split("/")
        in_repo_path = "/".join(in_repo_path)
        cached_repo_path = self.cache_path / orga / repo_name / commit_id
        spec_path = cached_repo_path / in_repo_path
        if not spec_path.exists():
            cached_repo_path = cached_repo_path.resolve().as_posix()
            subprocess.call(
                ["git", "clone", f"{uri.scheme}://{uri.netloc}/{orga}/{repo_name}.git", cached_repo_path]
            )
            # -C <working_dir> available in git 1.8.5+
            # https://github.com/git/git/blob/5fd09df3937f54c5cfda4f1087f5d99433cce527/Documentation/RelNotes/1.8.5.txt#L115-L116
            subprocess.call(["git", "-C", cached_repo_path, "checkout", "--force", commit_id])
    else:
        raise ValueError(f"Unknown uri scheme {uri.scheme}")

    if "spec_path" in self.context:
        self.context["spec_path"].append(spec_path)
    else:
        self.context["spec_path"] = [spec_path]

    with spec_path.open() as f:
        value_data = yaml.safe_load(f)

    loaded_spec =  super()._deserialize(value_data, attr, data, **kwargs)
    self.context["spec_path"].pop()
    return loaded_spec


@contextlib.contextmanager
def modified_sys_path(path: typing.Union[str, pathlib.Path]):
    # FIXME: Probably all specs containing implementation in same folder should use common prefix for this
    # to avoid namespace pollution e.g. spec_root.unet2d.UNet2D
    # or with something like ./unet2d.py::UNet2D
    sys.path.insert(0, str(path))
    yield
    del sys.path[0]


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


class URITransformer(NodeTransformer):
    def __init__(self, root_path=None):
        self.root_path = root_path

    def visit_URI(self, node):
        path = node.path

        if self.root_path:
            path = pathlib.Path(self.root_path) / pathlib.Path(node.path)

        with open(path) as f:
            res = yaml.safe_load(f)
        return self.Transform(node.loader.load(res))


def load_spec(
    uri: str, kwargs: Dict[str, Any] = None
) -> Union[spec_types.ModelSpec, spec_types.TransformationSpec, spec_types.ReaderSpec, spec_types.SamplerSpec]:

    data = {"spec": uri, "kwargs": kwargs or {}}
    last_dot = uri.rfind(".")
    second_last_dot = uri[:last_dot].rfind(".")
    spec_suffix = uri[second_last_dot + 1 : last_dot]
    if spec_suffix == "model":
        tree = schema.ModelSpec().load(data)
    elif spec_suffix == "transformation":
        tree = schema.TransformationSpec().load(data)
    elif spec_suffix == "reader":
        tree = schema.ReaderSpec().load(data)
    elif spec_suffix == "sampler":
        tree = schema.SamplerSpec().load(data)
    else:
        raise ValueError(f"Invalid spec suffix: {spec_suffix}")

    transformer = URITransformer()
    transformer.visit(tree)
    transformer = URITransformer(pathlib.Path(uri).parent)
    transformer.visit(tree)
    return tree
