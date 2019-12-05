from importlib import import_module
from urllib.parse import urlparse, ParseResult
import pathlib
import sys
import requests
import subprocess
import typing
import yaml
import contextlib

from marshmallow.fields import Str, Nested, List, Dict, Integer, Float, Tuple  # noqa

from pybio.spec.exceptions import InvalidDoiException, PyBioValidationException
from pybio.spec.spec_types import MagicTensorsValue, MagicShapeValue


@contextlib.contextmanager
def modified_sys_path(path: typing.Union[str, pathlib.Path]):
    # FIXME: Probably all specs containing implementation in same folder should use common prefix for this
    # to avoid namespace pollution e.g. spec_root.unet2d.UNet2D
    # or with something like ./unet2d.py::UNet2D
    sys.path.insert(0, str(path))
    yield
    del sys.path[0]



def resolve_local_path(path_str: str, context: dict) -> pathlib.Path:
    local_path = pathlib.Path(path_str)
    if local_path.is_absolute():
        return local_path.resolve()
    elif "spec_path" in context:
        so_far = context["spec_path"]
        return (so_far[-1].parent / local_path).resolve()
    else:
        return local_path.absolute().resolve()


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


class SpecURI(Nested):
    # todo: improve cache location
    cache_path = pathlib.Path(__file__).parent.parent.parent / "cache"

    def _deserialize(self, value, attr, data, **kwargs):

        uri = urlparse(value)

        if uri.fragment:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI fragment: {uri.fragment}")
        if uri.params:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI params: {uri.params}")
        if uri.query:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI query: {uri.query}")

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


class URI(Str):
    def _deserialize(self, *args, **kwargs) -> ParseResult:
        uri_str = super()._deserialize(*args, **kwargs)
        return urlparse(uri_str)


class Path(Str):
    def _deserialize(self, *args, **kwargs):
        path_str = super()._deserialize(*args, **kwargs)
        path = resolve_local_path(path_str, self.context)
        if not path.exists():
            raise PyBioValidationException(f"{path.as_posix()} does not exist!")
        return path


class ImportableSource(Str):
    def _deserialize(self, *args, **kwargs) -> typing.Any:
        source_str: str = super()._deserialize(*args, **kwargs)
        last_dot_idx = source_str.rfind(".")

        spec_dir = self.context["spec_path"].parent
        module_name = source_str[:last_dot_idx]
        object_name = source_str[last_dot_idx + 1 :]

        with modified_sys_path(spec_dir):
            dep = import_module(module_name)

        return getattr(dep, object_name)


class Axes(Str):
    def _deserialize(self, *args, **kwargs) -> str:
        axes_str = super()._deserialize(*args, **kwargs)
        valid_axes = "bczyx"
        if any(a not in valid_axes for a in axes_str):
            raise PyBioValidationException(f"Invalid axes! Valid axes are: {valid_axes}")

        return axes_str


class Dependencies(URI):
    pass


class Tensors(Nested):
    def __init__(self, *args, valid_magic_values: typing.List[MagicTensorsValue], **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_magic_values = valid_magic_values

    def _deserialize(
        self,
        value: typing.Any,
        attr: typing.Optional[str],
        data: typing.Optional[typing.Mapping[str, typing.Any]],
        **kwargs,
    ):
        if isinstance(value, str):
            try:
                value = MagicTensorsValue(value)
            except ValueError as e:
                raise PyBioValidationException(str(e)) from e

            if value in self.valid_magic_values:
                return value
            else:
                raise PyBioValidationException(f"Invalid magic value: {value.value}")

        elif isinstance(value, list):
            return self._load(value, data, many=True)
            # if all(isinstance(v, str) for v in value):
            #     return namedtuple("CustomTensors", value)
            # else:
            #     return self._load(value, data, many=True)
        else:
            raise PyBioValidationException(f"Invalid input type: {type(value)}")


class Shape(Nested):
    def __init__(self, *args, valid_magic_values: typing.List[MagicShapeValue], **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_magic_values = valid_magic_values

    def _deserialize(
        self,
        value: typing.Any,
        attr: typing.Optional[str],
        data: typing.Optional[typing.Mapping[str, typing.Any]],
        **kwargs,
    ):
        if isinstance(value, str):
            try:
                value = MagicShapeValue(value)
            except ValueError as e:
                raise PyBioValidationException(str(e)) from e

            if value in self.valid_magic_values:
                return value
            else:
                raise PyBioValidationException(f"Invalid magic value: {value.value}")

        elif isinstance(value, list):
            if any(not isinstance(v, int) for v in value):
                raise PyBioValidationException("Encountered non-integers in shape")

            return tuple(value)
        elif isinstance(value, dict):
            return self._load(value, data)
        else:
            raise PyBioValidationException(f"Invalid input type: {type(value)}")
