from importlib import import_module
from typing import Any

import yaml

import pathlib
from urllib.parse import urlparse, ParseResult

from marshmallow import ValidationError
from marshmallow.fields import *

def resolve_local_path(path_str: str, context: dict) -> pathlib.Path:
    local_path = pathlib.Path(path_str)
    if local_path.is_absolute():
        return local_path.resolve()
    elif "spec_path" in context:
        return (context["spec_path"].parent / local_path).resolve()
    else:
        return local_path.resolve()

class SpecURI(Nested):
    def _deserialize(self, value, attr, data, **kwargs):
        spec_path = pathlib.Path(value)
        self.context["spec_path"] = spec_path
        with spec_path.open() as f:
            value_data = yaml.safe_load(f)

        return super()._deserialize(value_data, attr, data, **kwargs)

class URI(Str):
    def _deserialize(self, *args, **kwargs) -> ParseResult:
        uri_str = super()._deserialize(*args, **kwargs)
        return urlparse(uri_str)

class Path(Str):
    def _deserialize(self, *args, **kwargs):
        path_str = super()._deserialize(*args, **kwargs)
        path = resolve_local_path(path_str, self.context)
        if not path.exists():
            raise ValidationError(f"{path.as_posix()} does not exist!")
        return path

class ImportableSource(Str):
    def _deserialize(self, *args, **kwargs) -> Any:
        source_str: str = super()._deserialize(*args, **kwargs)
        last_dot_idx = source_str.rfind(".")
        module_name = source_str[:last_dot_idx]
        object_name = source_str[last_dot_idx + 1:]
        dep = import_module(module_name)
        return getattr(dep, object_name)

class Axes(Str):
    def _deserialize(self, *args, **kwargs) -> str:
        axes_str = super()._deserialize(*args, **kwargs)
        valid_axes = "bczyx"
        if any(a not in valid_axes for a in axes_str):
            raise ValidationError(f"Invalid axes! Valid axes are: {valid_axes}")

        return axes_str

class Dependencies(URI):
    pass
