from urllib.parse import urlparse, ParseResult
import pathlib
import typing

from marshmallow.fields import Str, Nested, List, Dict, Integer, Float, Tuple, ValidationError  # noqa

from pybio.spec.exceptions import PyBioValidationException
from pybio.spec import node


class SpecURI(Nested):
    def _deserialize(self, value, attr, data, **kwargs):
        uri = urlparse(value)

        if uri.query:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI query: {uri.query}")
        if uri.fragment:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI fragment: {uri.fragment}")
        if uri.params:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI params: {uri.params}")

        return node.SpecURI(spec_schema=self.schema, scheme=uri.scheme, netloc=uri.netloc, path=uri.path)


class URI(Str):
    def _deserialize(self, *args, **kwargs) -> node.URI:
        uri_str = super()._deserialize(*args, **kwargs)
        uri = urlparse(uri_str)

        if uri.query:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI query: {uri.query}")
        if uri.fragment:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI fragment: {uri.fragment}")
        if uri.params:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI params: {uri.params}")

        return node.URI(scheme=uri.scheme, netloc=uri.netloc, path=uri.path)


class Path(Str):
    def _deserialize(self, *args, **kwargs):
        path_str = super()._deserialize(*args, **kwargs)
        return pathlib.Path(path_str)


class ImportableSource(Str):
    @staticmethod
    def _is_import(path):
        return "::" not in path

    @staticmethod
    def _is_filepath(path):
        return "::" in path

    def _deserialize(self, *args, **kwargs) -> typing.Any:
        source_str: str = super()._deserialize(*args, **kwargs)
        if self._is_import(source_str):
            last_dot_idx = source_str.rfind(".")

            module_name = source_str[:last_dot_idx]
            object_name = source_str[last_dot_idx + 1 :]
            return node.ImportableModule(callable_name=object_name, module_name=module_name)

        elif self._is_filepath(source_str):
            if source_str.startswith("/"):
                raise ValidationError("Only realative paths are allowed")

            parts = source_str.split("::")
            if len(parts) != 2:
                raise ValidationError("Incorrect filepath format, expected example.py::ClassName")

            module_path, object_name = parts

            spec_dir = pathlib.Path(self.context.get("spec_path", ".")).parent
            abs_path = spec_dir / pathlib.Path(module_path)

            return node.ImportablePath(callable_name=object_name, filepath=module_path)


class Axes(Str):
    def _deserialize(self, *args, **kwargs) -> str:
        axes_str = super()._deserialize(*args, **kwargs)
        valid_axes = "bczyx"
        if any(a not in valid_axes for a in axes_str):
            raise PyBioValidationException(f"Invalid axes! Valid axes are: {valid_axes}")

        return axes_str


class Dependencies(Str): # todo: make Debency inherit from URI
    pass


class Tensors(Nested):
    def __init__(self, *args, valid_magic_values: typing.List[node.MagicTensorsValue], **kwargs):
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
                value = node.MagicTensorsValue(value)
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
    def __init__(self, *args, valid_magic_values: typing.List[node.MagicShapeValue], **kwargs):
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
                value = node.MagicShapeValue(value)
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
