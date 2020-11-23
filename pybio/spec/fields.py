import pathlib
import typing
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy
from marshmallow import validate
from marshmallow.fields import (
    DateTime,  # noqa
    Dict,
    Field,
    Float,  # noqa
    Integer,  # noqa
    List,  # noqa
    Nested,
    Number,
    String,
    Tuple as MarshmallowTuple,
    ValidationError,
)

from pybio.spec import nodes
from pybio.spec.exceptions import PyBioValidationException


class Tuple(MarshmallowTuple):
    def _jsonschema_type_mapping(self):
        import marshmallow_jsonschema

        return {
            "type": "array",
            "items": [marshmallow_jsonschema.JSONSchema()._get_schema_for_field(self, tf) for tf in self.tuple_fields],
        }


class SpecURI(Nested):
    def _deserialize(self, value, attr, data, **kwargs):
        uri = urlparse(value)

        if uri.query:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI query: {uri.query}")
        if uri.fragment:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI fragment: {uri.fragment}")
        if uri.params:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI params: {uri.params}")

        # account for leading '/' for windows paths, e.g. '/C:/folder'
        # see https://stackoverflow.com/questions/43911052/urlparse-on-a-windows-file-scheme-uri-leaves-extra-slash-at-start
        path = url2pathname(uri.path)

        return nodes.SpecURI(spec_schema=self.schema, scheme=uri.scheme, netloc=uri.netloc, path=path, query="")


class URI(String):
    def _deserialize(self, *args, **kwargs) -> nodes.URI:
        uri_str = super()._deserialize(*args, **kwargs)
        uri = urlparse(uri_str)

        if uri.fragment:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI fragment: {uri.fragment}")
        if uri.params:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI params: {uri.params}")

        return nodes.URI(scheme=uri.scheme, netloc=uri.netloc, path=uri.path, query=uri.query)


class Path(String):
    def _deserialize(self, *args, **kwargs):
        path_str = super()._deserialize(*args, **kwargs)
        return pathlib.Path(path_str)


class SHA256(String):
    def _deserialize(self, *args, **kwargs):
        value_str = super()._deserialize(*args, **kwargs)
        return value_str


class ImportableSource(String):
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
            return nodes.ImportableModule(callable_name=object_name, module_name=module_name)

        elif self._is_filepath(source_str):
            if source_str.startswith("/"):
                raise ValidationError("Only realative paths are allowed")

            parts = source_str.split("::")
            if len(parts) != 2:
                raise ValidationError("Incorrect filepath format, expected example.py::ClassName")

            module_path, object_name = parts

            return nodes.ImportablePath(callable_name=object_name, filepath=module_path)
        else:
            raise ValidationError(source_str)


class Kwargs(Dict):
    def _deserialize(self, value, attr, data, **kwargs):
        return nodes.Kwargs(**value)


class Axes(String):
    def _deserialize(self, *args, **kwargs) -> str:
        axes_str = super()._deserialize(*args, **kwargs)
        valid_axes = self.metadata.get("valid_axes", "bczyx")
        if any(a not in valid_axes for a in axes_str):
            raise PyBioValidationException(f"Invalid axes! Valid axes consist of: {valid_axes}")

        return axes_str


class Dependencies(String):  # todo: make Debency inherit from URI
    pass


class Tensors(Nested):
    def __init__(self, *args, valid_magic_values: typing.List[nodes.MagicTensorsValue], **kwargs):
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
                value = nodes.MagicTensorsValue(value)
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
    def __init__(self, *args, valid_magic_values: typing.List[nodes.MagicShapeValue], **kwargs):
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
                value = nodes.MagicShapeValue(value)
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


class Array(Field):
    def __init__(self, inner: Field, **kwargs):
        self.inner = inner
        super().__init__(**kwargs)

    @property
    def dtype(self) -> typing.Union[typing.Type[int], typing.Type[float], typing.Type[str]]:
        if isinstance(self.inner, Integer):
            return int
        elif isinstance(self.inner, Float):
            return float
        elif isinstance(self.inner, String):
            return str
        else:
            raise NotImplementedError(self.inner)

    def _deserialize_inner(self, value):
        if isinstance(value, list):
            return [self._deserialize_inner(v) for v in value]
        else:
            return self.inner.deserialize(value)

    def deserialize(self, value: typing.Any, attr: str = None, data: typing.Mapping[str, typing.Any] = None, **kwargs):
        value = self._deserialize_inner(value)

        if isinstance(value, list):
            try:
                return numpy.array(value, dtype=self.dtype)
            except ValueError as e:
                raise PyBioValidationException(str(e)) from e
        else:
            return value
