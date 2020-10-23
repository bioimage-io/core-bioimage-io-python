import pathlib
import typing
from urllib.parse import urlparse

from marshmallow.fields import (
    DateTime,  # noqa
    Dict,
    Float,  # noqa
    Integer,  # noqa
    List,  # noqa
    Nested,
    Str,
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

        return nodes.SpecURI(spec_schema=self.schema, scheme=uri.scheme, netloc=uri.netloc, path=uri.path, query="")


class URI(Str):
    def _deserialize(self, *args, **kwargs) -> nodes.URI:
        uri_str = super()._deserialize(*args, **kwargs)
        uri = urlparse(uri_str)

        if uri.fragment:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI fragment: {uri.fragment}")
        if uri.params:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI params: {uri.params}")

        return nodes.URI(scheme=uri.scheme, netloc=uri.netloc, path=uri.path, query=uri.query)


class Path(Str):
    def _deserialize(self, *args, **kwargs):
        path_str = super()._deserialize(*args, **kwargs)
        return pathlib.Path(path_str)


class SHA256(Str):
    def _deserialize(self, *args, **kwargs):
        value_str = super()._deserialize(*args, **kwargs)
        return value_str


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


class Axes(Str):
    def _deserialize(self, *args, **kwargs) -> str:
        axes_str = super()._deserialize(*args, **kwargs)
        valid_axes = "bczyx"
        if any(a not in valid_axes for a in axes_str):
            raise PyBioValidationException(f"Invalid axes! Valid axes are: {valid_axes}")

        return axes_str


class Dependencies(Str):  # todo: make Debency inherit from URI
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
