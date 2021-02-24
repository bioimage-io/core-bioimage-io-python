from __future__ import annotations

import datetime
import distutils.version
import pathlib
import typing
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy
from marshmallow import ValidationError, fields as marshmallow_fields, validate as marshmallow_validate

from pybio.spec import raw_nodes
from pybio.spec.exceptions import PyBioValidationException

Dict = marshmallow_fields.Dict
Float = marshmallow_fields.Float
Integer = marshmallow_fields.Integer
List = marshmallow_fields.List
Nested = marshmallow_fields.Nested
Number = marshmallow_fields.Number
String = marshmallow_fields.String

if typing.TYPE_CHECKING:
    import pybio.spec.schema


class StrictVersion(marshmallow_fields.Field):
    def _deserialize(
        self,
        value: typing.Any,
        attr: typing.Optional[str],
        data: typing.Optional[typing.Mapping[str, typing.Any]],
        **kwargs,
    ):
        return distutils.version.StrictVersion(str(value))

    def _serialize(self, value: typing.Any, attr: str, obj: typing.Any, **kwargs):
        return str(value)


class DateTime(marshmallow_fields.DateTime):
    """
    Parses datetime in ISO8601 or if value already has datetime.datetime type
    returns this value
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, datetime.datetime):
            return value

        return super()._deserialize(value, attr, data, **kwargs)


class Tuple(marshmallow_fields.Tuple):
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

        return raw_nodes.SpecURI(spec_schema=self.schema, scheme=uri.scheme, netloc=uri.netloc, path=path, query="")


class URI(String):
    def _deserialize(self, *args, **kwargs) -> raw_nodes.URI:
        uri_str = super()._deserialize(*args, **kwargs)
        uri = urlparse(uri_str)

        if uri.fragment:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI fragment: {uri.fragment}")
        if uri.params:
            raise PyBioValidationException(f"Invalid URI: {uri}. Got URI params: {uri.params}")

        return raw_nodes.URI(scheme=uri.scheme, netloc=uri.netloc, path=uri.path, query=uri.query)


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
            return raw_nodes.ImportableModule(callable_name=object_name, module_name=module_name)

        elif self._is_filepath(source_str):
            if source_str.startswith("/"):
                raise ValidationError("Only realative paths are allowed")

            parts = source_str.split("::")
            if len(parts) != 2:
                raise ValidationError("Incorrect filepath format, expected example.py::ClassName")

            module_path, object_name = parts

            return raw_nodes.ImportablePath(callable_name=object_name, filepath=module_path)
        else:
            raise ValidationError(source_str)


# class Kwargs(Dict):
#     def _deserialize(self, value, attr, data, **kwargs):
#         return nodes.Kwargs(**value)


class Axes(String):
    def _deserialize(self, *args, **kwargs) -> str:
        axes_str = super()._deserialize(*args, **kwargs)
        valid_axes = self.metadata.get("valid_axes", "bczyx")
        if any(a not in valid_axes for a in axes_str):
            raise PyBioValidationException(f"Invalid axes! Valid axes consist of: {valid_axes}")

        return axes_str


class ProcMode(String):
    all_modes = ("fixed", "per_dataset", "per_sample")

    def __init__(
        self,
        *,
        validate: typing.Optional[
            typing.Union[
                typing.Callable[[typing.Any], typing.Any], typing.Iterable[typing.Callable[[typing.Any], typing.Any]]
            ]
        ] = None,
        valid_modes: typing.Sequence[str] = all_modes,
        required=True,
        **kwargs,
    ) -> None:
        assert all(vm in self.all_modes for vm in valid_modes), valid_modes

        if validate is None:
            validate = []

        if isinstance(validate, (list, tuple)):
            validate = list(validate)
        else:
            validate = [validate]

        validate.append(marshmallow_validate.OneOf(self.all_modes))
        super().__init__(validate=validate, required=required, **kwargs)


class Dependencies(String):  # todo: make Dependency inherit from URI
    pass


class Shape(marshmallow_fields.Field):  # todo: use marshmallow_union instead
    explicit_shape = List(Integer)

    def __init__(
        self,
        nested_schema: typing.Union[
            typing.Type[pybio.spec.schema.InputShape], typing.Type[pybio.spec.schema.OutputShape]
        ],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nested_schema = nested_schema

    def _jsonschema_type_mapping(self):
        return {"type": "array", "items": {"type": "number"}}

    def _deserialize(self, value, attr, data, partial=None, many=False, **kwargs):
        assert not many
        if isinstance(value, list):
            return self.explicit_shape.deserialize(value)
        else:
            return self.nested_schema().load(value)


class Array(marshmallow_fields.Field):
    def __init__(self, inner: marshmallow_fields.Field, **kwargs):
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


class Halo(List):
    def __init__(self):
        super().__init__(Integer, missing=None)

    def _deserialize(self, value, attr, data, **kwargs) -> typing.List[typing.Any]:
        if value is None:
            return [0] * len(data["shape"])
        else:
            return super()._deserialize(value, attr, data, **kwargs)
