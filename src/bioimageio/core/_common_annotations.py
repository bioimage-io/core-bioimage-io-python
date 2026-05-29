from __future__ import annotations

from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    Hashable,
    Mapping,
    TypeVar,
)

import pydantic
from pydantic_core.core_schema import (
    CoreSchema,
    chain_schema,
    is_instance_schema,
    json_or_python_schema,
    no_info_plain_validator_function,
    plain_serializer_function_ser_schema,
)
from typing_extensions import get_args

from .common import PerMember

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")


def _validate_from_mapping(d: Mapping[_K, _V]) -> MappingProxyType[_K, _V]:
    return MappingProxyType(dict(d))


class PydanticMappingProxyAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> CoreSchema:

        k_type, v_type = get_args(source_type)
        mapping_proxy_schema = chain_schema(
            [
                handler.generate_schema(dict[k_type, v_type]),
                no_info_plain_validator_function(_validate_from_mapping),
                is_instance_schema(MappingProxyType),
            ]
        )
        return json_or_python_schema(
            json_schema=mapping_proxy_schema,
            python_schema=mapping_proxy_schema,
            serialization=plain_serializer_function_ser_schema(dict),
        )


_T = TypeVar("_T")

PerMemberAnno = Annotated[PerMember[_T], PydanticMappingProxyAnnotation]
