from collections.abc import Mapping
from typing import Annotated, TypeVar

from ._common_annotations import PydanticMappingProxyAnnotation
from .axis import AxisId

_T = TypeVar("_T")

PerAxisAnno = Annotated[Mapping[AxisId, _T], PydanticMappingProxyAnnotation]
"""PerAxis annotated with `PydanticMappingProxyAnnotation` to be compatible with pydantic models."""
