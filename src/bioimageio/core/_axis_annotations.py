from typing import Annotated, TypeVar

from ._common_annotations import PydanticMappingProxyAnnotation
from .axis import PerAxis

_T = TypeVar("_T")

PerAxisAnno = Annotated[PerAxis[_T], PydanticMappingProxyAnnotation]
"""PerAxis annotated with `PydanticMappingProxyAnnotation` to be compatible with pydantic models."""
