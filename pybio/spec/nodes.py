from collections import Mapping, UserDict
from dataclasses import asdict, dataclass, field, make_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

import pybio


@dataclass
class Node:
    pass


class MagicTensorsValue(Enum):
    any = "any"
    same = "same"
    dynamic = "dynamic"


class MagicShapeValue(Enum):
    any = "any"
    dynamic = "dynamic"


@dataclass
class ImportablePath(Node):
    filepath: Union[Path]
    callable_name: str


@dataclass
class ImportableModule(Node):
    module_name: str
    callable_name: str


ImportableSource = Union[ImportableModule, ImportablePath]


@dataclass(init=False)
class Kwargs(Node, Mapping):
    __data: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, __data = None, **kwargs):
        assert __data is None or not kwargs
        self.__data = kwargs if __data is None else __data

    def __iter__(self):
        return iter(self.__data)

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, item):
        return self.__data[item]


@dataclass
class WithImportableSource:
    source: ImportableSource
    kwargs: Kwargs


@dataclass
class CiteEntry(Node):
    text: str
    doi: Optional[str]
    url: Optional[str]


@dataclass
class URI(Node):
    scheme: str
    netloc: str
    path: str
    query: str


@dataclass
class BaseSpec(Node):
    format_version: str
    name: str
    description: str

    authors: List[str]
    cite: List[CiteEntry]

    git_repo: str
    tags: List[str]
    license: str

    documentation: URI
    covers: List[URI]
    attachments: Dict[str, Any]

    config: Dict[str, Any]


@dataclass
class InputShape(Node):
    min: List[float]
    step: List[float]

    def __len__(self):
        return len(self.min)


@dataclass
class OutputShape(Node):
    reference_input: Optional[str]
    scale: List[float]
    offset: List[int]

    def __len__(self):
        return len(self.scale)


Axes = NewType("Axes", str)


@dataclass
class Array(Node):
    name: str
    axes: Optional[Axes]
    data_type: str
    data_range: Tuple[float, float]


@dataclass
class InputArray(Array):
    shape: Union[List[int], MagicShapeValue, InputShape]


@dataclass
class OutputArray(Array):
    shape: Union[List[int], MagicShapeValue, OutputShape]
    halo: List[int]


@dataclass
class SpecURI(URI):
    spec_schema: "pybio.spec.schema.BaseSpec"


@dataclass
class SpecWithKwargs(Node):
    spec: Union[SpecURI, BaseSpec]
    kwargs: Kwargs


Dependencies = NewType("Dependencies", Path)


@dataclass
class WithFileSource:
    source: URI
    sha256: str


@dataclass
class Weight(Node, WithFileSource):
    id: str
    test_input: Optional[URI]
    test_output: Optional[URI]


@dataclass
class ModelDetails(Node, WithImportableSource):
    language: str
    framework: str
    dependencies: Optional[Dependencies]


@dataclass
class ModelSpec(BaseSpec):
    inputs: Union[MagicTensorsValue, List[InputArray]]
    outputs: Union[MagicTensorsValue, List[OutputArray]]
    model: ModelDetails
    weights: List[Weight]


@dataclass
class Model(SpecWithKwargs):
    spec: Union[SpecURI, ModelSpec]


# helper nodes
@dataclass
class File(Node, WithFileSource):
    pass
