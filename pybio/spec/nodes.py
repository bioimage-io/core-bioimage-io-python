from collections import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

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


# @dataclass(init=False)
# class Kwargs(Node, Mapping):
#     __data: Dict[str, Any] = field(default_factory=dict)
#
#     def __init__(self, __data=None, **kwargs):
#         assert __data is None or not kwargs
#         self.__data = kwargs if __data is None else __data
#
#     def __iter__(self):
#         return iter(self.__data)
#
#     def __len__(self):
#         return len(self.__data)
#
#     def __getitem__(self, item):
#         return self.__data[item]
#
#     def __setitem__(self, key, value):
#         self.__data[key] = value


@dataclass
class WithImportableSource:
    source: ImportableSource
    sha256: str
    kwargs: Dict[str, Any]


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
class Spec(Node):
    format_version: Literal["0.3.0"]
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


Axes = NewType("Axes", str)


# @dataclass
# class ZeroMeanUnitVariance(Node):
#     mode: str
#     axes: Axes
#     mean: Optional[Union[float, List]]
#     std: Optional[Union[float, List]]


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


@dataclass
class Tensor(Node):
    name: str
    description: str
    axes: Optional[Axes]
    data_type: str
    data_range: Tuple[float, float]


@dataclass
class Preprocessing:
    name: Literal["zero_mean_unit_variance"]
    kwargs: Dict[str, Any]


@dataclass
class InputTensor(Tensor):
    shape: Union[List[int], MagicShapeValue, InputShape]
    preprocessing: List[Preprocessing]
    # preprocessing: List[Union[ZeroMeanUnitVariance]]



@dataclass
class OutputTensor(Tensor):
    shape: Union[List[int], MagicShapeValue, OutputShape]
    halo: List[int]


@dataclass
class SpecURI(URI):
    spec_schema: "pybio.spec.schema.Spec"


Dependencies = NewType("Dependencies", Path)


@dataclass
class WithFileSource:
    source: URI
    sha256: str


@dataclass
class Weight(Node, WithFileSource):
    id: str
    name: str
    description: str
    authors: List[str]
    covers: List[URI]
    test_inputs: List[URI]
    test_outputs: List[URI]
    documentation: Optional[URI]
    tags: List[str]
    attachments: Dict


@dataclass
class Model(Spec, WithImportableSource):
    language: Literal["python", "java"]
    framework: Literal["scikit-learn", "pytorch", "tensorflow"]
    dependencies: Optional[Dependencies]
    timestamp: datetime

    weights: Dict[Literal["pickle", "pytorch", "keras"], Weight]
    inputs: Union[MagicTensorsValue, List[InputTensor]]
    outputs: Union[MagicTensorsValue, List[OutputTensor]]

    config: Dict


@dataclass
class ModelParent(Node):
    """helper class to load model from spec uri"""

    spec: SpecURI


# helper nodes
@dataclass
class File(Node, WithFileSource):
    pass


@dataclass
class Resource(Node):
    uri: URI
