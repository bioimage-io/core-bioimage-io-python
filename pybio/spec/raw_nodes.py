from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, TYPE_CHECKING, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

if TYPE_CHECKING:
    import pybio.spec.schema

# Ideally only the current format version is valid.
# Older formats may be converter through `pybio.spec.utils.maybe_convert`,
# such that we only need to support the most up-to-date version.
FormatVersion = Literal["0.3.0"]

PreprocessingName = Literal["zero_mean_unit_variance"]
Language = Literal["python", "java"]
Framework = Literal["scikit-learn", "pytorch", "tensorflow"]
WeightsFormat = Literal["pickle", "pytorch", "keras"]

Dependencies = NewType("Dependencies", str)
Axes = NewType("Axes", str)


@dataclass
class Node:
    pass


@dataclass
class ImportablePath(Node):
    filepath: Union[Path]
    callable_name: str


@dataclass
class ImportableModule(Node):
    module_name: str
    callable_name: str


ImportableSource = Union[ImportableModule, ImportablePath]


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
    format_version: FormatVersion
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

    language: Language
    framework: Framework
    dependencies: Optional[Dependencies]
    timestamp: datetime

    config: dict


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
    name: PreprocessingName
    kwargs: Dict[str, Any]


@dataclass
class InputTensor(Tensor):
    shape: Union[List[int], InputShape]
    preprocessing: List[Preprocessing]


@dataclass
class OutputTensor(Tensor):
    shape: Union[List[int], OutputShape]
    halo: List[int]


@dataclass
class SpecURI(URI):
    spec_schema: pybio.spec.schema.Spec


@dataclass
class WithFileSource:
    source: URI
    sha256: str


@dataclass
class WeightsEntry(Node, WithFileSource):
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
    weights: Dict[WeightsFormat, WeightsEntry]
    inputs: List[InputTensor]
    outputs: List[OutputTensor]
