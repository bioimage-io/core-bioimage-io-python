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

PreprocessingName = Literal["binarize", "clip", "scale_linear", "sigmoid", "zero_mean_unit_variance", "scale_range"]
PostprocessingName = Literal[
    "binarize", "clip", "scale_linear", "sigmoid", "zero_mean_unit_variance", "scale_range", "scale_mean_variance"
]
Language = Literal["python", "java"]
Framework = Literal["scikit-learn", "pytorch", "tensorflow"]
WeightsFormat = Literal[
    "pickle", "pytorch_state_dict", "pytorch_script", "keras_hdf5", "tensorflow_js", "tensorflow_saved_model_bundle"
]

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
    reference_input: str
    scale: List[float]
    offset: List[int]

    def __len__(self):
        return len(self.scale)


@dataclass
class Preprocessing:
    name: PreprocessingName
    kwargs: Dict[str, Any]


@dataclass
class Postprocessing:
    name: PostprocessingName
    kwargs: Dict[str, Any]
    reference_tensor: Optional[str] = None


@dataclass
class InputTensor:
    name: str
    data_type: str
    shape: Union[List[int], InputShape]
    preprocessing: List[Preprocessing]
    description: Optional[str] = None
    axes: Optional[Axes] = None
    data_range: Tuple[float, float] = None


@dataclass
class OutputTensor:
    name: str
    data_type: str
    shape: Union[List[int], OutputShape]
    halo: List[int]
    postprocessing: List[Postprocessing]
    description: Optional[str] = None
    axes: Optional[Axes] = None
    data_range: Tuple[float, float] = None


@dataclass
class SpecURI(URI):
    spec_schema: pybio.spec.schema.Spec


@dataclass
class WithFileSource:
    source: URI
    sha256: str


@dataclass
class WeightsEntry(Node, WithFileSource):
    authors: List[str]
    attachments: Dict
    # tag: Optional[str]  # todo: add to schema and check schema. only valid for tensorflow_saved_model_bundle format
    # tensorflow_version: Optional[str]  # todo: add to schema and check schema. only valid for tensorflow_saved_model_bundle format


@dataclass
class Model(Spec, WithImportableSource):
    weights: Dict[WeightsFormat, WeightsEntry]

    inputs: List[InputTensor]
    outputs: List[OutputTensor]

    test_inputs: List[URI]
    test_outputs: List[URI]

    sample_inputs: List[URI]
    sample_outputs: List[URI]
