from __future__ import annotations

import distutils.version
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, TYPE_CHECKING, Tuple, Union

from marshmallow import missing

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
    "pickle",
    "pytorch_state_dict",
    "pytorch_script",
    "keras_hdf5",
    "tensorflow_js",
    "tensorflow_saved_model_bundle",
    "onnx",
]

Dependencies = NewType("Dependencies", str)
Axes = NewType("Axes", str)


@dataclass
class Node:
    pass


@dataclass
class ImportablePath(Node):
    filepath: Path = missing
    callable_name: str = missing


@dataclass
class ImportableModule(Node):
    module_name: str = missing
    callable_name: str = missing


ImportableSource = Union[ImportableModule, ImportablePath]


@dataclass
class WithImportableSource:
    source: ImportableSource = missing
    sha256: str = missing
    kwargs: Dict[str, Any] = missing


@dataclass
class CiteEntry(Node):
    text: str = missing
    doi: Optional[str] = missing
    url: Optional[str] = missing


@dataclass
class URI(Node):
    """URI as  scheme:[//authority]path[?query][#fragment]"""

    scheme: str = missing
    authority: str = missing
    path: str = missing
    query: str = missing
    fragment: str = missing

    def __str__(self):
        """scheme:[//authority]path[?query][#fragment]"""
        return (
            (self.scheme + ":" if self.scheme else "")
            + ("//" + self.authority if self.authority else "")
            + self.path
            + ("?" + self.query if self.query else "")
            + ("#" + self.fragment if self.fragment else "")
        )


@dataclass
class RunMode(Node):
    name: str = missing
    kwargs: Dict[str, Any] = missing


@dataclass
class Spec(Node):
    format_version: FormatVersion = missing
    name: str = missing
    description: str = missing

    authors: List[str] = missing
    cite: List[CiteEntry] = missing

    git_repo: Optional[str] = missing
    tags: List[str] = missing
    license: str = missing

    documentation: URI = missing
    covers: List[URI] = missing
    attachments: Dict[str, Any] = missing

    language: Language = missing
    framework: Framework = missing
    dependencies: Optional[Dependencies] = missing
    timestamp: datetime = missing

    run_mode: Optional[RunMode] = missing
    config: dict = missing


@dataclass
class ImplicitInputShape(Node):
    min: List[float] = missing
    step: List[float] = missing

    def __len__(self):
        return len(self.min)


@dataclass
class ImplicitOutputShape(Node):
    reference_input: str = missing
    scale: List[float] = missing
    offset: List[int] = missing

    def __len__(self):
        return len(self.scale)


@dataclass
class Preprocessing:
    name: PreprocessingName = missing
    kwargs: Dict[str, Any] = missing


@dataclass
class Postprocessing:
    name: PostprocessingName = missing
    kwargs: Dict[str, Any] = missing


@dataclass
class InputTensor:
    name: str = missing
    data_type: str = missing
    axes: Axes = missing
    shape: Union[List[int], ImplicitInputShape] = missing
    preprocessing: List[Preprocessing] = missing
    description: Optional[str] = missing
    data_range: Tuple[float, float] = missing


@dataclass
class OutputTensor:
    name: str = missing
    data_type: str = missing
    axes: Axes = missing
    shape: Union[List[int], ImplicitOutputShape] = missing
    halo: List[int] = missing
    postprocessing: List[Postprocessing] = missing
    description: Optional[str] = missing
    data_range: Tuple[float, float] = missing


@dataclass
class SpecURI(URI):
    spec_schema: pybio.spec.schema.Spec = missing


@dataclass
class WithFileSource:
    source: URI = missing
    sha256: str = missing


@dataclass
class WeightsEntry(Node, WithFileSource):
    authors: List[str] = missing
    attachments: Dict = missing
    parent: Optional[str] = missing
    # ONNX specific
    opset_version: Optional[int] = missing
    # tag: Optional[str]  # todo: check schema. only valid for tensorflow_saved_model_bundle format
    # todo: check schema. only valid for tensorflow_saved_model_bundle format
    tensorflow_version: Optional[distutils.version.StrictVersion] = missing


@dataclass
class ModelParent(Node):
    uri: URI = missing
    sha256: str = missing


@dataclass
class Model(Spec, WithImportableSource):
    inputs: List[InputTensor] = missing
    outputs: List[OutputTensor] = missing
    packaged_by: List[str] = missing
    parent: ModelParent = missing
    sample_inputs: List[URI] = missing
    sample_outputs: List[URI] = missing
    test_inputs: List[URI] = missing
    test_outputs: List[URI] = missing
    weights: Dict[WeightsFormat, WeightsEntry] = missing
