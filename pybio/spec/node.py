import dataclasses
from dataclasses import dataclass
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
    filepath: str
    callable_name: str


@dataclass
class ImportableModule(Node):
    module_name: str
    callable_name: str


ImportableSource = Union[ImportableModule, ImportablePath]


@dataclass
class WithImportableSource:
    source: ImportableSource
    required_kwargs: List[str]
    optional_kwargs: Dict[str, Any]


@dataclass
class CiteEntry(Node):
    text: str
    doi: Optional[str]
    url: Optional[str]


@dataclass
class BaseSpec(Node, WithImportableSource):
    name: str
    format_version: str
    description: str
    cite: List[CiteEntry]
    authors: List[str]
    documentation: Path
    tags: List[str]

    language: str
    framework: Optional[str]

    test_input: Optional[Path]
    test_output: Optional[Path]
    thumbnail: Optional[Path]


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
    shape: Union[Tuple[int, ...], MagicShapeValue, InputShape]


@dataclass
class OutputArray(Array):
    shape: Union[Tuple[int, ...], MagicShapeValue, OutputShape]
    halo: Tuple[int, ...]


@dataclass
class WithInputs:
    inputs: Union[MagicTensorsValue, List[InputArray]]


@dataclass
class WithOutputs:
    outputs: Union[MagicTensorsValue, List[OutputArray]]


@dataclass
class URI:
    scheme: str
    netloc: str
    path: str


@dataclass
class SpecURI(URI):
    spec_schema: "pybio.spec.schema.BaseSpec"


@dataclass
class SpecWithKwargs(Node):
    spec: Union[SpecURI, BaseSpec]
    kwargs: Dict[str, Any]


Dependencies = NewType("Dependencies", Path)


@dataclass
class TransformationSpec(BaseSpec, WithInputs, WithOutputs):
    dependencies: Dependencies


@dataclass
class Transformation(SpecWithKwargs):
    spec: Union[SpecURI, TransformationSpec]


@dataclass
class WithFileSource:
    source: URI
    hash: Dict[str, str]


@dataclass
class Weights(Node, WithFileSource):
    pass


@dataclass
class Prediction(Node):
    weights: Weights
    dependencies: Optional[Dependencies]
    preprocess: Optional[List[Transformation]]
    postprocess: Optional[List[Transformation]]


@dataclass
class ReaderSpec(BaseSpec, WithOutputs):
    dependencies: Optional[Dependencies]


@dataclass
class Reader(SpecWithKwargs):
    spec: Union[SpecURI, ReaderSpec]
    transformations: List[Transformation]


@dataclass
class SamplerSpec(BaseSpec, WithOutputs):
    dependencies: Optional[Dependencies]


@dataclass
class Sampler(SpecWithKwargs):
    spec: Union[SpecURI, SamplerSpec]
    readers: List[Reader]


@dataclass
class Optimizer(Node, WithImportableSource):
    pass


@dataclass
class Setup(Node):
    samplers: List[Sampler]
    preprocess: List[Transformation]
    postprocess: List[Transformation]
    losses: List[Transformation]
    optimizer: Optimizer
    # todo: make non-optional (here, but add as optional to schmea) todo: add real meta sampler
    sampler: Optional[Sampler] = None

    def __post_init__(self):
        assert len(self.samplers) == 1
        self.sampler = self.samplers[0]


@dataclass
class Training(Node, WithImportableSource):
    setup: Setup
    dependencies: Dependencies
    description: Optional[str]


@dataclass
class ModelSpec(BaseSpec, WithInputs, WithOutputs):
    prediction: Prediction
    training: Optional[Training]


@dataclass
class Model(SpecWithKwargs):
    spec: Union[SpecURI, ModelSpec]
