import pybio

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Any, Dict, NewType, Tuple, Union, Type, NamedTuple


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
class Importable(Node):
    callable_name: str


@dataclass
class ImportableFromPath(Importable):
    filepath: str


@dataclass
class ImportableFromModule(Importable):
    module_name: str


Source = Union[ImportableFromModule, ImportableFromPath]


@dataclass
class WithSource:
    source: Source
    required_kwargs: List[str]
    optional_kwargs: Dict[str, Any]

# Types for non-nested fields
Axes = NewType("Axes", str)
Dependencies = NewType("Dependencies", Path)

# Types for schema
@dataclass
class CiteEntry(Node):
    text: str
    doi: Optional[str]
    url: Optional[str]


@dataclass
class BaseSpec(Node, WithSource):
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
    halo: List[int]


@dataclass
class WithInputs:
    inputs: Union[MagicTensorsValue, List[InputArray]]


@dataclass
class WithOutputs:
    outputs: Union[MagicTensorsValue, List[OutputArray]]


@dataclass
class SpecWithKwargs(Node):
    spec: BaseSpec
    kwargs: Dict[str, Any]


@dataclass
class TransformationSpec(BaseSpec, WithInputs, WithOutputs):
    dependencies: Dependencies


@dataclass
class Transformation(SpecWithKwargs):
    spec: TransformationSpec


@dataclass
class Weights(Node):
    source: str
    hash: Dict[str, str]


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
    spec: ReaderSpec


@dataclass
class SamplerSpec(BaseSpec, WithOutputs):
    dependencies: Optional[Dependencies]


@dataclass
class Sampler(SpecWithKwargs):
    spec = SamplerSpec


@dataclass
class Optimizer(Node, WithSource):
    pass

@dataclass
class Setup(Node):
    readers: List[Reader]
    sampler: Sampler
    preprocess: List[Transformation]
    postprocess: List[Transformation]
    losses: List[Transformation]
    optimizer: Optimizer


@dataclass
class Training(Node, WithSource):
    setup: Setup
    dependencies: Dependencies
    description: Optional[str]


@dataclass
class ModelSpec(BaseSpec, WithInputs, WithOutputs):
    prediction: Prediction
    training: Optional[Training]


@dataclass
class Model(SpecWithKwargs):
    spec: ModelSpec


@dataclass
class URI:
    scheme: str
    netloc: str
    path: str


@dataclass
class SpecURI(URI):
    spec_schema: "pybio.spec.schema.BaseSpec"


@dataclass
class DataURI(URI):
    spec_schema: "pybio.spec.schema.BaseSpec"
