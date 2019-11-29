from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Callable, Any, Dict, NewType, Tuple, Union, Type, NamedTuple


class MagicTensorsValue(Enum):
    any = "any"
    same = "same"


class MagicShapeValue(Enum):
    any = "any"


# Types for non-nested fields
Axes = NewType("Axes", str)
Dependencies = NewType("Dependencies", Path)

# Types for schema
@dataclass
class CiteEntry:
    text: str
    doi: Optional[str]
    url: Optional[str]


@dataclass
class MinimalYAML:
    name: str
    format_version: str
    description: str
    cite: List[CiteEntry]
    authors: List[str]
    documentation: Path
    tags: List[str]

    language: str
    framework: Optional[str]
    source: Callable
    required_kwargs: List[str]
    optional_kwargs: Dict[str, Any]

    test_input: Optional[Path]
    test_output: Optional[Path]
    thumbnail: Optional[Path]


@dataclass
class InputShape:
    min: List[float]
    step: List[float]


@dataclass
class OutputShape:
    reference_input: Optional[str]
    scale: List[float]
    offset: List[int]
    halo: List[int]


@dataclass
class Tensor:
    name: str
    axes: Optional[Axes]
    data_type: str
    data_range: Tuple[float, float]


@dataclass
class InputTensor(Tensor):
    shape: Union[Tuple[int, ...], MagicShapeValue, InputShape]


@dataclass
class OutputTensor(Tensor):
    shape: Union[Tuple[int, ...], MagicShapeValue, OutputShape]


@dataclass
class WithInputs:
    inputs: Union[MagicTensorsValue, Type[NamedTuple], List[InputTensor]]


@dataclass
class WithOutputs:
    outputs: Union[MagicTensorsValue, Type[NamedTuple], List[OutputTensor]]


@dataclass
class Transformation(MinimalYAML, WithInputs, WithOutputs):
    dependencies: Dependencies


@dataclass
class BaseSpec:
    spec: MinimalYAML
    kwargs: Dict[str, Any]


@dataclass
class TransformationSpec(BaseSpec):
    spec: Transformation


@dataclass
class Weights:
    source: str
    hash: Dict[str, str]


@dataclass
class Prediction:
    weights: Weights
    dependencies: Optional[Dependencies]
    preprocess: Optional[List[TransformationSpec]]
    postprocess: Optional[List[TransformationSpec]]


@dataclass
class Reader(MinimalYAML):
    dependencies: Optional[Dependencies]


@dataclass
class ReaderSpec(BaseSpec):
    spec: Reader


@dataclass
class Sampler(MinimalYAML, WithOutputs):
    dependencies: Optional[Dependencies]


@dataclass
class SamplerSpec(BaseSpec):
    spec = Sampler


@dataclass
class Optimizer:
    source: Callable
    required_kwargs: List[str]
    optional_kwargs: Dict[str, Any]


@dataclass
class Setup:
    reader: ReaderSpec
    sampler: SamplerSpec
    preprocess: Optional[List[TransformationSpec]]
    loss: List[TransformationSpec]
    optimizer: Optimizer


@dataclass
class Training:
    setup: Setup
    source: Callable
    required_kwargs: List[str]
    optional_kwargs: Dict[str, Any]
    dependencies: Dependencies
    description: Optional[str]


@dataclass
class Model(MinimalYAML, WithInputs, WithOutputs):
    prediction: Prediction
    training: Optional[Training]


@dataclass
class ModelSpec(BaseSpec):
    spec: Model
