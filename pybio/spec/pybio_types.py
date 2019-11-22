from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Callable, Any, Dict, NewType, Tuple, Union


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
class Transformation(MinimalYAML):
    dependencies: Dependencies
    inputs: Union[MagicTensorsValue, List[InputTensor]]
    outputs: Union[MagicTensorsValue, List[OutputTensor]]


@dataclass
class BaseSpec:
    spec: Path
    kwargs: Dict[str, Any]


@dataclass
class TransformationSpec(BaseSpec):
    pass


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
    pass


@dataclass
class Sampler(MinimalYAML):
    dependencies: Optional[Dependencies]
    outputs: Union[MagicTensorsValue, List[OutputTensor]]


@dataclass
class SamplerSpec(BaseSpec):
    pass


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
class Model(MinimalYAML):
    prediction: Prediction
    inputs: Union[MagicTensorsValue, List[InputTensor]]
    outputs: Union[MagicTensorsValue, List[OutputTensor]]
    training: Optional[Training]

@dataclass
class ModelSpec(BaseSpec):
    pass
