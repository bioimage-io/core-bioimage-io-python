from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


from . import raw_nodes
from pybio.core.protocols import Tensor

Axes = raw_nodes.Axes
CiteEntry = raw_nodes.CiteEntry
Dependencies = raw_nodes.Dependencies
FormatVersion = raw_nodes.FormatVersion
Framework = raw_nodes.Framework
InputShape = raw_nodes.InputShape
InputTensor = raw_nodes.InputTensor
Language = raw_nodes.Language
Node = raw_nodes.Node
OutputShape = raw_nodes.OutputShape
OutputTensor = raw_nodes.OutputTensor
Preprocessing = raw_nodes.Preprocessing
PreprocessingName = raw_nodes.PreprocessingName
Postprocessing = raw_nodes.Postprocessing
PostprocessingName = raw_nodes.PostprocessingName
WeightsFormat = raw_nodes.WeightsFormat


@dataclass
class InputShape(raw_nodes.InputShape):
    def matches(self, tensor: Tensor):
        raise NotImplementedError  # todo: make InputShape handle min,step and explicit shape and add a MinStepInputShape for parsing?


@dataclass
class ImportedSource:
    factory: Callable

    def __call__(self, *args, **kwargs):
        return self.factory(*args, **kwargs)


@dataclass
class WithImportedSource(raw_nodes.WithImportableSource):
    source: ImportedSource


@dataclass
class Spec(raw_nodes.Spec):
    documentation: Path
    covers: List[Path]


@dataclass
class WithFileSource(raw_nodes.WithFileSource):
    source: Path


@dataclass
class WeightsEntry(raw_nodes.WeightsEntry, WithFileSource):
    pass


@dataclass
class Model(raw_nodes.Model, WithImportedSource):
    weights: Dict[WeightsFormat, WeightsEntry]

    test_inputs: List[Path]
    test_outputs: List[Path]
