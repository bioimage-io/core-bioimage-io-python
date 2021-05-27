from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


from . import raw_nodes
from bioimageio.core.protocols import Tensor

Axes = raw_nodes.Axes
CiteEntry = raw_nodes.CiteEntry
Dependencies = raw_nodes.Dependencies
FormatVersion = raw_nodes.FormatVersion
Framework = raw_nodes.Framework
ImplicitInputShape = raw_nodes.ImplicitInputShape
InputTensor = raw_nodes.InputTensor
Language = raw_nodes.Language
Node = raw_nodes.Node
ImplicitOutputShape = raw_nodes.ImplicitOutputShape
OutputTensor = raw_nodes.OutputTensor
Preprocessing = raw_nodes.Preprocessing
PreprocessingName = raw_nodes.PreprocessingName
Postprocessing = raw_nodes.Postprocessing
PostprocessingName = raw_nodes.PostprocessingName
WeightsFormat = raw_nodes.WeightsFormat
RunMode = raw_nodes.RunMode


@dataclass
class ImportedSource:
    factory: Callable

    def __call__(self, *args, **kwargs):
        return self.factory(*args, **kwargs)


@dataclass
class Spec(raw_nodes.Spec):
    documentation: Path
    covers: List[Path]


@dataclass
class WeightsEntry(raw_nodes.WeightsEntry):
    source: Path


@dataclass
class Model(raw_nodes.Model):
    weights: Dict[WeightsFormat, WeightsEntry]

    source: ImportedSource
    test_inputs: List[Path]
    test_outputs: List[Path]
