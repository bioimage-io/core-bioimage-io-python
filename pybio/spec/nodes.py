from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


from .raw_nodes import (
    Axes,  # noqa
    CiteEntry,  # noqa
    Dependencies,  # noqa
    FormatVersion,  # noqa
    Framework,  # noqa
    InputShape,  # noqa
    InputTensor,  # noqa
    Language,  # noqa
    Model as _RawModel,
    Node,  # noqa
    OutputShape,  # noqa
    OutputTensor,  # noqa
    Preprocessing,  # noqa
    PreprocessingName,  # noqa
    Spec as _RawSpec,
    Tensor,  # noqa
    WeightsEntry as _RawWeightsEntry,
    WeightsFormat,
    WithFileSource as _RawWithFileSource,
    WithImportableSource as _RawWithImprtableSource,
)


@dataclass
class ImportedSource:
    factory: Callable

    def __call__(self, *args, **kwargs):
        return self.factory(*args, **kwargs)


@dataclass
class WithImportedSource(_RawWithImprtableSource):
    source: ImportedSource


@dataclass
class Spec(_RawSpec):
    documentation: Path
    covers: List[Path]


@dataclass
class WithFileSource(_RawWithFileSource):
    source: Path


@dataclass
class WeightsEntry(_RawWeightsEntry, WithFileSource):
    covers: List[Path]
    test_inputs: List[Path]
    test_outputs: List[Path]
    documentation: Optional[Path]


@dataclass
class Model(_RawModel, WithImportedSource):

    weights: Dict[WeightsFormat, WeightsEntry]
