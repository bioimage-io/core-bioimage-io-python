import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

from marshmallow import missing
from marshmallow.utils import _Missing

from bioimageio.spec.model import raw_nodes as model_raw_nodes
from bioimageio.spec.rdf import raw_nodes as rdf_raw_nodes
from bioimageio.spec.shared import raw_nodes


@dataclass
class Node(raw_nodes.RawNode):
    pass


@dataclass
class ResourceDescription(Node, raw_nodes.ResourceDescription):
    pass


@dataclass
class URI(Node, raw_nodes.URI):
    pass


@dataclass
class ParametrizedInputShape(Node, raw_nodes.ParametrizedInputShape):
    pass


@dataclass
class ImplicitOutputShape(Node, raw_nodes.ImplicitOutputShape):
    pass


@dataclass
class Dependencies(Node, raw_nodes.Dependencies):
    file: pathlib.Path = missing


@dataclass
class LocalImportableModule(Node, raw_nodes.ImportableModule):
    """intermediate between raw_nodes.ImportableModule and nodes.ImportedSource. Used by SourceNodeTransformer"""

    root_path: pathlib.Path = missing


@dataclass
class ResolvedImportableSourceFile(Node, raw_nodes.ImportableSourceFile):
    """intermediate between raw_nodes.ImportableSourceFile and nodes.ImportedSource. Used by SourceNodeTransformer"""

    source_file: pathlib.Path = missing


@dataclass
class CiteEntry(Node, rdf_raw_nodes.CiteEntry):
    pass


@dataclass
class Author(Node, rdf_raw_nodes.Author):
    pass


@dataclass
class Badge(Node, rdf_raw_nodes.Badge):
    pass


@dataclass
class RDF(rdf_raw_nodes.RDF, Node):
    covers: Union[_Missing, List[Path]] = missing


@dataclass
class CollectionEntry(Node, rdf_raw_nodes.CollectionEntry):
    source: URI = missing


@dataclass
class ModelCollectionEntry(CollectionEntry, rdf_raw_nodes.ModelCollectionEntry):
    download_url: URI = missing


@dataclass
class ModelParent(Node, model_raw_nodes.ModelParent):
    pass


@dataclass
class Collection(RDF, rdf_raw_nodes.Collection):
    pass


@dataclass
class RunMode(Node, model_raw_nodes.RunMode):
    pass


@dataclass
class Preprocessing(Node, model_raw_nodes.Preprocessing):
    pass


@dataclass
class Postprocessing(Node, model_raw_nodes.Postprocessing):
    pass


@dataclass
class InputTensor(Node, model_raw_nodes.InputTensor):
    axes: Tuple[str, ...] = missing

    def __post_init__(self):
        super().__post_init__()
        # raw node has string with single letter axes names. Here we use a tuple of string here (like xr.DataArray).
        self.axes = tuple(self.axes)


@dataclass
class OutputTensor(Node, model_raw_nodes.OutputTensor):
    axes: Tuple[str, ...] = missing

    def __post_init__(self):
        super().__post_init__()
        # raw node has string with single letter axes names. Here we use a tuple of string here (like xr.DataArray).
        self.axes = tuple(self.axes)


@dataclass
class ImportedSource(Node):
    factory: Callable

    def __call__(self, *args, **kwargs):
        return self.factory(*args, **kwargs)


@dataclass
class KerasHdf5WeightsEntry(Node, model_raw_nodes.KerasHdf5WeightsEntry):
    source: Path = missing


@dataclass
class OnnxWeightsEntry(Node, model_raw_nodes.OnnxWeightsEntry):
    source: Path = missing


@dataclass
class PytorchStateDictWeightsEntry(Node, model_raw_nodes.PytorchStateDictWeightsEntry):
    source: Path = missing
    architecture: Union[_Missing, ImportedSource] = missing


@dataclass
class PytorchScriptWeightsEntry(Node, model_raw_nodes.PytorchScriptWeightsEntry):
    source: Path = missing


@dataclass
class TensorflowJsWeightsEntry(Node, model_raw_nodes.TensorflowJsWeightsEntry):
    source: Path = missing


@dataclass
class TensorflowSavedModelBundleWeightsEntry(Node, model_raw_nodes.TensorflowSavedModelBundleWeightsEntry):
    source: Path = missing


@dataclass
class Attachments(Node, model_raw_nodes.Attachments):
    files: List[Path] = missing


WeightsEntry = Union[
    KerasHdf5WeightsEntry,
    OnnxWeightsEntry,
    PytorchScriptWeightsEntry,
    PytorchStateDictWeightsEntry,
    TensorflowJsWeightsEntry,
    TensorflowSavedModelBundleWeightsEntry,
]


@dataclass
class Model(model_raw_nodes.Model, RDF, Node):
    test_inputs: List[Path] = missing
    test_outputs: List[Path] = missing
    weights: Dict[model_raw_nodes.WeightsFormat, WeightsEntry] = missing
