import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Union

from marshmallow import missing
from marshmallow.utils import _Missing

from bioimageio.spec.model.v0_3 import raw_nodes as model_raw_nodes
from bioimageio.spec.rdf.v0_2 import raw_nodes as rdf_raw_nodes
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
    pass


@dataclass
class OutputTensor(Node, model_raw_nodes.OutputTensor):
    pass


@dataclass
class _WeightsEntryBase(Node, model_raw_nodes._WeightsEntryBase):
    source: Path = missing


@dataclass
class KerasHdf5WeightsEntry(_WeightsEntryBase, model_raw_nodes.KerasHdf5WeightsEntry):
    pass


@dataclass
class OnnxWeightsEntry(_WeightsEntryBase, model_raw_nodes.OnnxWeightsEntry):
    pass


@dataclass
class PytorchStateDictWeightsEntry(_WeightsEntryBase, model_raw_nodes.PytorchStateDictWeightsEntry):
    pass


@dataclass
class PytorchScriptWeightsEntry(_WeightsEntryBase, model_raw_nodes.PytorchScriptWeightsEntry):
    pass


@dataclass
class TensorflowJsWeightsEntry(_WeightsEntryBase, model_raw_nodes.TensorflowJsWeightsEntry):
    pass


@dataclass
class TensorflowSavedModelBundleWeightsEntry(_WeightsEntryBase, model_raw_nodes.TensorflowSavedModelBundleWeightsEntry):
    pass


WeightsEntry = Union[
    KerasHdf5WeightsEntry,
    OnnxWeightsEntry,
    PytorchScriptWeightsEntry,
    PytorchStateDictWeightsEntry,
    TensorflowJsWeightsEntry,
    TensorflowSavedModelBundleWeightsEntry,
]


@dataclass
class ImportedSource(Node):
    factory: Callable

    def __call__(self, *args, **kwargs):
        return self.factory(*args, **kwargs)


@dataclass
class Model(model_raw_nodes.Model, RDF, Node):
    source: Union[_Missing, ImportedSource] = missing
    test_inputs: List[Path] = missing
    test_outputs: List[Path] = missing
    weights: Dict[model_raw_nodes.WeightsFormat, WeightsEntry] = missing
