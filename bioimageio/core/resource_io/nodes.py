import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Union

from marshmallow import missing
from marshmallow.utils import _Missing

from bioimageio.spec.model.v0_3 import base_nodes as model_base_nodes
from bioimageio.spec.rdf.v0_2 import base_nodes as rdf_base_nodes
from bioimageio.spec.shared import base_nodes
from bioimageio.spec.shared.common import DataClassFilterUnknownKwargsMixin


@dataclass
class Node(base_nodes.NodeBase):
    pass


@dataclass
class ResourceDescription(Node, base_nodes.ResourceDescription):
    pass


@dataclass
class URI(Node, base_nodes.URI):
    pass


@dataclass
class ImplicitInputShape(Node, base_nodes.ImplicitInputShape):
    pass


@dataclass
class ImplicitOutputShape(Node, base_nodes.ImplicitOutputShape):
    pass


@dataclass
class Dependencies(Node, base_nodes.Dependencies):
    file: pathlib.Path = missing


@dataclass
class LocalImportableModule(Node, base_nodes.ImportableModule):
    """intermediate between raw_nodes.ImportableModule and nodes.ImportedSource. Used by SourceNodeTransformer"""

    root_path: pathlib.Path = missing


@dataclass
class ResolvedImportableSourceFile(Node, base_nodes.ImportableSourceFile):
    """intermediate between raw_nodes.ImportableSourceFile and nodes.ImportedSource. Used by SourceNodeTransformer"""

    source_file: pathlib.Path = missing


@dataclass
class CiteEntry(Node, rdf_base_nodes.CiteEntry):
    pass


@dataclass
class Author(Node, rdf_base_nodes.Author):
    pass


@dataclass
class Badge(Node, rdf_base_nodes.Badge):
    pass


# to pass mypy:
# separate dataclass and abstract class as a workaround for abstract dataclasses
# from https://github.com/python/mypy/issues/5374#issuecomment-650656381
@dataclass  # use super init to allow for additional unknown kwargs
class _RDF(Node, rdf_base_nodes._RDF):
    covers: Union[_Missing, List[Path]] = missing


class RDF(_RDF, rdf_base_nodes.RDF, DataClassFilterUnknownKwargsMixin):
    def __init__(self, **kwargs):  # todo: improve signature
        known_kwargs = self.get_known_kwargs(kwargs)
        super().__init__(**known_kwargs)


@dataclass
class CollectionEntry(Node, rdf_base_nodes.CollectionEntry):
    source: URI = missing


@dataclass
class ModelCollectionEntry(CollectionEntry, rdf_base_nodes.ModelCollectionEntry):
    download_url: URI = missing


@dataclass
class Collection(RDF, rdf_base_nodes.Collection):
    pass


@dataclass
class RunMode(Node, model_base_nodes.RunMode):
    pass


@dataclass
class Preprocessing(Node, model_base_nodes.Preprocessing):
    pass


@dataclass
class Postprocessing(Node, model_base_nodes.Postprocessing):
    pass


@dataclass
class InputTensor(Node, model_base_nodes.InputTensor):
    pass


@dataclass
class OutputTensor(Node, model_base_nodes.OutputTensor):
    pass


@dataclass
class WeightsEntryBase(Node, model_base_nodes.WeightsEntryBase):
    source: Path = missing


@dataclass
class KerasHdf5WeightsEntry(WeightsEntryBase, model_base_nodes.KerasHdf5WeightsEntry):
    pass


@dataclass
class OnnxWeightsEntry(WeightsEntryBase, model_base_nodes.OnnxWeightsEntry):
    pass


@dataclass
class PytorchStateDictWeightsEntry(WeightsEntryBase, model_base_nodes.PytorchStateDictWeightsEntry):
    pass


@dataclass
class PytorchScriptWeightsEntry(WeightsEntryBase, model_base_nodes.PytorchScriptWeightsEntry):
    pass


@dataclass
class TensorflowJsWeightsEntry(WeightsEntryBase, model_base_nodes.TensorflowJsWeightsEntry):
    pass


@dataclass
class TensorflowSavedModelBundleWeightsEntry(WeightsEntryBase, model_base_nodes.TensorflowSavedModelBundleWeightsEntry):
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
class Model(model_base_nodes.Model, RDF, Node):
    source: Union[_Missing, ImportedSource] = missing
    test_inputs: List[Path] = missing
    test_outputs: List[Path] = missing
    weights: Dict[model_base_nodes.WeightsFormat, WeightsEntry] = missing
