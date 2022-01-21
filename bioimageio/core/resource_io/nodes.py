import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from marshmallow import missing
from marshmallow.utils import _Missing

from bioimageio.spec.model import raw_nodes as model_raw_nodes
from bioimageio.spec.rdf import raw_nodes as rdf_raw_nodes
from bioimageio.spec.collection import raw_nodes as collection_raw_nodes
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
class CiteEntry(Node, rdf_raw_nodes.CiteEntry):
    pass


@dataclass
class Author(Node, model_raw_nodes.Author):
    pass


@dataclass
class Maintainer(Node, model_raw_nodes.Maintainer):
    pass


@dataclass
class Badge(Node, rdf_raw_nodes.Badge):
    pass


@dataclass
class RDF(rdf_raw_nodes.RDF, ResourceDescription):
    badges: Union[_Missing, List[Badge]] = missing
    covers: Union[_Missing, List[Path]] = missing


@dataclass
class CollectionEntry(Node, collection_raw_nodes.CollectionEntry):
    source: URI = missing


@dataclass
class ModelParent(Node, model_raw_nodes.ModelParent):
    pass


@dataclass
class Collection(collection_raw_nodes.Collection, RDF):
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
class TorchscriptWeightsEntry(Node, model_raw_nodes.TorchscriptWeightsEntry):
    source: Path = missing


@dataclass
class TensorflowJsWeightsEntry(Node, model_raw_nodes.TensorflowJsWeightsEntry):
    source: Path = missing


@dataclass
class TensorflowSavedModelBundleWeightsEntry(Node, model_raw_nodes.TensorflowSavedModelBundleWeightsEntry):
    source: Path = missing


@dataclass
class Attachments(Node, rdf_raw_nodes.Attachments):
    files: Union[_Missing, List[Path]] = missing
    unknown: Union[_Missing, Dict[str, Any]] = missing


WeightsEntry = Union[
    KerasHdf5WeightsEntry,
    OnnxWeightsEntry,
    PytorchStateDictWeightsEntry,
    TensorflowJsWeightsEntry,
    TensorflowSavedModelBundleWeightsEntry,
    TorchscriptWeightsEntry,
]


@dataclass
class Model(model_raw_nodes.Model, RDF):
    authors: List[Author] = missing
    maintainers: Union[_Missing, List[Maintainer]] = missing
    test_inputs: List[Path] = missing
    test_outputs: List[Path] = missing
    weights: Dict[model_raw_nodes.WeightsFormat, WeightsEntry] = missing
