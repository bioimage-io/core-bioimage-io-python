import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from marshmallow import missing
from marshmallow.utils import _Missing

from bioimageio.spec.collection import raw_nodes as collection_raw_nodes
from bioimageio.spec.dataset import raw_nodes as dataset_raw_nodes
from bioimageio.spec.model.v0_4 import raw_nodes as model_raw_nodes
from bioimageio.spec.rdf import raw_nodes as rdf_raw_nodes
from bioimageio.spec.shared import raw_nodes
from bioimageio.spec.workflow import raw_nodes as workflow_raw_nodes


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
class Author(Node, rdf_raw_nodes.Author):
    pass


@dataclass
class Maintainer(Node, rdf_raw_nodes.Maintainer):
    pass


@dataclass
class Badge(Node, rdf_raw_nodes.Badge):
    pass


@dataclass
class Attachments(Node, rdf_raw_nodes.Attachments):
    files: Union[_Missing, List[Path]] = missing
    unknown: Union[_Missing, Dict[str, Any]] = missing


@dataclass
class RDF(rdf_raw_nodes.RDF, ResourceDescription):
    authors: Union[_Missing, List[Author]] = missing
    attachments: Union[_Missing, Attachments] = missing
    badges: Union[_Missing, List[Badge]] = missing
    cite: Union[_Missing, List[CiteEntry]] = missing
    maintainers: Union[_Missing, List[Maintainer]] = missing


@dataclass
class CollectionEntry(Node, collection_raw_nodes.CollectionEntry):
    source: URI = missing


@dataclass
class Collection(collection_raw_nodes.Collection, RDF):
    collection: List[CollectionEntry] = missing


@dataclass
class Dataset(Node, dataset_raw_nodes.Dataset):
    pass


@dataclass
class LinkedDataset(Node, model_raw_nodes.LinkedDataset):
    pass


@dataclass
class ModelParent(Node, model_raw_nodes.ModelParent):
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
    preprocessing: Union[_Missing, List[Preprocessing]] = missing

    def __post_init__(self):
        super().__post_init__()
        # raw node has string with single letter axes names. Here we use a tuple of string here (like xr.DataArray).
        self.axes = tuple(self.axes)


@dataclass
class OutputTensor(Node, model_raw_nodes.OutputTensor):
    axes: Tuple[str, ...] = missing
    postprocessing: Union[_Missing, List[Postprocessing]] = missing

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
class WeightsEntryBase(model_raw_nodes._WeightsEntryBase):
    dependencies: Union[_Missing, Dependencies] = missing


@dataclass
class KerasHdf5WeightsEntry(WeightsEntryBase, model_raw_nodes.KerasHdf5WeightsEntry):
    source: Path = missing


@dataclass
class OnnxWeightsEntry(WeightsEntryBase, model_raw_nodes.OnnxWeightsEntry):
    source: Path = missing


@dataclass
class PytorchStateDictWeightsEntry(WeightsEntryBase, model_raw_nodes.PytorchStateDictWeightsEntry):
    source: Path = missing
    architecture: Union[_Missing, ImportedSource] = missing


@dataclass
class TorchscriptWeightsEntry(WeightsEntryBase, model_raw_nodes.TorchscriptWeightsEntry):
    source: Path = missing


@dataclass
class TensorflowJsWeightsEntry(WeightsEntryBase, model_raw_nodes.TensorflowJsWeightsEntry):
    source: Path = missing


@dataclass
class TensorflowSavedModelBundleWeightsEntry(WeightsEntryBase, model_raw_nodes.TensorflowSavedModelBundleWeightsEntry):
    source: Path = missing


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
    inputs: List[InputTensor] = missing
    outputs: List[OutputTensor] = missing
    parent: Union[_Missing, ModelParent] = missing
    run_mode: Union[_Missing, RunMode] = missing
    test_inputs: List[Path] = missing
    test_outputs: List[Path] = missing
    training_data: Union[_Missing, Dataset, LinkedDataset] = missing
    weights: Dict[model_raw_nodes.WeightsFormat, WeightsEntry] = missing


@dataclass
class Axis(Node, workflow_raw_nodes.Axis):
    pass


@dataclass
class BatchAxis(Node, workflow_raw_nodes.Axis):
    pass


@dataclass
class Input(Node, workflow_raw_nodes.Input):
    pass


@dataclass
class Option(Node, workflow_raw_nodes.Option):
    pass


@dataclass
class Output(Node, workflow_raw_nodes.Output):
    pass


@dataclass
class Workflow(workflow_raw_nodes.Workflow, RDF):
    inputs: List[Input] = missing
    options: List[Option] = missing
    outputs: List[Output] = missing
