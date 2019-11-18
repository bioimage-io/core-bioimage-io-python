import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from pybio.spec.spec import Spec, InputTensorSpec, OutputTensorSpec
from pybio.spec.common import CommonSpec
from pybio.spec.reader import ReaderSpec
from pybio.spec.sampler import SamplerSpec
from pybio.spec.source import SpecWithSource
from pybio.spec.transformation import TransformationSpec

logger = logging.getLogger(__name__)


def resolve_path_str(path: str, rel_to_file_path: Path) -> str:
    if path.startswith("."):
        return (rel_to_file_path / path).resolve().as_posix()
    else:
        return path


class PredictionSpec(Spec):
    def __init__(
        self,
        _rel_path: Path,
        weights: Dict[str, Any],
        preprocess: List[Union[str, Dict[str, Any]]],
        postprocess: List[Union[str, Dict[str, Any]]],
        dependencies: Optional[str] = None,
    ):
        super().__init__(name="prediction", _rel_path=_rel_path)
        self.weights = SpecWithSource(name="prediction weights", _rel_path=_rel_path, **weights)
        self.preprocess = preprocess
        self.postprocess = postprocess
        if dependencies is not None:
            logger.warning("handling dependencies not yet implemented")


class TrainingSetupSpec(Spec):
    def __init__(
        self,
        _rel_path: Path,
        reader: Dict[str, Any],
        sampler: Dict[str, Any],
        preprocess: List[Dict[str, Any]],
        loss: List[Dict[str, Any]],
        optimizer: Dict[str, Any],
    ):
        assert isinstance(preprocess, list), type(preprocess)
        assert all(isinstance(el, dict) for el in preprocess), [type(el) for el in preprocess]
        assert isinstance(loss, list), type(loss)
        assert all(isinstance(el, dict) for el in loss), [type(el) for el in loss]

        super().__init__(name="training setup", _rel_path=_rel_path)
        self.reader = ReaderSpec.load(_rel_path=_rel_path, **reader)
        self.sampler = SamplerSpec.load(_rel_path=_rel_path, **sampler)

        self.preprocess = [TransformationSpec.load(_rel_path=_rel_path, **p) for p in preprocess]
        self.loss = [TransformationSpec.load(_rel_path=_rel_path, **p) for p in loss]
        self.optimizer = SpecWithSource(name="optimizer", _rel_path=_rel_path, **optimizer)


class TrainingSpec(SpecWithSource):
    def __init__(self, _rel_path: Path, setup: Dict[str, Any], dependencies: str, description: str, **super_kwargs):

        super().__init__(name="training", _rel_path=_rel_path, **super_kwargs)
        self.setup = TrainingSetupSpec(_rel_path=_rel_path, **setup)
        self.dependencies = dependencies  # todo: handle dependencies
        self.description = description


class ModelSpec(CommonSpec):
    """Language specific interpretation of a .model.yaml specification """

    def __init__(
        self,
        _rel_path: Path,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        prediction: Dict[str, Any],
        training: Optional[Dict[str, Any]] = None,
        **super_kwargs
    ):
        super().__init__(_rel_path=_rel_path, **super_kwargs)
        self.inputs = [InputTensorSpec(**el) for el in inputs]
        self.outputs = [OutputTensorSpec(**el) for el in outputs]

        self.prediction = PredictionSpec(_rel_path=_rel_path, **prediction)
        self.training = None if training is None else TrainingSpec(_rel_path=_rel_path, **training)


def parse_model_spec(model_spec_path: str):
    return ModelSpec.load(model_spec_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # model_spec = ModelSpec.from_yaml("file:/repos/configuration/models/UNet2dExample.model.yaml")
    model_spec = ModelSpec.load(
        "/repos/example-unet-configurations/models/unet-2d-nuclei-broad/UNet2DNucleiBroad.model.yaml"
    )
    # model_config = ModelConfig.from_spec(model_spec)
