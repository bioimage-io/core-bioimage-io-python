import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from pybio.spec import Spec, InputTensorSpec, OutputTensorSpec, CommonSpec, SpecWithSource
from pybio.spec.reader import ReaderSpec
from pybio.spec.sampler import SamplerSpec

logger = logging.getLogger(__name__)


def resolve_path_str(path: str, rel_to_file_path: Path) -> str:
    if path.startswith("."):
        return (rel_to_file_path / path).resolve().as_posix()
    else:
        return path


class TransformationSpec(Spec):
    def __init__(self, name: str, file_path: Path):
        super().__init__(name=name, file_path=file_path)


class PredictionSpec(Spec):
    def __init__(
        self,
        file_path: Path,
        weights: Dict[str, Any],
        preprocess: List[Union[str, Dict[str, Any]]],
        postprocess: List[Union[str, Dict[str, Any]]],
        dependencies: Optional[str] = None,
    ):
        super().__init__(name="prediction", file_path=file_path)
        self.weights = SpecWithSource(name="prediction weights", file_path=file_path, **weights)
        self.preprocess = preprocess
        self.postprocess = postprocess
        if dependencies is not None:
            logger.warning("handling dependencies not yet implemented")


class TrainingSetupSpec(Spec):
    def __init__(
        self,
        file_path: Path,
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

        super().__init__(name="training setup", file_path=file_path)
        self.reader = ReaderSpec.from_yaml(spec=resolve_path_str(reader["spec"], file_path), **reader["kwargs"])
        self.sampler = ReaderSpec.from_yaml(spec=resolve_path_str(sampler["spec"], file_path), **sampler["kwargs"])

        self.preprocess = [
            TransformationSpec.from_yaml(spec=resolve_path_str(p["spec"], file_path), **p["kwargs"]) for p in preprocess
        ]
        self.loss = [
            TransformationSpec.from_yaml(spec=resolve_path_str(p["spec"], file_path), **p["kwargs"]) for p in loss
        ]
        self.optimizer = SpecWithSource(name="optimizer", **optimizer)


class TrainingSpec(SpecWithSource):
    def __init__(
        self,
        file_path: Path,
        dependencies: str,
        description: str,
        setup: Dict[str, Any],
        source: str,
        hash: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        assert isinstance(dependencies, str), type(dependencies)  # todo: handle dependencies
        assert isinstance(description, str), type(description)

        super().__init__(name="training", file_path=file_path, source=source, hash=hash, kwargs=kwargs)
        self.setup = TrainingSetupSpec(file_path=file_path, **setup)
        self.description = description


class ModelSpec(CommonSpec):
    """Language specific interpretation of a .model.yaml specification """

    def __init__(
        self,
        file_path: Path,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        prediction: Dict[str, Any],
        training: Optional[Dict[str, Any]] = None,
        **common_kwargs
    ):
        super().__init__(file_path=file_path, **common_kwargs)
        self.inputs = [InputTensorSpec(**el) for el in inputs]
        self.outputs = [OutputTensorSpec(**el) for el in outputs]

        self.prediction = PredictionSpec(file_path=file_path, **prediction)
        self.training = None if training is None else TrainingSpec(file_path=file_path, **training)


def parse_model_spec(model_spec_path: Union[str, Path]):
    if not isinstance(model_spec_path, Path):
        model_spec_path = Path(model_spec_path)

    return ModelSpec.from_yaml(model_spec_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model_spec = ModelSpec.from_yaml("file:/repos/configuration/models/UNet2dExample.model.yaml")

    # model_config = ModelConfig.from_spec(model_spec)
