import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from pycore.spec import Spec, Source, InputTensorSpec, OutputTensorSpec, StandardSpec
from pycore.spec.reader import ReaderSpec
from pycore.spec.sampler import SamplerSpec


@dataclass
class WeightsSpec(Spec):
    source: str
    hash: Dict[str, str]


@dataclass
class PredictionSpec(Spec):
    weights: WeightsSpec
    preprocess: List[Union[str, Dict[str, Any]]]
    postprocess: List[Union[str, Dict[str, Any]]]

    dependencies: str  # conda:./environment.yaml  # this is a file to the dependencies

    @classmethod
    def interpret(cls, file_path: Path, spec_kwargs: Dict[str, Any]):
        spec_kwargs["weights"] = WeightsSpec.from_dict(file_path, spec_kwargs["weights"])
        return super().interpret(file_path, spec_kwargs)


@dataclass
class TrainingSetupSpec(Spec):
    reader: ReaderSpec
    sampler: SamplerSpec
    #   spec: ../samplers/RandomBatch.sampler.yaml  # return an iteratable and random accessible object with `length`, `position` and other properties
    #   kwargs: {seed: 123, output_shape: [1, 2, 256, 256], overlap: [0, 0, 100, 100]}
    # preprocess:
    #   - spec: ../transformations/NormalizeZeroMeanUnitVariance.transformation.yaml
    #     kwargs: {eps: 1.0e-6, apply_to: [0]}
    # loss:
    #   # how can we specify an 'object'?
    #   # local ../../trafos/my_trafo.transformation.yaml
    #   # repo https:://github.com/..../:my_trafor.transformation.yaml@1.0.0
    #   # Sigmoid (fetch the name from the 'pytorch-core' for the given language and framework)
    #   - source: ../transformations/Sigmoid.transformation.yaml
    #     kwargs: {apply_to: [0]}
    #   - source: ../transformations/BCELoss.transformation.yaml


@dataclass
class TrainingSpec(Spec):
    setup: TrainingSetupSpec
    optimizer: Dict[str, Any]
    # validation:
    #   - {}
    # callback_engine:
    #    - {}
    source: Source
    kwargs: Dict[str, Any]
    # enable different ways of specifying the dependencies.
    # this would hold all training dependencies, e.g. as a frozen conda environment
    # or as a pom.xml
    dependencies: str
    description: str


@dataclass
class ModelSpec(StandardSpec):
    """Language specific interpretation of a .model.yaml specification """

    inputs: List[InputTensorSpec]
    outputs: List[OutputTensorSpec]
    prediction: PredictionSpec

    thumbnail: Optional[str]
    test_input: Optional[str]
    test_output: Optional[str]
    training: Optional[Dict[str, Any]]

    @classmethod
    def interpret(cls, file_path, spec_dict):
        # load subspecs
        spec_dict["inputs"] = [InputTensorSpec.from_dict(file_path, ipt) for ipt in spec_dict["inputs"]]
        spec_dict["outputs"] = [OutputTensorSpec.from_dict(file_path, ipt) for ipt in spec_dict["outputs"]]
        spec_dict["prediction"] = PredictionSpec.from_dict(file_path, spec_dict["prediction"])

        return super().interpret(file_path, spec_dict)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model_spec = ModelSpec.from_yaml("file:/repos/configuration/models/UNet2dExample.model.yaml")
    # model_config = ModelConfig.from_spec(model_spec)
