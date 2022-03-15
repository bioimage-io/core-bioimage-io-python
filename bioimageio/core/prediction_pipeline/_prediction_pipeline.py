import abc
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any

import xarray as xr
from marshmallow import missing

from bioimageio.core.resource_io import nodes
from bioimageio.core.statistical_measures import Measure
from ._combined_processing import CombinedProcessing
from ._model_adapters import ModelAdapter, create_model_adapter
from ..resource_io.nodes import InputTensor, Model, OutputTensor


@dataclass
class NamedImplicitOutputShape:
    reference_input: str = missing
    scale: List[Tuple[str, float]] = missing
    offset: List[Tuple[str, int]] = missing

    def __len__(self):
        return len(self.scale)


class PredictionPipeline(abc.ABC):
    """
    Represents model computation including preprocessing and postprocessing
    Note: Ideally use the PredictionPipeline as a context manager
    """

    @abc.abstractmethod
    def __enter__(self):
        ...

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

    @abc.abstractmethod
    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """
        Compute predictions
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the pipeline
        """
        ...

    @property
    @abc.abstractmethod
    def input_specs(self) -> List[InputTensor]:
        """
        specs of inputs
        """
        ...

    @property
    @abc.abstractmethod
    def output_specs(self) -> List[OutputTensor]:
        """
        specs of outputs
        """
        ...

    @abc.abstractmethod
    def load(self) -> None:
        """
        optional step: load model onto devices before calling forward if not using it as context manager
        """
        ...

    @abc.abstractmethod
    def unload(self) -> None:
        """
        free any device memory in use
        """
        ...


class _PredictionPipelineImpl(PredictionPipeline):
    def __init__(
        self, *, name: str, bioimageio_model: Model, processing: CombinedProcessing, model: ModelAdapter
    ) -> None:
        if bioimageio_model.run_mode:
            raise NotImplementedError(f"Not yet implemented inference for run mode '{bioimageio_model.run_mode.name}'")

        self._name = name
        self._input_specs = bioimageio_model.inputs
        self._output_specs = bioimageio_model.outputs
        self._processing = processing
        self._model: ModelAdapter = model

    def __call__(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        return self.forward(*input_tensors)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    @property
    def name(self):
        return self._name

    @property
    def input_specs(self):
        return self._input_specs

    @property
    def output_specs(self):
        return self._output_specs

    def predict(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """Predict input_tensor with the model without applying pre/postprocessing."""
        return self._model.forward(*input_tensors)

    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """Apply preprocessing, run prediction and apply postprocessing."""
        preprocessed, sample_stats = self._processing.apply_preprocessing(*input_tensors)
        prediction = self.predict(*preprocessed)
        return self._processing.apply_postprocessing(*prediction, input_sample_statistics=sample_stats)[0]

    def preprocess(self, *input_tensors: xr.DataArray) -> Tuple[List[xr.DataArray], Dict[str, Dict[Measure, Any]]]:
        """Apply preprocessing."""
        return self._processing.apply_preprocessing(*input_tensors)

    def postprocess(
        self, *input_tensors: xr.DataArray, input_sample_statistics
    ) -> Tuple[List[xr.DataArray], Dict[str, Dict[Measure, Any]]]:
        """Apply postprocessing."""
        return self._processing.apply_postprocessing(*input_tensors, input_sample_statistics=input_sample_statistics)

    def load(self):
        self._model.load()

    def unload(self):
        self._model.unload()


def create_prediction_pipeline(
    *, bioimageio_model: nodes.Model, devices: Optional[Sequence[str]] = None, weight_format: Optional[str] = None
) -> PredictionPipeline:
    """
    Creates prediction pipeline which includes:
    * preprocessing
    * model prediction
    * postprocessing
    """
    model_adapter: ModelAdapter = create_model_adapter(
        bioimageio_model=bioimageio_model, devices=devices, weight_format=weight_format
    )

    processing = CombinedProcessing(bioimageio_model.inputs, bioimageio_model.outputs)

    return _PredictionPipelineImpl(
        name=bioimageio_model.name, bioimageio_model=bioimageio_model, model=model_adapter, processing=processing
    )
