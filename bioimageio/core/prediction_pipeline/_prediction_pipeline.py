import abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

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
    """

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


class _PredictionPipelineImpl(PredictionPipeline):
    def __init__(
        self, *, name: str, bioimageio_model: Model, processing: CombinedProcessing, model: ModelAdapter
    ) -> None:
        super().__init__(bioimageio_model=bioimageio_model)
        if bioimageio_model.run_mode:
            raise NotImplementedError(f"Not yet implemented inference for run mode '{bioimageio_model.run_mode.name}'")

        self._name = name
        self._input_specs = bioimageio_model.inputs
        self._output_specs = bioimageio_model.outputs
        self._processing = processing
        self._model: ModelAdapter = model

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

    def __call__(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        return self.forward(*input_tensors)


def enforce_min_shape(min_shape, step, axes):
    """Hack: pick a bigger shape than min shape

    Some models come with super tiny minimal shapes, that make the processing
    too slow. While dryrun is not implemented, we'll "guess" a sensible shape
    and hope it will fit into memory.

    """
    MIN_SIZE_2D = 256
    MIN_SIZE_3D = 64

    assert len(min_shape) == len(step) == len(axes)

    spacial_increments = sum(i != 0 for i, a in zip(step, axes) if a in "xyz")
    if spacial_increments > 2:
        target_size = MIN_SIZE_3D
    else:
        target_size = MIN_SIZE_2D

    factors = [math.ceil((target_size - s) / i) for s, i, a in zip(min_shape, step, axes) if a in "xyz"]
    if sum(f > 0 for f in factors) == 0:
        return min_shape

    m = max(factors)
    return [s + i * m for s, i in zip(min_shape, step)]


def create_prediction_pipeline(
    *, bioimageio_model: nodes.Model, devices: Optional[List[str]] = None, weight_format: Optional[str] = None
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
