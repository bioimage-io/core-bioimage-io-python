import abc
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import xarray as xr
from bioimageio.core.resource_io import nodes
from marshmallow import missing

from ._model_adapters import ModelAdapter, create_model_adapter
from ._postprocessing import make_postprocessing

from ._preprocessing import make_preprocessing
from ._types import Transform
from ..resource_io.nodes import ImplicitOutputShape, InputTensor, Model, OutputTensor


@dataclass
class NamedImplicitOutputShape:
    reference_input: str = missing
    scale: List[Tuple[str, float]] = missing
    offset: List[Tuple[str, int]] = missing

    def __len__(self):
        return len(self.scale)


class PredictionPipeline(ModelAdapter):
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
        self,
        *,
        name: str,
        bioimageio_model: Model,
        preprocessing: Sequence[Transform],
        model: ModelAdapter,
        postprocessing: Sequence[Transform],
    ) -> None:
        self._name = name
        self._input_specs = bioimageio_model.inputs
        self._output_specs = bioimageio_model.outputs
        self._preprocessing = preprocessing
        self._model: ModelAdapter = model
        self._postprocessing = postprocessing

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
        assert len(self._preprocessing) == len(input_tensors)
        preprocessed = [fn(x) for fn, x in zip(self._preprocessing, input_tensors)]
        prediction = self.predict(*preprocessed)
        assert len(self._postprocessing) == len(prediction)
        return [fn(x) for fn, x in zip(self._postprocessing, prediction)]

    def preprocess(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """Apply preprocessing."""
        assert len(self._preprocessing) == len(input_tensors)
        return [fn(x) for fn, x in zip(self._preprocessing, input_tensors)]

    def postprocess(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """Apply postprocessing."""
        assert len(self._postprocessing) == len(input_tensors)
        return [fn(x) for fn, x in zip(self._postprocessing, input_tensors)]

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

    preprocessing: List[Transform] = []
    for ipt in bioimageio_model.inputs:
        try:
            input_shape = ipt.shape.min
            step = ipt.shape.step
            input_shape = enforce_min_shape(input_shape, step, ipt.axes)
        except AttributeError:
            input_shape = ipt.shape

        preprocessing_spec = [] if ipt.preprocessing is missing else ipt.preprocessing.copy()
        preprocessing.append(make_preprocessing(preprocessing_spec))

    postprocessing: List[Transform] = []
    for out in bioimageio_model.outputs:
        postprocessing_spec = [] if out.postprocessing is missing else out.postprocessing.copy()
        postprocessing.append(make_postprocessing(postprocessing_spec))

    return _PredictionPipelineImpl(
        name=bioimageio_model.name,
        bioimageio_model=bioimageio_model,
        preprocessing=preprocessing,
        model=model_adapter,
        postprocessing=postprocessing,
    )
