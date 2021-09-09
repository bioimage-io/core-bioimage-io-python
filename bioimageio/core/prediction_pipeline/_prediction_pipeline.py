import abc
import math
from typing import List, Optional, Tuple

import xarray as xr
from bioimageio.core.resource_io import nodes
from marshmallow import missing

from ._model_adapters import ModelAdapter, create_model_adapter
from ._postprocessing import make_postprocessing

# from ._preprocessing import ADD_BATCH_DIM, make_ensure_dtype_preprocessing
from ._preprocessing import make_preprocessing
from ._types import Transform


class PredictionPipeline(ModelAdapter):
    """
    Represents model computation including preprocessing and postprocessing
    """

    @abc.abstractmethod
    def forward(self, input_tensor: xr.DataArray) -> xr.DataArray:
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
    def input_axes(self) -> str:
        """
        Input axes excepted by this pipeline
        Note: one character axes names
        """
        ...

    @property
    @abc.abstractmethod
    def input_shape(self) -> List[Tuple[str, int]]:
        """
        Named input dimensions
        """
        ...

    @property
    @abc.abstractmethod
    def output_axes(self) -> str:
        """
        Output axes of this pipeline
        Note: one character axes names
        """
        ...

    @property
    @abc.abstractmethod
    def halo(self) -> List[Tuple[str, int]]:
        """
        Size of output borders that have unreliable data due to artifacts
        """
        ...

    @property
    @abc.abstractmethod
    def scale(self) -> List[Tuple[str, float]]:
        """
        Scale of output tensor relative to input
        """
        ...

    @property
    @abc.abstractmethod
    def offset(self) -> List[Tuple[str, int]]:
        """
        Offset of output tensor relative to input
        """
        ...


class _PredictionPipelineImpl(PredictionPipeline):
    def __init__(
        self,
        *,
        name: str,
        input_axes: str,  # TODO shouldn't this be a list for multple input tensors?
        input_shape: List[Tuple[str, int]],
        output_axes: str,
        halo: List[Tuple[str, int]],
        scale: List[Tuple[str, float]],
        offset: List[Tuple[str, int]],
        preprocessing: Transform,
        model: ModelAdapter,
        postprocessing: Transform,
    ) -> None:
        self._name = name
        self._halo = halo
        self._scale = scale
        self._offset = offset
        self._input_axes = input_axes
        self._output_axes = output_axes
        self._input_shape = input_shape
        self._preprocessing = preprocessing
        self._model: ModelAdapter = model
        self._postprocessing = postprocessing

    @property
    def name(self):
        return self._name

    @property
    def halo(self):
        return self._halo

    @property
    def scale(self):
        return self._scale

    @property
    def offset(self):
        return self._offset

    @property
    def input_axes(self):
        return self._input_axes

    @property
    def output_axes(self):
        return self._output_axes

    @property
    def input_shape(self):
        return self._input_shape

    # todo: separate preprocessing/actual forward/postprocessing
    def forward(self, input_tensor: xr.DataArray) -> xr.DataArray:
        preprocessed = self._preprocessing(input_tensor)
        prediction = self._model.forward(preprocessed)
        return self._postprocessing(prediction)


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
    if len(bioimageio_model.inputs) != 1 or len(bioimageio_model.outputs) != 1:
        raise NotImplementedError("Only models with single input and output are supported")

    model_adapter: ModelAdapter = create_model_adapter(
        bioimageio_model=bioimageio_model, devices=devices, weight_format=weight_format
    )

    input = bioimageio_model.inputs[0]
    input_axes = input.axes
    try:
        input_shape = input.shape.min
        step = input.shape.step
        input_shape = enforce_min_shape(input_shape, step, input_axes)
    except AttributeError:
        input_shape = input.shape

    preprocessing_spec = [] if input.preprocessing is missing else input.preprocessing.copy()

    input_named_shape = list(zip(input_axes, input_shape))
    preprocessing: Transform = make_preprocessing(preprocessing_spec)

    output = bioimageio_model.outputs[0]
    # TODO are we using the halo here at all?
    halo_shape = output.halo or [0 for _ in output.axes]
    output_axes = bioimageio_model.outputs[0].axes
    # TODO don't we also have fixed output shape?
    scale = output.shape.scale
    offset = output.shape.offset
    postprocessing_spec = [] if output.postprocessing is missing else output.postprocessing.copy()
    halo_named_shape = list(zip(output_axes, halo_shape))

    if isinstance(output.shape, list):
        raise NotImplementedError("Expected implicit output shape")

    named_scale = list(zip(output_axes, scale))
    named_offset = list(zip(output_axes, offset))

    postprocessing: Transform = make_postprocessing(postprocessing_spec)

    return _PredictionPipelineImpl(
        name=bioimageio_model.name,
        input_axes=input_axes,
        input_shape=input_named_shape,
        output_axes=output_axes,
        halo=halo_named_shape,
        scale=named_scale,
        offset=named_offset,
        preprocessing=preprocessing,
        model=model_adapter,
        postprocessing=postprocessing,
    )
