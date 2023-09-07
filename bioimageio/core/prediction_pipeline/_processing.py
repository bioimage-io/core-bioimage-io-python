"""Here pre- and postprocessing operations are implemented according to their definitions in bioimageio.spec:
see https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/preprocessing_spec_latest.md
and https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/postprocessing_spec_latest.md
"""
from abc import ABC, abstractmethod
import numbers
from dataclasses import InitVar, dataclass, field, fields
from typing import Dict, Generic,  Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union
from typing_extensions import Self

import numpy
import numpy as np
from pydantic import model_validator  # type: ignore
from pydantic import field_validator
import xarray as xr
from bioimageio.spec._internal.base_nodes import Node
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import Processing as ProcessingSpec, ProcessingKwargs, Binarize, Clip
from bioimageio.spec.model.v0_5 import TensorId
from numpy.typing import DTypeLike
from bioimageio.core.statistical_measures import Mean, Measure, Percentile, Std, MeasureValue
from ._utils import FIXED, PER_DATASET, PER_SAMPLE, DatasetMode, Mode, RequiredMeasure, SampleMode, Sample

from typing import Literal, TypedDict, get_args

def _get_fixed(
    fixed: Union[float, Sequence[float]], tensor: xr.DataArray, axes: Optional[Sequence[str]]
) -> Union[float, xr.DataArray]:
    if axes is None:
        return fixed

    fixed_shape = tuple(s for d, s in tensor.sizes.items() if d not in axes)
    fixed_dims = tuple(d for d in tensor.dims if d not in axes)
    fixed = np.array(fixed).reshape(fixed_shape)
    return xr.DataArray(fixed, dims=fixed_dims)



PKwargs = TypeVar("PKwargs", bound=ProcessingKwargs)
ProcInput = TypeVar("ProcInput", xr.DataArray, Sample)

class ProcessingBase(Node, Generic[PKwargs], ABC, frozen=True):
    """base class for all Pre- and Postprocessing transformations."""

    tensor_id: TensorId
    """id of tensor to operate on"""
    kwargs: PKwargs
    computed_measures: Dict[RequiredMeasure, MeasureValue] = field(default_factory=dict)

    @model_validator(mode="after")
    def check_required_measures_in_computed(self) -> Self:
        for req in self.required_measures:
            if req not in self.computed_measures:
                raise ValueError(f"Missing computed {req}.")

        return self

    @classmethod
    def get_required_measures(cls, tensor_id: TensorId, kwargs: PKwargs) -> Tuple[RequiredMeasure, ...]:
        return ()

    @property
    def required_measures(self) -> Tuple[RequiredMeasure, ...]:
        return self.get_required_measures(tensor_id=self.tensor_id, kwargs=self.kwargs)

    def __call__(self, __input: ProcInput, /) -> ProcInput:
        if isinstance(__input, xr.DataArray):
            return self.apply(__input)
        else:
            return self.apply_to_sample(__input)

    @abstractmethod
    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        """apply processing"""
        ...

    def apply_to_sample(self, sample: Sample) -> Sample:
        ret = dict(sample)
        ret[self.tensor_id] = self.apply(sample[self.tensor_id])
        return ret

class Processing(ProcessingSpec, ProcessingBase[PKwargs], frozen=True):
    pass
#
# Pre- and Postprocessing implementations
#
class NonSpecProcessing(ProcessingBase[PKwargs], frozen=True):
    """processings operations beyond what is currently defined in bioimageio.spec"""
    pass


class AssertDtype(NonSpecProcessing[ProcessingKwargs], frozen=True):
    """Helper Processing to assert dtype."""
    id: Literal["assert_dtype"] = "assert_dtype"

    dtype: Union[str, Sequence[str]]
    _assert_with: Tuple[Type[DTypeLike], ...]

    def __pydantic_postinit__(self):
        if isinstance(self.dtype, str):
            dtype = [self.dtype]
        else:
            dtype = self.dtype

        object.__setattr__(self, "_assert_with", tuple(type(numpy.dtype(dt)) for dt in dtype))

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        assert isinstance(tensor.dtype, self._assert_with)
        return tensor


class Binarize(Processing[BinarizeKwargs]):
    """'output = tensor > threshold'."""

    threshold: float = MISSING  # make dataclass inheritance work for py<3.10 by using an explicit MISSING value.

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor > self.threshold


@dataclass
class Clip(Processing):
    """Limit tensor values to [min, max]."""

    min: float = MISSING
    max: float = MISSING

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor.clip(min=self.min, max=self.max)


@dataclass
class EnsureDtype(Processing):
    """Helper Processing to cast dtype if needed."""

    dtype: str = MISSING

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor.astype(self.dtype)


@dataclass
class ScaleLinear(Processing):
    """Scale the tensor with a fixed multiplicative and additive factor."""

    gain: Union[float, Sequence[float]] = MISSING
    offset: Union[float, Sequence[float]] = MISSING
    axes: Optional[Sequence[str]] = None

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        scale_axes = tuple(ax for ax in tensor.dims if (ax not in self.axes and ax != "b"))
        if scale_axes:
            gain = xr.DataArray(np.atleast_1d(self.gain), dims=scale_axes)
            offset = xr.DataArray(np.atleast_1d(self.offset), dims=scale_axes)
        else:
            gain = self.gain
            offset = self.offset

        return tensor * gain + offset

    def __post_init__(self):
        super().__post_init__()
        if self.axes is None:
            assert isinstance(self.gain, (int, float))
            assert isinstance(self.offset, (int, float))


@dataclass
class ScaleMeanVariance(Processing):
    """Scale the tensor s.t. its mean and variance match a reference tensor."""

    mode: Literal[SampleMode, DatasetMode] = PER_SAMPLE
    reference_tensor: TensorName = MISSING
    axes: Optional[Sequence[str]] = None
    eps: float = 1e-6

    def get_required_measures(self) -> RequiredMeasures:
        axes = None if self.axes is None else tuple(self.axes)
        return {
            self.mode: {
                self.tensor_id: {Mean(axes=axes), Std(axes=axes)},
                self.reference_tensor: {Mean(axes=axes), Std(axes=axes)},
            }
        }

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        axes = None if self.axes is None else tuple(self.axes)
        assert self.mode in (PER_SAMPLE, PER_DATASET)
        mean = self.get_computed_measure(self.tensor_id, Mean(axes), mode=self.mode)
        std = self.get_computed_measure(self.tensor_id, Std(axes), mode=self.mode)
        ref_mean = self.get_computed_measure(self.reference_tensor, Mean(axes), mode=self.mode)
        ref_std = self.get_computed_measure(self.reference_tensor, Std(axes), mode=self.mode)

        return (tensor - mean) / (std + self.eps) * (ref_std + self.eps) + ref_mean


@dataclass
class ScaleRange(Processing):
    """Scale with percentiles."""

    mode: Literal[SampleMode, DatasetMode] = PER_SAMPLE
    axes: Optional[Sequence[str]] = None
    min_percentile: float = 0.0
    max_percentile: float = 100.0
    eps: float = 1e-6
    reference_tensor: Optional[TensorName] = None

    def get_required_measures(self) -> RequiredMeasures:
        axes = None if self.axes is None else tuple(self.axes)
        measures = {Percentile(self.min_percentile, axes=axes), Percentile(self.max_percentile, axes=axes)}
        return {self.mode: {self.reference_tensor or self.tensor_id: measures}}

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        ref_name = self.reference_tensor or self.tensor_id
        axes = None if self.axes is None else tuple(self.axes)
        v_lower = self.get_computed_measure(ref_name, Percentile(self.min_percentile, axes=axes))
        v_upper = self.get_computed_measure(ref_name, Percentile(self.max_percentile, axes=axes))

        return (tensor - v_lower) / (v_upper - v_lower + self.eps)

    def __post_init__(self):
        super().__post_init__()
        self.axes = None if self.axes is None else tuple(self.axes)  # make sure axes is Tuple[str] or None


@dataclass
class Sigmoid(Processing):
    """1 / (1 + e^(-tensor))."""

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return 1.0 / (1.0 + np.exp(-tensor))


@dataclass
class ZeroMeanUnitVariance(Processing):
    """normalize to zero mean, unit variance."""

    mode: Mode = PER_SAMPLE
    mean: Optional[Union[float, Sequence[float]]] = None
    std: Optional[Union[float, Sequence[float]]] = None
    axes: Optional[Sequence[str]] = None
    eps: float = 1.0e-6

    def get_required_measures(self) -> RequiredMeasures:
        if self.mode == FIXED:
            return {}
        else:
            axes = None if self.axes is None else tuple(self.axes)
            return {self.mode: {self.tensor_id: {Mean(axes=axes), Std(axes=axes)}}}

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        axes = None if self.axes is None else tuple(self.axes)
        if self.mode == FIXED:
            assert self.mean is not None and self.std is not None
            mean = _get_fixed(self.mean, tensor, axes)
            std = _get_fixed(self.std, tensor, axes)
        elif self.mode in (PER_SAMPLE, PER_DATASET):
            assert self.mean is None and self.std is None
            mean = self.get_computed_measure(self.tensor_id, Mean(axes), mode=self.mode)
            std = self.get_computed_measure(self.tensor_id, Std(axes), mode=self.mode)
        else:
            raise ValueError(self.mode)

        return (tensor - mean) / (std + self.eps)


class _KNOWN_PREPROCESSING(TypedDict):

class _KnownProcessing(TypedDict):
    pre: Mapping[PreprocessingName, Type[Processing]]
    post: Mapping[PostprocessingName, Type[Processing]]

KNOWN_PROCESSING = _KnownProcessing(
    pre={
        "binarize": Binarize,
        "clip": Clip,
        "scale_linear": ScaleLinear,
        "scale_range": ScaleRange,
        "sigmoid": Sigmoid,
        "zero_mean_unit_variance": ZeroMeanUnitVariance,
    },
    post={
        "binarize": Binarize,
        "clip": Clip,
        "scale_linear": ScaleLinear,
        "scale_mean_variance": ScaleMeanVariance,
        "scale_range": ScaleRange,
        "sigmoid": Sigmoid,
        "zero_mean_unit_variance": ZeroMeanUnitVariance,
    },
)
