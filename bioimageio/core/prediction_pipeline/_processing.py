from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field, fields
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    Generic,
    Literal,
    Mapping,
    NamedTuple,
    NotRequired,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    get_args,
)

import numpy
import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from typing_extensions import LiteralString, Self

from bioimageio.core.statistical_measures import Mean, Measure, MeasureValue, Percentile, Std
from bioimageio.spec._internal.base_nodes import Node, NodeWithExplicitlySetFields
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import TensorId

from ._utils import FIXED, PER_DATASET, PER_SAMPLE, DatasetMode, Mode, RequiredMeasure, Sample, SampleMode

Binarize = Union[v0_4.Binarize, v0_5.Binarize]
BinarizeKwargs = Union[v0_4.BinarizeKwargs, v0_5.BinarizeKwargs]
Clip = Union[v0_4.Clip, v0_5.Clip]
ClipKwargs = Union[v0_4.ClipKwargs, v0_5.ClipKwargs]
EnsureDtypeKwargs = v0_5.EnsureDtypeKwargs
Processing = Union[v0_4.Processing, v0_5.Processing]
ProcessingKwargs = Union[v0_4.ProcessingKwargs, v0_5.ProcessingKwargs]
ScaleLinear = Union[v0_4.ScaleLinear, v0_5.ScaleLinear]
ScaleLinearKwargs = Union[v0_4.ScaleLinearKwargs, v0_5.ScaleLinearKwargs]
ScaleMeanVariance = Union[v0_4.ScaleMeanVariance, v0_5.ScaleMeanVariance]
ScaleMeanVarianceKwargs = Union[v0_4.ScaleMeanVarianceKwargs, v0_5.ScaleMeanVarianceKwargs]
ScaleRange = Union[v0_4.ScaleRange, v0_5.ScaleRange]
ScaleRangeKwargs = Union[v0_4.ScaleRangeKwargs, v0_5.ScaleRangeKwargs]
ZeroMeanUnitVariance = Union[v0_4.ZeroMeanUnitVariance, v0_5.ZeroMeanUnitVariance]
ZeroMeanUnitVarianceKwargs = Union[v0_4.ZeroMeanUnitVarianceKwargs, v0_5.ZeroMeanUnitVarianceKwargs]


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


RCV = TypeVar("RCV", RequiredMeasure, MeasureValue)


@dataclass
class _NamedMeasures(Generic[RCV]):
    def get_set(self) -> Set[RCV]:
        return {getattr(self, f.name) for f in fields(self)}


_NoRequiredMeasures = _NamedMeasures[RequiredMeasure]
_NoMeasureValues = _NamedMeasures[MeasureValue]

R = TypeVar("R", bound=_NamedMeasures[RequiredMeasure])
C = TypeVar("C", bound=_NamedMeasures[MeasureValue])


@dataclass
class ProcessingImplBase(Generic[PKwargs, R, C], ABC):
    """Base class for all Pre- and Postprocessing implementations."""

    tensor_id: TensorId
    """id of tensor to operate on"""
    kwargs: PKwargs
    computed_measures: InitVar[Mapping[RequiredMeasure, MeasureValue]] = field(
        default=MappingProxyType[RequiredMeasure, MeasureValue]({})
    )
    required: R = field(init=False)
    computed: C = field(init=False)

    def __post_init__(self, computed_measures: Mapping[RequiredMeasure, MeasureValue]) -> None:
        self.required = self.get_required_measures(self.tensor_id, self.kwargs)
        selected = {}
        for f in fields(self.required):
            req = getattr(self.required, f.name)
            if req in computed_measures:
                selected[f.name] = computed_measures[req]
            else:
                raise ValueError(f"Missing computed measure: {req} (as '{f.name}').")

    @classmethod
    @abstractmethod
    def get_required_measures(cls, tensor_id: TensorId, kwargs: PKwargs) -> R:
        ...

    @property
    def required_measures(self) -> Set[RequiredMeasure]:
        return self.required.get_set()

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

    def get_spec(self) -> v0_5.Processing:
        raise NotImplementedError


class ProcessingImplBaseWoMeasures(ProcessingImplBase[PKwargs, _NoRequiredMeasures, _NoMeasureValues]):
    @classmethod
    def get_required_measures(cls, tensor_id: TensorId, kwargs: PKwargs) -> _NoRequiredMeasures:
        return _NamedMeasures()


class AssertProcessing(NodeWithExplicitlySetFields, ABC, frozen=True):
    id: str
    kwargs: ProcessingKwargs
    fields_to_set_explicitly: ClassVar[FrozenSet[LiteralString]] = frozenset({"id"})


class AssertDtypeKwargs(ProcessingKwargs, frozen=True):
    dtype: Union[str, Sequence[str]]


class AssertDtype(AssertProcessing, frozen=True):
    id: Literal["assert_dtype"] = "assert_dtype"
    kwargs: AssertDtypeKwargs


class AssertDtypeImpl(ProcessingImplBaseWoMeasures[AssertDtypeKwargs]):
    _assert_with: Tuple[Type[DTypeLike], ...]

    def __post_init__(self, computed_measures: Mapping[RequiredMeasure, MeasureValue]) -> None:
        super().__post_init__(computed_measures)
        if isinstance(self.kwargs.dtype, str):
            dtype = [self.kwargs.dtype]
        else:
            dtype = self.kwargs.dtype

        assert_w = tuple(type(numpy.dtype(dt)) for dt in dtype)
        self._assert_with = assert_w

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        assert isinstance(tensor.dtype, self._assert_with)
        return tensor


class BinarizeImpl(ProcessingImplBaseWoMeasures[BinarizeKwargs]):
    """'output = tensor > threshold'."""

    kwargs: BinarizeKwargs

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor > self.kwargs.threshold

    def get_spec(self):
        return v0_5.Binarize(kwargs=self.kwargs)


class ClipImpl(ProcessingImplBaseWoMeasures[ClipKwargs]):
    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor.clip(min=self.kwargs.min, max=self.kwargs.max)

    def get_spec(self):
        return v0_5.Clip(kwargs=self.kwargs)


class EnsureDtypeImpl(ProcessingImplBaseWoMeasures[EnsureDtypeKwargs]):
    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor.astype(self.kwargs.dtype)


class ScaleLinearImpl(ProcessingImplBaseWoMeasures[ScaleLinearKwargs]):
    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        joint_axes = self.kwargs.axes or ()
        batch_axis_names = ("b", "batch")
        scale_along = tuple(
            ax for ax in tensor.dims if isinstance(ax, str) and ax not in joint_axes and ax not in batch_axis_names
        )
        if scale_along:
            gain = xr.DataArray(np.atleast_1d(self.kwargs.gain), dims=scale_along)
            offset = xr.DataArray(np.atleast_1d(self.kwargs.offset), dims=scale_along)
        else:
            assert isinstance(self.kwargs.gain, float) or len(self.kwargs.gain) == 1
            gain = self.kwargs.gain if isinstance(self.kwargs.gain, float) else self.kwargs.gain[0]
            assert isinstance(self.kwargs.offset, float) or len(self.kwargs.offset) == 1
            offset = self.kwargs.offset if isinstance(self.kwargs.offset, float) else self.kwargs.offset[0]

        return tensor * gain + offset

    def get_spec(self):
        if isinstance(self.kwargs, v0_4.ScaleLinearKwargs):
            raise NotImplementedError

        return v0_5.ScaleLinear(kwargs=self.kwargs)


@dataclass
class _MeanStd(_NamedMeasures[RCV]):
    mean: RCV
    std: RCV


@dataclass
class _MeanStdAndRef(_MeanStd[RCV]):
    ref_mean: RCV
    ref_std: RCV


class ScaleMeanVarianceImpl(
    ProcessingImplBase[ScaleMeanVarianceKwargs, _MeanStdAndRef[RequiredMeasure], _MeanStdAndRef[MeasureValue]]
):
    @classmethod
    def get_required_measures(cls, tensor_id: TensorId, kwargs: ScaleMeanVarianceKwargs):
        axes = tuple(kwargs.axes) if isinstance(kwargs.axes, str) else kwargs.axes
        return _MeanStdAndRef(
            mean=RequiredMeasure(Mean(axes), tensor_id, mode=kwargs.mode),
            std=RequiredMeasure(Std(axes), tensor_id, mode=kwargs.mode),
            ref_mean=RequiredMeasure(Mean(axes), kwargs.reference_tensor, mode=kwargs.mode),
            ref_std=RequiredMeasure(Std(axes), kwargs.reference_tensor, mode=kwargs.mode),
        )

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        c = self.computed
        eps = self.kwargs.eps
        return (tensor - c.mean) / (c.std + eps) * (c.ref_std + eps) + c.ref_mean

    def get_spec(self):
        if isinstance(self.kwargs, v0_4.ScaleMeanVarianceKwargs):
            raise NotImplementedError

        return v0_5.ScaleMeanVariance(kwargs=self.kwargs)


@dataclass
class _MinMaxPerc(_NamedMeasures[RCV]):
    lower: RCV
    upper: RCV


class ScaleRangeImpl(ProcessingImplBase[ScaleRangeKwargs, _MinMaxPerc[RequiredMeasure], _MinMaxPerc[MeasureValue]]):
    # def get_required_measures(self):
    @classmethod
    def get_required_measures(cls, tensor_id: TensorId, kwargs: ScaleRangeKwargs) -> _MinMaxPerc[RequiredMeasure]:
        ref_name = kwargs.reference_tensor or tensor_id
        axes = None if kwargs.axes is None else tuple(kwargs.axes)
        return _MinMaxPerc(
            lower=RequiredMeasure(Percentile(kwargs.min_percentile, axes=axes), ref_name, kwargs.mode),
            upper=RequiredMeasure(Percentile(kwargs.max_percentile, axes=axes), ref_name, kwargs.mode),
        )

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        c = self.computed
        return (tensor - c.lower) / (c.upper - c.lower + self.kwargs.eps)

    def get_spec(self):
        if isinstance(self.kwargs, v0_4.ScaleRangeKwargs):
            raise NotImplementedError

        return v0_5.ScaleRange(kwargs=self.kwargs)


@dataclass
class SigmoidImpl(ProcessingImplBaseWoMeasures[ProcessingKwargs]):
    """1 / (1 + e^(-tensor))."""

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return 1.0 / (1.0 + np.exp(-tensor))  # type: ignore


@dataclass
class ZeroMeanUnitVarianceImpl(
    ProcessingImplBase[
        ZeroMeanUnitVarianceKwargs,
        Union[_NoRequiredMeasures, _MeanStd[RequiredMeasure]],
        Union[_NoMeasureValues, _MeanStd[MeasureValue]],
    ]
):
    """normalize to zero mean, unit variance."""

    @classmethod
    def get_required_measures(
        cls, tensor_id: TensorId, kwargs: ZeroMeanUnitVarianceKwargs
    ) -> Union[_NoRequiredMeasures, _MeanStd[RequiredMeasure]]:
        if kwargs.mode == FIXED:
            return _NamedMeasures()
        else:
            axes = None if kwargs.axes is None else tuple(kwargs.axes)
            return _MeanStd(
                mean=RequiredMeasure(Mean(axes=axes), tensor_id, kwargs.mode),
                std=RequiredMeasure(Std(axes=axes), tensor_id, kwargs.mode),
            )

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        if self.kwargs.mode == FIXED:
            assert self.kwargs.mean is not None
            assert self.kwargs.std is not None
            assert not isinstance(self.computed, _MeanStd)
            axes = None if self.kwargs.axes is None else tuple(self.kwargs.axes)
            mean = _get_fixed(self.kwargs.mean, tensor, axes)
            std = _get_fixed(self.kwargs.std, tensor, axes)
        else:
            assert self.kwargs.mode in (PER_SAMPLE, PER_DATASET)
            assert self.kwargs.mean is None
            assert self.kwargs.std is None
            assert isinstance(self.computed, _MeanStd)
            mean = self.computed.mean
            std = self.computed.std

        return (tensor - mean) / (std + self.kwargs.eps)


IMPLEMENTED_PREPROCESSING = {
    v0_5.Binarize.model_fields["id"].default
    # binarize = Binarize
    # clip = Clip
    # scale_linear = ScaleLinear
    # scale_range = ScaleRange
    # sigmoid = Sigmoid
    # zero_mean_unit_variance = ZeroMeanUnitVariance
}

class IMPLEMENTED_POSTPROCESSING:
    binarize = Binarize
    clip = Clip
    scale_linear = ScaleLinear
    scale_mean_variance = ScaleMeanVariance
    scale_range = ScaleRange
    sigmoid = Sigmoid
    zero_mean_unit_variance = ZeroMeanUnitVariance


class IMPLEMENTED_PROCESSING(IMPLEMENTED_PREPROCESSING, IMPLEMENTED_POSTPROCESSING):
    pass
