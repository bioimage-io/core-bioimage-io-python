import collections.abc
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field, fields
from types import MappingProxyType
from typing import (
    ClassVar,
    FrozenSet,
    Generic,
    Hashable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy
import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from typing_extensions import LiteralString, assert_never

from bioimageio.core.common import MeasureValue, ProcessingDescrBase, ProcessingKwargs, RequiredMeasure, Sample
from bioimageio.core.stat_measures import Mean, Percentile, Std
from bioimageio.spec._internal.base_nodes import NodeWithExplicitlySetFields
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import NonBatchAxisId, TensorId

AssertProcessingId = Literal["assert_dtype"]


class AssertProcessingBase(NodeWithExplicitlySetFields):
    id: AssertProcessingId
    fields_to_set_explicitly: ClassVar[FrozenSet[LiteralString]] = frozenset({"id"})


class AssertDtypeKwargs(v0_5.ProcessingKwargs):
    dtype: Union[str, Sequence[str]]


class AssertDtype(AssertProcessingBase):
    id: Literal["assert_dtype"] = "assert_dtype"
    kwargs: AssertDtypeKwargs


M = TypeVar("M", RequiredMeasure, MeasureValue)


@dataclass
class NamedMeasures(Generic[M]):
    """Named Measures that specifies all required/computed measures of a Processing instance"""

    def get_set(self) -> Set[M]:
        return {getattr(self, f.name) for f in fields(self)}


# The two generics are conceptually a higher kinded generic
R = TypeVar("R", bound=NamedMeasures[RequiredMeasure])
C = TypeVar("C", bound=NamedMeasures[MeasureValue])


PKwargs = TypeVar("PKwargs", bound=ProcessingKwargs)
ProcInput = TypeVar("ProcInput", xr.DataArray, Sample)


@dataclass(frozen=True)
class ProcessingImplBase(Generic[PKwargs, R, C], ABC):
    """Base class for all Pre- and Postprocessing implementations."""

    tensor_id: TensorId
    """id of tensor to operate on"""
    kwargs: PKwargs
    computed_measures: InitVar[Mapping[RequiredMeasure, MeasureValue]] = field(
        default=MappingProxyType[RequiredMeasure, MeasureValue]({})
    )
    assert type(R) is type(C), "R and C are conceptually a higher kindes generic, their class has to be identical"
    required: R = field(init=False)
    computed: C = field(init=False)

    def __post_init__(self, computed_measures: Mapping[RequiredMeasure, MeasureValue]) -> None:
        object.__setattr__(self, "required", self.get_required_measures(self.tensor_id, self.kwargs))
        selected = {}
        for f in fields(self.required):
            req = getattr(self.required, f.name)
            if req in computed_measures:
                selected[f.name] = computed_measures[req]
            else:
                raise ValueError(f"Missing computed measure: {req} (as '{f.name}').")

        object.__setattr__(self, "computed", self.required.__class__(**selected))

    @abstractmethod
    @classmethod
    def get_required_measures(cls, tensor_id: TensorId, kwargs: PKwargs) -> NamedMeasures[RequiredMeasure]:
        ...

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

    @abstractmethod
    def get_descr(self) -> Union[ProcessingDescrBase, AssertProcessingBase]:
        ...


@dataclass(frozen=True)
class ProcessingImplBaseWoMeasures(
    ProcessingImplBase[PKwargs, NamedMeasures[RequiredMeasure], NamedMeasures[MeasureValue]]
):
    @classmethod
    def get_required_measures(cls, tensor_id: TensorId, kwargs: PKwargs) -> NamedMeasures[RequiredMeasure]:
        return NamedMeasures()


@dataclass(frozen=True)
class AssertDtypeImpl(ProcessingImplBaseWoMeasures[AssertDtypeKwargs]):
    kwargs_class = AssertDtypeKwargs
    _assert_with: Tuple[Type[DTypeLike], ...] = field(init=False)

    def __post_init__(self, computed_measures: Mapping[RequiredMeasure, MeasureValue]) -> None:
        super().__post_init__(computed_measures)
        if isinstance(self.kwargs.dtype, str):
            dtype = [self.kwargs.dtype]
        else:
            dtype = self.kwargs.dtype

        object.__setattr__(self, "assert_with", tuple(type(numpy.dtype(dt)) for dt in dtype))

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        assert isinstance(tensor.dtype, self._assert_with)
        return tensor

    def get_descr(self):
        return AssertDtype(kwargs=self.kwargs)


@dataclass(frozen=True)
class BinarizeImpl(ProcessingImplBaseWoMeasures[Union[v0_4.BinarizeKwargs, v0_5.BinarizeKwargs]]):
    """'output = tensor > threshold'."""

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor > self.kwargs.threshold

    def get_descr(self):
        return v0_5.BinarizeDescr(kwargs=self.kwargs)


@dataclass(frozen=True)
class ClipImpl(ProcessingImplBaseWoMeasures[Union[v0_4.ClipKwargs, v0_5.ClipKwargs]]):
    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor.clip(min=self.kwargs.min, max=self.kwargs.max)

    def get_descr(self):
        return v0_5.ClipDescr(kwargs=self.kwargs)


@dataclass(frozen=True)
class EnsureDtypeImpl(ProcessingImplBaseWoMeasures[v0_5.EnsureDtypeKwargs]):
    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor.astype(self.kwargs.dtype)

    def get_descr(self):
        return v0_5.EnsureDtypeDescr(kwargs=self.kwargs)


class ScaleLinearImpl04(ProcessingImplBaseWoMeasures[Union[v0_4.ScaleLinearKwargs, v0_5.ScaleLinearKwargs]]):
    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        axis = (
            self.kwargs.axis
            if isinstance(self.kwargs, v0_5.ScaleLinearKwargs)
            else _get_complement_axis(tensor, self.kwargs.axes)
        )
        if axis:
            gain = xr.DataArray(np.atleast_1d(self.kwargs.gain), dims=axis)
            offset = xr.DataArray(np.atleast_1d(self.kwargs.offset), dims=axis)
        else:
            assert isinstance(self.kwargs.gain, (float, int)) or len(self.kwargs.gain) == 1
            gain = self.kwargs.gain if isinstance(self.kwargs.gain, (float, int)) else self.kwargs.gain[0]
            assert isinstance(self.kwargs.offset, (float, int)) or len(self.kwargs.offset) == 1
            offset = self.kwargs.offset if isinstance(self.kwargs.offset, (float, int)) else self.kwargs.offset[0]

        return tensor * gain + offset


@dataclass(frozen=True)
class ScaleLinearImpl(ProcessingImplBaseWoMeasures[Union[v0_4.ScaleLinearKwargs, v0_5.ScaleLinearKwargs]]):
    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        axis = (
            self.kwargs.axis
            if isinstance(self.kwargs, v0_5.ScaleLinearKwargs)
            else _get_complement_axis(tensor, self.kwargs.axes)
        )
        if axis:
            gain = xr.DataArray(np.atleast_1d(self.kwargs.gain), dims=axis)
            offset = xr.DataArray(np.atleast_1d(self.kwargs.offset), dims=axis)
        else:
            assert isinstance(self.kwargs.gain, (float, int)) or len(self.kwargs.gain) == 1
            gain = self.kwargs.gain if isinstance(self.kwargs.gain, (float, int)) else self.kwargs.gain[0]
            assert isinstance(self.kwargs.offset, (float, int)) or len(self.kwargs.offset) == 1
            offset = self.kwargs.offset if isinstance(self.kwargs.offset, (float, int)) else self.kwargs.offset[0]

        return tensor * gain + offset

    def get_descr(self):
        if isinstance(self.kwargs, v0_4.ScaleLinearKwargs):
            raise NotImplementedError

        return v0_5.ScaleLinearDescr(kwargs=self.kwargs)


@dataclass
class NamedMeasuresScaleMeanVariance(NamedMeasures[M]):
    mean: M
    std: M
    ref_mean: M
    ref_std: M


@dataclass(frozen=True)
class ScaleMeanVarianceImpl(
    ProcessingImplBase[
        Union[v0_4.ScaleMeanVarianceKwargs, v0_5.ScaleMeanVarianceKwargs],
        NamedMeasuresScaleMeanVariance[RequiredMeasure],
        NamedMeasuresScaleMeanVariance[MeasureValue],
    ]
):
    @classmethod
    def get_required_measures(
        cls, tensor_id: TensorId, kwargs: Union[v0_4.ScaleMeanVarianceKwargs, v0_5.ScaleMeanVarianceKwargs]
    ):
        if kwargs.axes is None:
            axes = None
        elif isinstance(kwargs.axes, str):
            axes = tuple(NonBatchAxisId(a) for a in kwargs.axes)
        elif isinstance(kwargs.axes, collections.abc.Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
            axes = tuple(kwargs.axes)
        else:
            assert_never(kwargs.axes)

        return NamedMeasuresScaleMeanVariance(
            mean=RequiredMeasure(Mean(axes), tensor_id, mode=kwargs.mode),
            std=RequiredMeasure(Std(axes), tensor_id, mode=kwargs.mode),
            ref_mean=RequiredMeasure(Mean(axes), cast(TensorId, kwargs.reference_tensor), mode=kwargs.mode),
            ref_std=RequiredMeasure(Std(axes), cast(TensorId, kwargs.reference_tensor), mode=kwargs.mode),
        )

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        c = self.computed
        eps = self.kwargs.eps
        return (tensor - c.mean) / (c.std + eps) * (c.ref_std + eps) + c.ref_mean

    def get_descr(self):
        if isinstance(self.kwargs, v0_4.ScaleMeanVarianceKwargs):
            raise NotImplementedError

        return v0_5.ScaleMeanVarianceDescr(kwargs=self.kwargs)


@dataclass
class NamedMeasuresScaleRange(NamedMeasures[M]):
    lower: M
    upper: M


@dataclass(frozen=True)
class ScaleRangeImpl(
    ProcessingImplBase[
        Union[v0_4.ScaleRangeKwargs, v0_5.ScaleRangeKwargs],
        NamedMeasuresScaleRange[RequiredMeasure],
        NamedMeasuresScaleRange[MeasureValue],
    ]
):
    @classmethod
    def get_required_measures(cls, tensor_id: TensorId, kwargs: Union[v0_4.ScaleRangeKwargs, v0_5.ScaleRangeKwargs]):
        ref_name = kwargs.reference_tensor or tensor_id
        axes = None if kwargs.axes is None else tuple(NonBatchAxisId(a) for a in kwargs.axes)
        return NamedMeasuresScaleRange(
            lower=RequiredMeasure(Percentile(kwargs.min_percentile, axes=axes), cast(TensorId, ref_name), kwargs.mode),
            upper=RequiredMeasure(Percentile(kwargs.max_percentile, axes=axes), cast(TensorId, ref_name), kwargs.mode),
        )

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        c = self.computed
        return (tensor - c.lower) / (c.upper - c.lower + self.kwargs.eps)

    def get_descr(self):
        if isinstance(self.kwargs, v0_4.ScaleRangeKwargs):
            raise NotImplementedError

        return v0_5.ScaleRangeDescr(kwargs=self.kwargs)


@dataclass(frozen=True)
class SigmoidImpl(ProcessingImplBaseWoMeasures[v0_5.ProcessingKwargs]):
    """1 / (1 + e^(-tensor))."""

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return 1.0 / (1.0 + np.exp(-tensor))  # type: ignore

    def get_descr(self):
        return v0_5.SigmoidDescr()


@dataclass
class NamedMeasuresZeroMeanUnitVariance(NamedMeasures[M]):
    mean: M
    std: M


@dataclass(frozen=True)
class ZeroMeanUnitVarianceImpl(
    ProcessingImplBase[
        Union[v0_4.ZeroMeanUnitVarianceKwargs, v0_5.ZeroMeanUnitVarianceKwargs],
        NamedMeasuresZeroMeanUnitVariance[RequiredMeasure],
        NamedMeasuresZeroMeanUnitVariance[MeasureValue],
    ]
):
    """normalize to zero mean, unit variance."""

    @classmethod
    def get_required_measures(
        cls, tensor_id: TensorId, kwargs: Union[v0_4.ZeroMeanUnitVarianceKwargs, v0_5.ZeroMeanUnitVarianceKwargs]
    ):
        axes = None if kwargs.axes is None else tuple(NonBatchAxisId(a) for a in kwargs.axes)
        assert kwargs.mode != "fixed"  # should use FixedZeroMeanUnitVarianceImpl
        return NamedMeasuresZeroMeanUnitVariance(
            mean=RequiredMeasure(Mean(axes=axes), tensor_id, kwargs.mode),
            std=RequiredMeasure(Std(axes=axes), tensor_id, kwargs.mode),
        )

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        mean = self.computed.mean
        std = self.computed.std
        return (tensor - mean) / (std + self.kwargs.eps)

    def get_descr(self):
        if isinstance(self.kwargs, v0_4.ZeroMeanUnitVarianceKwargs):
            raise NotImplementedError

        return v0_5.ZeroMeanUnitVarianceDescr(kwargs=self.kwargs)


@dataclass(frozen=True)
class FixedZeroMeanUnitVarianceImpl(
    ProcessingImplBaseWoMeasures[Union[v0_4.ZeroMeanUnitVarianceKwargs, v0_5.FixedZeroMeanUnitVarianceKwargs]]
):
    """normalize to zero mean, unit variance with precomputed values."""

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        if isinstance(self.kwargs, v0_5.FixedZeroMeanUnitVarianceKwargs):
            axis = self.kwargs.axis
        elif isinstance(self.kwargs.mean, float) and isinstance(self.kwargs.std, float):
            axis = None
        else:
            axis = _get_complement_axis(tensor, self.kwargs.axes)

        mean = xr.DataArray(self.kwargs.mean, dims=axis)
        std = xr.DataArray(self.kwargs.std, dims=axis)
        return (tensor - mean) / std

    def get_descr(self):
        if isinstance(self.kwargs, v0_4.ZeroMeanUnitVarianceKwargs):
            raise NotImplementedError

        return v0_5.FixedZeroMeanUnitVarianceDescr(kwargs=self.kwargs)


ProcDescr = Union[
    AssertDtype, v0_4.PreprocessingDescr, v0_4.PostprocessingDescr, v0_5.PreprocessingDescr, v0_5.PostprocessingDescr
]

# get_impl_class which also returns the kwargs class
# def get_impl_class(proc_spec: ProcDescr):
#     if isinstance(proc_spec, AssertDtype):
#         return AssertDtypeImpl, AssertDtypeKwargs
#     elif isinstance(proc_spec, v0_4.BinarizeDescr):
#         return BinarizeImpl, v0_4.BinarizeKwargs
#     elif isinstance(proc_spec, v0_5.BinarizeDescr):
#         return BinarizeImpl, v0_5.BinarizeKwargs
#     elif isinstance(proc_spec, (v0_4.ClipDescr, v0_5.ClipDescr)):
#         return ClipImpl, v0_5.ClipKwargs
#     elif isinstance(proc_spec, v0_5.EnsureDtypeDescr):
#         return EnsureDtypeImpl, v0_5.EnsureDtypeKwargs
#     elif isinstance(proc_spec, v0_5.FixedZeroMeanUnitVarianceDescr):
#         return FixedZeroMeanUnitVarianceImpl, v0_5.FixedZeroMeanUnitVarianceKwargs
#     elif isinstance(proc_spec, (v0_4.ScaleLinearDescr, v0_5.ScaleLinearDescr)):
#         return ScaleLinearImpl, v0_5.ScaleLinearKwargs
#     elif isinstance(proc_spec, (v0_4.ScaleMeanVarianceDescr, v0_5.ScaleMeanVarianceDescr)):
#         return ScaleMeanVarianceImpl, v0_5.ScaleMeanVarianceKwargs
#     elif isinstance(proc_spec, (v0_4.ScaleRangeDescr, v0_5.ScaleRangeDescr)):
#         return ScaleRangeImpl, v0_5.ScaleRangeKwargs
#     elif isinstance(proc_spec, (v0_4.SigmoidDescr, v0_5.SigmoidDescr)):
#         return SigmoidImpl, v0_5.ProcessingKwargs
#     elif isinstance(proc_spec, v0_4.ZeroMeanUnitVarianceDescr) and proc_spec.kwargs.mode == "fixed":
#         return FixedZeroMeanUnitVarianceImpl, v0_5.FixedZeroMeanUnitVarianceKwargs
#     elif isinstance(
#         proc_spec,  # pyright: ignore[reportUnnecessaryIsInstance
#         (v0_4.ZeroMeanUnitVarianceDescr, v0_5.ZeroMeanUnitVarianceDescr),
#     ):
#         return ZeroMeanUnitVarianceImpl, v0_5.ZeroMeanUnitVarianceKwargs
#     else:
#         assert_never(proc_spec)

ProcessingImpl = Union[
    AssertDtypeImpl,
    BinarizeImpl,
    ClipImpl,
    EnsureDtypeImpl,
    FixedZeroMeanUnitVarianceImpl,
    FixedZeroMeanUnitVarianceImpl,
    ScaleLinearImpl,
    ScaleMeanVarianceImpl,
    ScaleRangeImpl,
    SigmoidImpl,
    ZeroMeanUnitVarianceImpl,
]


def get_impl_class(proc_spec: ProcDescr) -> Type[ProcessingImpl]:
    if isinstance(proc_spec, AssertDtype):
        return AssertDtypeImpl
    elif isinstance(proc_spec, (v0_4.BinarizeDescr, v0_5.BinarizeDescr)):
        return BinarizeImpl
    elif isinstance(proc_spec, (v0_4.ClipDescr, v0_5.ClipDescr)):
        return ClipImpl
    elif isinstance(proc_spec, v0_5.EnsureDtypeDescr):
        return EnsureDtypeImpl
    elif isinstance(proc_spec, v0_5.FixedZeroMeanUnitVarianceDescr):
        return FixedZeroMeanUnitVarianceImpl
    elif isinstance(proc_spec, (v0_4.ScaleLinearDescr, v0_5.ScaleLinearDescr)):
        return ScaleLinearImpl
    elif isinstance(proc_spec, (v0_4.ScaleMeanVarianceDescr, v0_5.ScaleMeanVarianceDescr)):
        return ScaleMeanVarianceImpl
    elif isinstance(proc_spec, (v0_4.ScaleRangeDescr, v0_5.ScaleRangeDescr)):
        return ScaleRangeImpl
    elif isinstance(proc_spec, (v0_4.SigmoidDescr, v0_5.SigmoidDescr)):
        return SigmoidImpl
    elif isinstance(proc_spec, v0_4.ZeroMeanUnitVarianceDescr) and proc_spec.kwargs.mode == "fixed":
        return FixedZeroMeanUnitVarianceImpl
    elif isinstance(
        proc_spec,  # pyright: ignore[reportUnnecessaryIsInstance]
        (v0_4.ZeroMeanUnitVarianceDescr, v0_5.ZeroMeanUnitVarianceDescr),
    ):
        return ZeroMeanUnitVarianceImpl
    else:
        assert_never(proc_spec)


def _get_complement_axis(tensor: xr.DataArray, axes: Optional[Sequence[Hashable]]) -> Optional[Hashable]:
    if axes is None:
        return None

    v04_AXIS_TYPE_MAP = {
        "b": "batch",
        "t": "time",
        "i": "index",
        "c": "channel",
        "x": "space",
        "y": "space",
        "z": "space",
    }
    converted_axes = [v04_AXIS_TYPE_MAP.get(a, a) for a in map(str, axes)] + ["batch"]
    complement_axes = [a for a in tensor.dims if str(a) not in converted_axes]
    if len(complement_axes) != 1:
        raise ValueError(
            f"Expected a single complement axis, but axes '{converted_axes}' (orignally '{axes}') "
            f"for tensor dims '{tensor.dims}' leave '{complement_axes}'."
        )

    return complement_axes[0]
