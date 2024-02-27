import collections.abc
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import (
    Collection,
    Hashable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from typing_extensions import Self, assert_never

from bioimageio.core._op_base import Operator
from bioimageio.core.common import (
    AxisId,
    Sample,
    Stat,
    Tensor,
    TensorId,
)
from bioimageio.core.stat_calculators import StatsCalculator
from bioimageio.core.stat_measures import (
    DatasetMean,
    DatasetMeasure,
    DatasetPercentile,
    DatasetStd,
    MeanMeasure,
    Measure,
    MeasureValue,
    SampleMean,
    SamplePercentile,
    SampleStd,
    StdMeasure,
)
from bioimageio.spec.model import v0_4, v0_5


def convert_axis_ids(
    axes: Union[Sequence[AxisId], v0_4.AxesInCZYX], mode: Literal["per_sample", "per_dataset"]
) -> Tuple[AxisId, ...]:
    if not isinstance(axes, str):
        return tuple(axes)

    axis_map = dict(b=AxisId("batch"), c=AxisId("channel"), i=AxisId("index"))
    if mode == "per_sample":
        ret = []
    elif mode == "per_dataset":
        ret = [AxisId("batch")]
    else:
        assert_never(mode)

    ret.extend([axis_map.get(a, AxisId(a)) for a in axes])
    return tuple(ret)


@dataclass
class _SimpleOperator(Operator, ABC):
    input: TensorId
    output: TensorId

    @property
    def required_measures(self) -> Collection[Measure]:
        return set()

    # @property
    # def required_tensors(self) -> Set[TensorId]:
    #     return {self.input}

    # @property
    # def produced_tensors(self) -> Set[TensorId]:
    #     return {self.output}

    def __call__(self, sample: Sample) -> None:
        sample.data[self.output] = self._apply(sample.data[self.input], sample.stat)

    @abstractmethod
    def _apply(self, input: Tensor, stat: Stat) -> Tensor: ...


@dataclass
class AddKnownDatasetStats(Operator):
    dataset_stats: Mapping[DatasetMeasure, MeasureValue]

    @property
    def required_measures(self) -> Set[Measure]:
        return set()

    def __call__(self, sample: Sample) -> None:
        sample.stat.update(self.dataset_stats.items())


# @dataclass
# class UpdateStats(Operator):
#     """Calculates sample and/or dataset measures"""

#     measures: Union[Sequence[Measure], Set[Measure], Mapping[Measure, MeasureValue]]
#     """sample and dataset `measuers` to be calculated by this operator. Initial/fixed
#     dataset measure values may be given, see `keep_updating_dataset_stats` for details.
#     """
#     keep_updating_dataset_stats: Optional[bool] = None
#     """indicates if operator calls should keep updating dataset statistics or not

#     default (None): if `measures` is a `Mapping` (i.e. initial measure values are
#     given) no further updates to dataset statistics is conducted, otherwise (w.o.
#     initial measure values) dataset statistics are updated by each processed sample.
#     """
#     _keep_updating_dataset_stats: bool = field(init=False)
#     _stats_calculator: StatsCalculator = field(init=False)

#     @property
#     def required_measures(self) -> Set[Measure]:
#         return set()

#     def __post_init__(self):
#         self._stats_calculator = StatsCalculator(self.measures)
#         if self.keep_updating_dataset_stats is None:
#             self._keep_updating_dataset_stats = not isinstance(self.measures, collections.abc.Mapping)
#         else:
#             self._keep_updating_dataset_stats = self.keep_updating_dataset_stats

#     def __call__(self, sample: Sample) -> None:
#         if self._keep_updating_dataset_stats:
#             sample.stat.update(self._stats_calculator.update_and_get_all(sample))
#         else:
#             sample.stat.update(self._stats_calculator.skip_update_and_get_all(sample))


@dataclass
class UpdateStats(Operator):
    """Calculates sample and/or dataset measures"""

    stats_calculator: StatsCalculator
    """`StatsCalculator` to be used by this operator."""
    keep_updating_initial_dataset_stats: bool = False
    """indicates if operator calls should keep updating initial dataset statistics or not;
    if the `stats_calculator` was not provided with any initial dataset statistics,
    these are always updated with every new sample.
    """
    _keep_updating_dataset_stats: bool = field(init=False)

    @property
    def required_measures(self) -> Set[Measure]:
        return set()

    def __post_init__(self):
        self._keep_updating_initial_dataset_stats = (
            self.keep_updating_initial_dataset_stats or not self.stats_calculator.has_dataset_measures
        )

    def __call__(self, sample: Sample) -> None:
        if self._keep_updating_dataset_stats:
            sample.stat.update(self.stats_calculator.update_and_get_all(sample))
        else:
            sample.stat.update(self.stats_calculator.skip_update_and_get_all(sample))


@dataclass
class Binarize(_SimpleOperator):
    """'output = tensor > threshold'."""

    threshold: float

    def _apply(self, input: Tensor, stat: Stat) -> xr.DataArray:
        return input > self.threshold

    # @classmethod
    # def from_descr(cls, descr: Union[v0_4.BinarizeDescr, v0_5.BinarizeDescr]):
    #     return cls(threshold=descr.kwargs.threshold)

    # def get_descr(self):
    #     return v0_5.BinarizeDescr(kwargs=v0_5.BinarizeKwargs(threshold=self.threshold))
    @classmethod
    def from_proc_descr(cls, descr: Union[v0_4.BinarizeDescr, v0_5.BinarizeDescr], tensor_id: TensorId) -> Self:
        return cls(input=tensor_id, output=tensor_id, threshold=descr.kwargs.threshold)


@dataclass
class Clip(_SimpleOperator):
    min: Optional[float] = None
    """minimum value for clipping"""
    max: Optional[float] = None
    """maximum value for clipping"""

    def __post_init__(self):
        assert self.min is not None or self.max is not None, "missing min or max value"
        assert (
            self.min is None or self.max is None or self.min < self.max
        ), f"expected min < max, but {self.min} !< {self.max}"

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        return input.clip(self.min, self.max)

    @classmethod
    def from_proc_descr(cls, descr: Union[v0_4.ClipDescr, v0_5.ClipDescr], tensor_id: TensorId) -> Self:
        return cls(input=tensor_id, output=tensor_id, min=descr.kwargs.min, max=descr.kwargs.max)


@dataclass
class EnsureDtype(_SimpleOperator):
    dtype: DTypeLike

    @classmethod
    def from_proc_descr(cls, descr: v0_5.EnsureDtypeDescr, tensor_id: TensorId):
        return cls(input=tensor_id, output=tensor_id, dtype=descr.kwargs.dtype)

    def get_descr(self):
        return v0_5.EnsureDtypeDescr(kwargs=v0_5.EnsureDtypeKwargs(dtype=str(self.dtype)))

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        return input.astype(self.dtype)


@dataclass
class ScaleLinear(_SimpleOperator):
    gain: Union[float, xr.DataArray] = 1.0
    """multiplicative factor"""

    offset: Union[float, xr.DataArray] = 0.0
    """additive term"""

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        return input * self.gain + self.offset

    # @classmethod
    # def from_descr(cls, descr: ScaleLinearDescr) -> Self:
    #     ...

    @classmethod
    def from_proc_descr(cls, descr: Union[v0_4.ScaleLinearDescr, v0_5.ScaleLinearDescr], tensor_id: TensorId) -> Self:
        kwargs = descr.kwargs
        if isinstance(kwargs, v0_5.ScaleLinearKwargs):
            axis = kwargs.axis
        elif kwargs.axes is not None:
            raise NotImplementedError("ScaleLinear operator from v0_4.ScaleLinearDescr with axes")
        else:
            axis = None

        if axis:
            gain = xr.DataArray(np.atleast_1d(kwargs.gain), dims=axis)
            offset = xr.DataArray(np.atleast_1d(kwargs.offset), dims=axis)
        else:
            assert isinstance(kwargs.gain, (float, int)) or len(kwargs.gain) == 1
            gain = kwargs.gain if isinstance(kwargs.gain, (float, int)) else kwargs.gain[0]
            assert isinstance(kwargs.offset, (float, int)) or len(kwargs.offset) == 1
            offset = kwargs.offset if isinstance(kwargs.offset, (float, int)) else kwargs.offset[0]

        return cls(input=tensor_id, output=tensor_id, gain=gain, offset=offset)


@dataclass
class ScaleMeanVariance(_SimpleOperator):
    axes: Optional[Sequence[AxisId]] = None
    reference_tensor: Optional[TensorId] = None
    eps: float = 1e-6
    mean: Union[SampleMean, DatasetMean] = field(init=False)
    std: Union[SampleStd, DatasetStd] = field(init=False)
    ref_mean: Union[SampleMean, DatasetMean] = field(init=False)
    ref_std: Union[SampleStd, DatasetStd] = field(init=False)

    @property
    def required_measures(self):
        return {self.mean, self.std, self.ref_mean, self.ref_std}

    def __post_init__(self):
        axes = None if self.axes is None else tuple(self.axes)
        ref_tensor = self.reference_tensor or self.input
        if axes is None or AxisId("batch") not in axes:
            Mean = SampleMean
            Std = SampleStd
        else:
            Mean = DatasetMean
            Std = DatasetStd

        self.mean = Mean(tensor_id=self.input, axes=axes)
        self.std = Std(tensor_id=self.input, axes=axes)
        self.ref_mean = Mean(tensor_id=ref_tensor, axes=axes)
        self.ref_std = Std(tensor_id=ref_tensor, axes=axes)

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        mean = stat[self.mean]
        std = stat[self.std] + self.eps
        ref_mean = stat[self.ref_mean]
        ref_std = stat[self.ref_std] + self.eps
        return (input - mean) / std * ref_std + ref_mean

    @classmethod
    def from_proc_descr(
        cls, descr: Union[v0_4.ScaleMeanVarianceDescr, v0_5.ScaleMeanVarianceDescr], tensor_id: TensorId
    ) -> Self:
        kwargs = descr.kwargs
        axes = _get_axes(descr.kwargs)

        return cls(
            input=tensor_id,
            output=tensor_id,
            reference_tensor=cast(TensorId, kwargs.reference_tensor),
            axes=axes,
            eps=kwargs.eps,
        )


def _get_axes(
    kwargs: Union[
        v0_4.ZeroMeanUnitVarianceKwargs,
        v0_5.ZeroMeanUnitVarianceKwargs,
        v0_4.ScaleRangeKwargs,
        v0_5.ScaleRangeKwargs,
        v0_4.ScaleMeanVarianceKwargs,
        v0_5.ScaleMeanVarianceKwargs,
    ]
) -> Union[Tuple[AxisId, ...], None]:
    if kwargs.axes is None:
        axes = None
    elif isinstance(kwargs.axes, str):
        axes = convert_axis_ids(kwargs.axes, kwargs["mode"])
    elif isinstance(kwargs.axes, collections.abc.Sequence):
        axes = tuple(kwargs.axes)
    else:
        assert_never(kwargs.axes)

    return axes


@dataclass
class ScaleRange(_SimpleOperator):
    lower_percentile: InitVar[Optional[Union[SamplePercentile, DatasetPercentile]]] = None
    upper_percentile: InitVar[Optional[Union[SamplePercentile, DatasetPercentile]]] = None
    lower: Union[SamplePercentile, DatasetPercentile] = field(init=False)
    upper: Union[SamplePercentile, DatasetPercentile] = field(init=False)

    eps: float = 1e-6

    def __post_init__(
        self,
        lower_percentile: Optional[Union[SamplePercentile, DatasetPercentile]],
        upper_percentile: Optional[Union[SamplePercentile, DatasetPercentile]],
    ):
        if lower_percentile is None:
            tid = self.input if upper_percentile is None else upper_percentile.tensor_id
            self.lower = DatasetPercentile(n=0, tensor_id=tid)
        else:
            self.lower = lower_percentile

        if upper_percentile is None:
            self.upper = DatasetPercentile(n=100, tensor_id=self.lower.tensor_id)
        else:
            self.upper = upper_percentile

        assert self.lower.tensor_id == self.upper.tensor_id
        assert self.lower.n < self.upper.n
        assert self.lower.axes == self.upper.axes

    @property
    def required_measures(self):
        return {self.lower, self.upper}

    @classmethod
    def from_proc_descr(cls, descr: Union[v0_4.ScaleRangeDescr, v0_5.ScaleRangeDescr], tensor_id: TensorId):
        kwargs = descr.kwargs
        ref_tensor = cast(TensorId, kwargs.reference_tensor) or tensor_id
        axes = _get_axes(descr.kwargs)
        if axes is None or AxisId("batch") in axes:
            Percentile = DatasetPercentile
        else:
            Percentile = SamplePercentile

        return cls(
            input=tensor_id,
            output=tensor_id,
            lower_percentile=Percentile(n=kwargs.min_percentile, axes=axes, tensor_id=ref_tensor),
            upper_percentile=Percentile(n=kwargs.max_percentile, axes=axes, tensor_id=ref_tensor),
        )

    def _apply(self, input: xr.DataArray, stat: Stat) -> xr.DataArray:
        lower = stat[self.lower]
        upper = stat[self.upper]
        return (input - lower) / (upper - lower + self.eps)

    def get_descr(self):
        assert self.lower.axes == self.upper.axes
        assert self.lower.tensor_id == self.upper.tensor_id

        return v0_5.ScaleRangeDescr(
            kwargs=v0_5.ScaleRangeKwargs(
                axes=self.lower.axes,
                min_percentile=self.lower.n,
                max_percentile=self.upper.n,
                eps=self.eps,
                reference_tensor=self.lower.tensor_id,
            )
        )


@dataclass
class Sigmoid(_SimpleOperator):
    """1 / (1 + e^(-input))."""

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        return 1.0 / (1.0 + np.exp(-input))  # type: ignore

    @property
    def required_measures(self) -> Collection[Measure]:
        return {}

    @classmethod
    def from_proc_descr(cls, descr: Union[v0_4.SigmoidDescr, v0_5.SigmoidDescr], tensor_id: TensorId) -> Self:
        assert isinstance(descr, (v0_4.SigmoidDescr, v0_5.SigmoidDescr))
        return cls(input=tensor_id, output=tensor_id)

    def get_descr(self):
        return v0_5.SigmoidDescr()


@dataclass
class ZeroMeanUnitVariance(_SimpleOperator):
    """normalize to zero mean, unit variance."""

    mean: MeanMeasure
    std: StdMeasure

    eps: float = 1e-6

    def __post_init__(self):
        assert self.mean.axes == self.std.axes

    @property
    def required_measures(self) -> Set[Union[MeanMeasure, StdMeasure]]:
        return {self.mean, self.std}

    @classmethod
    def from_proc_descr(
        cls, descr: Union[v0_4.ZeroMeanUnitVarianceDescr, v0_5.ZeroMeanUnitVarianceDescr], tensor_id: TensorId
    ):
        axes = _get_axes(descr.kwargs)

        if axes is None or AxisId("batch") in axes:
            Mean = DatasetMean
            Std = DatasetStd
        else:
            Mean = SampleMean
            Std = SampleStd

        return cls(
            input=tensor_id,
            output=tensor_id,
            mean=Mean(axes=axes, tensor_id=tensor_id),
            std=Std(axes=axes, tensor_id=tensor_id),
        )

    def _apply(self, input: xr.DataArray, stat: Stat) -> xr.DataArray:
        mean = stat[self.mean]
        std = stat[self.std]
        return (input - mean) / (std + self.eps)

    def get_descr(self):
        return v0_5.ZeroMeanUnitVarianceDescr(kwargs=v0_5.ZeroMeanUnitVarianceKwargs(axes=self.mean.axes, eps=self.eps))


@dataclass
class FixedZeroMeanUnitVariance(_SimpleOperator):
    """normalize to zero mean, unit variance with precomputed values."""

    mean: Union[float, xr.DataArray]
    std: Union[float, xr.DataArray]

    eps: float = 1e-6

    def __post_init__(self):
        assert (
            isinstance(self.mean, (int, float)) or isinstance(self.std, (int, float)) or self.mean.dims == self.std.dims
        )

    @classmethod
    def from_proc_descr(
        cls,
        descr: v0_5.FixedZeroMeanUnitVarianceDescr,
        tensor_id: TensorId,
    ) -> Self:
        return cls(
            input=tensor_id,
            output=tensor_id,
            mean=xr.DataArray(descr.kwargs.mean, dims=(descr.kwargs.axis,)),
            std=xr.DataArray(descr.kwargs.std, dims=(descr.kwargs.axis,)),
        )

    def get_descr(self):
        if isinstance(self.mean, (int, float)):
            assert isinstance(self.std, (int, float))
            axis = None
            mean = self.mean
            std = self.std
        else:
            assert isinstance(self.std, xr.DataArray)
            assert len(self.mean.dims) == 1
            axis = AxisId(str(self.mean.dims[0]))
            mean = tuple(self.mean)
            std = tuple(self.std)

        return v0_5.FixedZeroMeanUnitVarianceDescr(
            kwargs=v0_5.FixedZeroMeanUnitVarianceKwargs(axis=axis, mean=mean, std=std)
        )

    def _apply(self, input: xr.DataArray, stat: Stat) -> xr.DataArray:
        return (input - self.mean) / (self.std + self.eps)


ProcDescr = Union[v0_4.PreprocessingDescr, v0_4.PostprocessingDescr, v0_5.PreprocessingDescr, v0_5.PostprocessingDescr]

Processing = Union[
    AddKnownDatasetStats,
    Binarize,
    Clip,
    EnsureDtype,
    FixedZeroMeanUnitVariance,
    ScaleLinear,
    ScaleMeanVariance,
    ScaleRange,
    Sigmoid,
    UpdateStats,
    ZeroMeanUnitVariance,
]


def get_proc_class(proc_spec: ProcDescr):
    if isinstance(proc_spec, (v0_4.BinarizeDescr, v0_5.BinarizeDescr)):
        return Binarize
    elif isinstance(proc_spec, (v0_4.ClipDescr, v0_5.ClipDescr)):
        return Clip
    elif isinstance(proc_spec, v0_5.EnsureDtypeDescr):
        return EnsureDtype
    elif isinstance(proc_spec, v0_5.FixedZeroMeanUnitVarianceDescr):
        return FixedZeroMeanUnitVariance
    elif isinstance(proc_spec, (v0_4.ScaleLinearDescr, v0_5.ScaleLinearDescr)):
        return ScaleLinear
    elif isinstance(proc_spec, (v0_4.ScaleMeanVarianceDescr, v0_5.ScaleMeanVarianceDescr)):
        return ScaleMeanVariance
    elif isinstance(proc_spec, (v0_4.ScaleRangeDescr, v0_5.ScaleRangeDescr)):
        return ScaleRange
    elif isinstance(proc_spec, (v0_4.SigmoidDescr, v0_5.SigmoidDescr)):
        return Sigmoid
    elif isinstance(proc_spec, v0_4.ZeroMeanUnitVarianceDescr) and proc_spec.kwargs.mode == "fixed":
        return FixedZeroMeanUnitVariance
    elif isinstance(
        proc_spec,
        (v0_4.ZeroMeanUnitVarianceDescr, v0_5.ZeroMeanUnitVarianceDescr),
    ):
        return ZeroMeanUnitVariance
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
