from __future__ import annotations

import collections.abc
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import partial
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    Literal,
    cast,
)

import numpy as np
import scipy  # pyright: ignore[reportMissingTypeStubs]
import xarray as xr
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Discriminator, Tag, TypeAdapter
from typing_extensions import Self, assert_never

from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import (
    _convert_proc,  # pyright: ignore[reportPrivateUsage]
)

from ._op_base import (
    BlockwiseOperator,
    Operator,
    SamplewiseOperator,
    SimpleOperator,
    get_relevant_preceding_ops,
)
from ._ops_cellpose import CellposeFlowDynamics
from ._ops_stardist import StardistPostprocessing2D, StardistPostprocessing3D
from ._utils.type_hints import cast_and_assert
from .axis import CHANNEL_AXIS_ID, AxisId
from .common import DTypeStr, MemberId
from .digest_spec import get_member_id, import_callable
from .sample import Sample
from .stat_calculators import StatsCalculator
from .stat_measures import (
    DatasetMean,
    DatasetMeasureBase,
    DatasetQuantile,
    DatasetStd,
    MeanMeasure,
    Measure,
    MeasureValue,
    SampleMean,
    SampleQuantile,
    SampleStd,
    Stat,
    StdMeasure,
)
from .tensor import Tensor


def _convert_axis_ids(
    axes: v0_4.AxesInCZYX,
    mode: Literal["per_sample", "per_dataset"],
) -> tuple[AxisId, ...]:
    if not isinstance(axes, str):
        return tuple(axes)

    if mode == "per_sample":
        ret = []
    elif mode == "per_dataset":
        ret = [AxisId(v0_5.BATCH_AXIS_ID)]
    else:
        assert_never(mode)

    ret.extend([AxisId(a) for a in axes])
    return tuple(ret)


@dataclass(frozen=True)
class AddKnownDatasetStats(BlockwiseOperator):
    dataset_stats: Mapping[Measure, MeasureValue]

    @property
    def affects_statistics(self) -> bool:
        return False

    def __post_init__(self):
        # keep only dataset measures
        object.__setattr__(
            self,
            "dataset_stats",
            {k: v for k, v in self.dataset_stats.items() if k.scope == "dataset"},
        )

    def __call__(self, sample: Sample) -> None:  #  | SampleBlock
        sample.stat.update(self.dataset_stats)
        sample.applied_ops += (self,)


class _StatsCalculatorReference(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    measures: tuple[Measure, ...]
    dataset_fits_in_memory: bool


@dataclass(frozen=True)
class UpdateStats(SamplewiseOperator):
    """Calculates sample and/or dataset measures"""

    stats_calculator: StatsCalculator | _StatsCalculatorReference = field(
        repr=False, compare=False
    )
    """Calculator, or a non-executable reference after deserialization."""
    keep_updating_initial_dataset_stats: bool = False
    """indicates if operator calls should keep updating initial dataset statistics or not;
    if the `stats_calculator` was not provided with any initial dataset statistics,
    these are always updated with every new sample.
    """
    _keep_updating_dataset_stats: bool = field(init=False, compare=False, repr=False)
    calculator_reference: _StatsCalculatorReference = field(init=False, repr=False)

    @property
    def affects_statistics(self) -> bool:
        return False

    def __post_init__(self):
        calculator = self.stats_calculator
        reference = (
            _StatsCalculatorReference(
                measures=calculator.measures,
                dataset_fits_in_memory=calculator.dataset_fits_in_memory,
            )
            if isinstance(calculator, StatsCalculator)
            else calculator
        )
        object.__setattr__(self, "calculator_reference", reference)
        object.__setattr__(
            self,
            "_keep_updating_dataset_stats",
            self.keep_updating_initial_dataset_stats
            or (
                isinstance(calculator, StatsCalculator)
                and not calculator.has_dataset_measures
            ),
        )

    def __call__(self, sample: Sample) -> None:
        if not isinstance(self.stats_calculator, StatsCalculator):
            raise RuntimeError(  # noqa: TRY004
                "A deserialized UpdateStats is a reference and cannot run"
            )
        if self._keep_updating_dataset_stats:
            sample.stat.update(self.stats_calculator.update_and_get_all(sample))
        else:
            sample.stat.update(self.stats_calculator.skip_update_and_get_all(sample))
        sample.applied_ops += (self,)


@dataclass(frozen=True)
class Binarize(SimpleOperator):
    """'output = tensor > threshold'."""

    threshold: float | Sequence[float]
    axis: AxisId | None = None

    def __post_init__(self):
        if not isinstance(self.threshold, (int, float)):
            object.__setattr__(self, "threshold", tuple(self.threshold))

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        return x > self.threshold

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    def get_descr(self):
        if self.axis is None:
            assert isinstance(self.threshold, (int, float))
            kwargs = v0_5.BinarizeKwargs(threshold=float(self.threshold))
        else:
            assert not isinstance(self.threshold, (int, float))
            kwargs = v0_5.BinarizeAlongAxisKwargs(
                threshold=list(self.threshold), axis=self.axis
            )

        return v0_5.BinarizeDescr(kwargs=kwargs)

    @classmethod
    def from_proc_descr(
        cls, descr: v0_4.BinarizeDescr | v0_5.BinarizeDescr, member_id: MemberId
    ) -> Self:
        if isinstance(descr.kwargs, (v0_4.BinarizeKwargs, v0_5.BinarizeKwargs)):
            return cls(
                input=member_id, output=member_id, threshold=descr.kwargs.threshold
            )
        elif isinstance(descr.kwargs, v0_5.BinarizeAlongAxisKwargs):
            return cls(
                input=member_id,
                output=member_id,
                threshold=descr.kwargs.threshold,
                axis=descr.kwargs.axis,
            )
        else:
            assert_never(descr.kwargs)


@dataclass(frozen=True)
class Clip(SimpleOperator):
    min: float | SampleQuantile | DatasetQuantile | None = None
    """minimum value for clipping"""
    max: float | SampleQuantile | DatasetQuantile | None = None
    """maximum value for clipping"""

    def __post_init__(self):
        if self.min is None and self.max is None:
            raise ValueError("missing min or max value")

        if (
            isinstance(self.min, float)
            and isinstance(self.max, float)
            and self.min >= self.max
        ):
            raise ValueError(f"expected min < max, but {self.min} >= {self.max}")

        if isinstance(self.min, (SampleQuantile, DatasetQuantile)) and isinstance(
            self.max, (SampleQuantile, DatasetQuantile)
        ):
            if self.min.axes != self.max.axes:
                raise NotImplementedError(
                    f"expected min and max quantiles with same axes, but got {self.min.axes} and {self.max.axes}"
                )
            if self.min.q >= self.max.q:
                raise ValueError(
                    f"expected min quantile < max quantile, but {self.min.q} >= {self.max.q}"
                )

    def get_required_measures(self) -> Collection[Measure]:
        return {
            arg
            for arg in (self.min, self.max)
            if isinstance(arg, (SampleQuantile, DatasetQuantile))
        }

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        if isinstance(self.min, (SampleQuantile, DatasetQuantile)):
            min_value = stat[self.min]
            if isinstance(min_value, (int, float)):
                # use clip for scalar value
                min_clip_arg = min_value
            else:
                # clip does not support non-scalar values
                x = Tensor.from_xarray(
                    cast_and_assert(
                        xr.DataArray,
                        x.data.where(x.data >= min_value.data, min_value.data),
                    )
                )
                min_clip_arg = None
        else:
            min_clip_arg = self.min

        if isinstance(self.max, (SampleQuantile, DatasetQuantile)):
            max_value = stat[self.max]
            if isinstance(max_value, (int, float)):
                # use clip for scalar value
                max_clip_arg = max_value
            else:
                # clip does not support non-scalar values
                x = Tensor.from_xarray(
                    cast_and_assert(
                        xr.DataArray,
                        x.data.where(x.data <= max_value.data, max_value.data),
                    )
                )
                max_clip_arg = None
        else:
            max_clip_arg = self.max

        if min_clip_arg is not None or max_clip_arg is not None:
            x = x.clip(min_clip_arg, max_clip_arg)

        return x

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    def get_descr(self):
        kwargs: dict[str, Any] = {}
        quantile: SampleQuantile | DatasetQuantile | None = None
        if isinstance(self.min, (SampleQuantile, DatasetQuantile)):
            kwargs["min_percentile"] = self.min.q * 100
            quantile = self.min
        else:
            kwargs["min"] = self.min

        if isinstance(self.max, (SampleQuantile, DatasetQuantile)):
            kwargs["max_percentile"] = self.max.q * 100
            quantile = self.max
        else:
            kwargs["max"] = self.max

        if quantile is not None:
            kwargs["axes"] = quantile.axes

        return v0_5.ClipDescr(kwargs=v0_5.ClipKwargs(**kwargs))

    @classmethod
    def from_proc_descr(
        cls,
        descr: v0_4.ClipDescr | v0_5.ClipDescr,
        member_id: MemberId,
        preceding_ops: Sequence[Operator] = (),
    ) -> Self:
        if isinstance(descr, v0_5.ClipDescr):
            dataset_mode, axes = _get_axes(descr.kwargs)
            if dataset_mode:
                Quantile = DatasetQuantile
            else:
                Quantile = partial(SampleQuantile, method="inverted_cdf")

            relevant_ops = get_relevant_preceding_ops(member_id, preceding_ops)

            if descr.kwargs.min is not None:
                min_arg = descr.kwargs.min
            elif descr.kwargs.min_percentile is not None:
                min_arg = Quantile(
                    q=descr.kwargs.min_percentile / 100,
                    axes=axes,
                    member_id=member_id,
                    preceding_ops=relevant_ops,
                )
            else:
                min_arg = None

            if descr.kwargs.max is not None:
                max_arg = descr.kwargs.max
            elif descr.kwargs.max_percentile is not None:
                max_arg = Quantile(
                    q=descr.kwargs.max_percentile / 100,
                    axes=axes,
                    member_id=member_id,
                    preceding_ops=relevant_ops,
                )
            else:
                max_arg = None

        elif isinstance(descr, v0_4.ClipDescr):
            min_arg = descr.kwargs.min
            max_arg = descr.kwargs.max
        else:
            assert_never(descr)

        return cls(
            input=member_id,
            output=member_id,
            min=min_arg,
            max=max_arg,
        )


@dataclass(frozen=True)
class EnsureDtype(SimpleOperator):
    dtype: DTypeStr

    @classmethod
    def from_proc_descr(cls, descr: v0_5.EnsureDtypeDescr, member_id: MemberId):
        return cls(input=member_id, output=member_id, dtype=descr.kwargs.dtype)

    def get_descr(self):
        return v0_5.EnsureDtypeDescr(kwargs=v0_5.EnsureDtypeKwargs(dtype=self.dtype))

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        return x.astype(self.dtype)


@dataclass(frozen=True)
class ScaleLinear(SimpleOperator):
    gain: float | tuple[float, ...] = 1.0
    """multiplicative factor"""

    offset: float | tuple[float, ...] = 0.0
    """additive term"""
    axis: AxisId | None = None
    """Axis for tuple-valued gain or offset."""

    def __post_init__(self):
        if not isinstance(self.gain, (int, float, tuple)) or not isinstance(
            self.offset, (int, float, tuple)
        ):
            raise TypeError("gain and offset must be scalars or tuples")
        if self.axis is None and (
            isinstance(self.gain, tuple) or isinstance(self.offset, tuple)
        ):
            raise ValueError("tuple-valued gain or offset requires an axis")

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        gain = (
            xr.DataArray(list(self.gain), dims=self.axis)
            if isinstance(self.gain, tuple)
            else self.gain
        )
        offset = (
            xr.DataArray(list(self.offset), dims=self.axis)
            if isinstance(self.offset, tuple)
            else self.offset
        )
        return x * gain + offset

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    def get_descr(self):
        if self.axis is not None:
            kwargs = v0_5.ScaleLinearAlongAxisKwargs(
                axis=self.axis,
                gain=list(self.gain) if isinstance(self.gain, tuple) else self.gain,
                offset=list(self.offset)
                if isinstance(self.offset, tuple)
                else self.offset,
            )
        else:
            assert isinstance(self.gain, (int, float)) and isinstance(
                self.offset, (int, float)
            )
            kwargs = v0_5.ScaleLinearKwargs(gain=self.gain, offset=self.offset)

        return v0_5.ScaleLinearDescr(kwargs=kwargs)

    @classmethod
    def from_proc_descr(
        cls,
        descr: v0_4.ScaleLinearDescr | v0_5.ScaleLinearDescr,
        member_id: MemberId,
    ) -> Self:
        kwargs = descr.kwargs
        if isinstance(kwargs, v0_5.ScaleLinearKwargs):
            axis = None
        elif isinstance(kwargs, v0_5.ScaleLinearAlongAxisKwargs):
            axis = kwargs.axis
        elif isinstance(kwargs, v0_4.ScaleLinearKwargs):
            if kwargs.axes is not None:
                raise NotImplementedError(
                    "model.v0_4.ScaleLinearKwargs with axes not implemented, please consider updating the model to v0_5."
                )
            axis = None
        else:
            assert_never(kwargs)

        if axis is not None:
            gain = (
                kwargs.gain
                if isinstance(kwargs.gain, (float, int))
                else tuple(kwargs.gain)
            )
            offset = (
                kwargs.offset
                if isinstance(kwargs.offset, (float, int))
                else tuple(kwargs.offset)
            )
        else:
            assert isinstance(kwargs.gain, (float, int)) or len(kwargs.gain) == 1, (
                kwargs.gain
            )
            gain = (
                kwargs.gain if isinstance(kwargs.gain, (float, int)) else kwargs.gain[0]
            )
            assert isinstance(kwargs.offset, (float, int)) or len(kwargs.offset) == 1
            offset = (
                kwargs.offset
                if isinstance(kwargs.offset, (float, int))
                else kwargs.offset[0]
            )

        return cls(
            input=member_id, output=member_id, gain=gain, offset=offset, axis=axis
        )


@dataclass(frozen=True)
class ScaleMeanVariance(SimpleOperator):
    axes: Sequence[AxisId] | None = None
    reference_tensor: MemberId | None = None
    eps: float = 1e-6
    mean: MeanMeasure = field(init=False)
    std: StdMeasure = field(init=False)
    ref_mean: MeanMeasure = field(init=False)
    ref_std: StdMeasure = field(init=False)
    _mean: MeanMeasure | None = field(default=None, repr=False, compare=False)
    _std: StdMeasure | None = field(default=None, repr=False, compare=False)
    _ref_mean: MeanMeasure | None = field(default=None, repr=False, compare=False)
    _ref_std: StdMeasure | None = field(default=None, repr=False, compare=False)

    def get_required_measures(self) -> Collection[Measure]:
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

        object.__setattr__(self, "axes", axes)
        object.__setattr__(
            self,
            "mean",
            self._mean
            if self._mean is not None
            else Mean(member_id=self.input, axes=axes),
        )
        object.__setattr__(
            self,
            "std",
            self._std
            if self._std is not None
            else Std(member_id=self.input, axes=axes),
        )
        object.__setattr__(
            self,
            "ref_mean",
            self._ref_mean
            if self._ref_mean is not None
            else Mean(member_id=ref_tensor, axes=axes),
        )
        object.__setattr__(
            self,
            "ref_std",
            self._ref_std
            if self._ref_std is not None
            else Std(member_id=ref_tensor, axes=axes),
        )
        if (
            any(
                m.axes != axes
                for m in (self.mean, self.std, self.ref_mean, self.ref_std)
            )
            or any(m.member_id != self.input for m in (self.mean, self.std))
            or any(m.member_id != ref_tensor for m in (self.ref_mean, self.ref_std))
            or not isinstance(self.mean, Mean)
            or not isinstance(self.std, Std)
            or not isinstance(self.ref_mean, Mean)
            or not isinstance(self.ref_std, Std)
        ):
            raise ValueError(
                "measures must match the operator's members, axes and scope"
            )

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        mean = stat[self.mean]
        std = stat[self.std] + self.eps
        ref_mean = stat[self.ref_mean]
        ref_std = stat[self.ref_std] + self.eps
        return (x - mean) / std * ref_std + ref_mean

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    def get_descr(self):
        return v0_5.ScaleMeanVarianceDescr(
            kwargs=v0_5.ScaleMeanVarianceKwargs(
                reference_tensor=self.reference_tensor or self.input,
                axes=self.axes,
                eps=self.eps,
            )
        )

    @classmethod
    def from_proc_descr(
        cls,
        descr: v0_4.ScaleMeanVarianceDescr | v0_5.ScaleMeanVarianceDescr,
        member_id: MemberId,
        preceding_ops: Sequence[Operator] = (),
    ) -> Self:
        kwargs = descr.kwargs
        _, axes = _get_axes(descr.kwargs)
        ref_tensor = MemberId(str(kwargs.reference_tensor))
        if axes is None or AxisId("batch") not in axes:
            Mean, Std = SampleMean, SampleStd
        else:
            Mean, Std = DatasetMean, DatasetStd
        input_ops = get_relevant_preceding_ops(member_id, preceding_ops)
        ref_ops = get_relevant_preceding_ops(ref_tensor, preceding_ops)

        return cls(
            input=member_id,
            output=member_id,
            reference_tensor=ref_tensor,
            axes=axes,
            eps=kwargs.eps,
            _mean=Mean(member_id=member_id, axes=axes, preceding_ops=input_ops),
            _std=Std(member_id=member_id, axes=axes, preceding_ops=input_ops),
            _ref_mean=Mean(member_id=ref_tensor, axes=axes, preceding_ops=ref_ops),
            _ref_std=Std(member_id=ref_tensor, axes=axes, preceding_ops=ref_ops),
        )


def _get_axes(
    kwargs: v0_4.ZeroMeanUnitVarianceKwargs
    | v0_5.ZeroMeanUnitVarianceKwargs
    | v0_4.ScaleRangeKwargs
    | v0_5.ScaleRangeKwargs
    | v0_4.ScaleMeanVarianceKwargs
    | v0_5.ScaleMeanVarianceKwargs
    | v0_5.ClipKwargs,
) -> tuple[bool, tuple[AxisId, ...] | None]:
    if kwargs.axes is None:
        return True, None
    elif isinstance(kwargs.axes, str):
        axes = _convert_axis_ids(kwargs.axes, kwargs["mode"])
        return AxisId("b") in axes, axes
    elif isinstance(kwargs.axes, collections.abc.Sequence):
        axes = tuple(kwargs.axes)
        return AxisId("batch") in axes, axes
    else:
        assert_never(kwargs.axes)


@dataclass(frozen=True)
class ScaleRange(SimpleOperator):
    _lower: SampleQuantile | DatasetQuantile | None = field(
        default=None, repr=False, compare=False
    )
    _upper: SampleQuantile | DatasetQuantile | None = field(
        default=None, repr=False, compare=False
    )
    lower: SampleQuantile | DatasetQuantile = field(init=False)
    upper: SampleQuantile | DatasetQuantile = field(init=False)

    eps: float = 1e-6

    def __post_init__(
        self,
    ):
        if self._lower is None:
            m = self.input if self._upper is None else self._upper.member_id
            object.__setattr__(self, "lower", DatasetQuantile(q=0.0, member_id=m))
        else:
            object.__setattr__(self, "lower", self._lower)

        if self._upper is None:
            object.__setattr__(
                self,
                "upper",
                DatasetQuantile(q=1.0, member_id=self.lower.member_id),
            )
        else:
            object.__setattr__(self, "upper", self._upper)

        assert self.lower.member_id == self.upper.member_id
        assert self.lower.q < self.upper.q
        assert self.lower.axes == self.upper.axes

    def get_required_measures(self) -> Collection[Measure]:
        return {self.lower, self.upper}

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls,
        descr: v0_4.ScaleRangeDescr | v0_5.ScaleRangeDescr,
        member_id: MemberId,
        preceding_ops: Sequence[Operator] = (),
    ):
        kwargs = descr.kwargs
        ref_tensor = (
            member_id
            if kwargs.reference_tensor is None
            else MemberId(str(kwargs.reference_tensor))
        )
        dataset_mode, axes = _get_axes(descr.kwargs)
        if dataset_mode:
            Quantile = DatasetQuantile
        else:
            Quantile = partial(SampleQuantile, method="linear")

        relevant_ops = get_relevant_preceding_ops(ref_tensor, preceding_ops)
        return cls(
            input=member_id,
            output=member_id,
            _lower=Quantile(
                q=kwargs.min_percentile / 100,
                axes=axes,
                member_id=ref_tensor,
                preceding_ops=relevant_ops,
            ),
            _upper=Quantile(
                q=kwargs.max_percentile / 100,
                axes=axes,
                member_id=ref_tensor,
                preceding_ops=relevant_ops,
            ),
        )

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        lower = stat[self.lower]
        upper = stat[self.upper]
        return (x - lower) / (upper - lower + self.eps)

    def get_descr(self):
        assert self.lower.axes == self.upper.axes
        assert self.lower.member_id == self.upper.member_id

        return v0_5.ScaleRangeDescr(
            kwargs=v0_5.ScaleRangeKwargs(
                axes=None
                if self.lower.axes is None
                else [v0_5.AxisId(a) for a in self.lower.axes],
                min_percentile=self.lower.q * 100,
                max_percentile=self.upper.q * 100,
                eps=self.eps,
                reference_tensor=self.lower.member_id,
            )
        )


@dataclass(frozen=True)
class Sigmoid(SimpleOperator):
    """1 / (1 + e^(-input))."""

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        return Tensor.from_xarray(
            cast_and_assert(
                xr.DataArray, x.data.copy(data=1.0 / (1.0 + np.exp(-x.data.data)))
            )
        )

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls, descr: v0_4.SigmoidDescr | v0_5.SigmoidDescr, member_id: MemberId
    ) -> Self:
        assert isinstance(descr, (v0_4.SigmoidDescr, v0_5.SigmoidDescr))
        return cls(input=member_id, output=member_id)

    def get_descr(self):
        return v0_5.SigmoidDescr()


@dataclass(frozen=True)
class Softmax(SimpleOperator):
    """Softmax activation function."""

    axis: AxisId = CHANNEL_AXIS_ID

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        axis_idx = x.dims.index(self.axis)
        result = scipy.special.softmax(x.data, axis=axis_idx)
        result_xr = cast_and_assert(xr.DataArray, x.data.copy(data=result))
        return Tensor.from_xarray(result_xr)

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(cls, descr: v0_5.SoftmaxDescr, member_id: MemberId) -> Self:
        assert isinstance(descr, v0_5.SoftmaxDescr)
        return cls(input=member_id, output=member_id, axis=descr.kwargs.axis)

    def get_descr(self):
        return v0_5.SoftmaxDescr(kwargs=v0_5.SoftmaxKwargs(axis=v0_5.AxisId(self.axis)))


@dataclass(frozen=True)
class ZeroMeanUnitVariance(SimpleOperator):
    """normalize to zero mean, unit variance."""

    mean: MeanMeasure
    std: StdMeasure

    eps: float = 1e-6

    def __post_init__(self):
        assert self.mean.axes == self.std.axes

    def get_required_measures(self) -> Collection[Measure]:
        return {self.mean, self.std}

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls,
        descr: v0_4.ZeroMeanUnitVarianceDescr | v0_5.ZeroMeanUnitVarianceDescr,
        member_id: MemberId,
        preceding_ops: Sequence[Operator] = (),
    ):
        dataset_mode, axes = _get_axes(descr.kwargs)
        if dataset_mode:
            Mean = DatasetMean
            Std = DatasetStd
        else:
            Mean = SampleMean
            Std = SampleStd

        relevant_ops = get_relevant_preceding_ops(member_id, preceding_ops)
        return cls(
            input=member_id,
            output=member_id,
            mean=Mean(axes=axes, member_id=member_id, preceding_ops=relevant_ops),
            std=Std(axes=axes, member_id=member_id, preceding_ops=relevant_ops),
        )

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        mean = stat[self.mean]
        std = stat[self.std]
        return (x - mean) / (std + self.eps)

    def get_descr(self):
        return v0_5.ZeroMeanUnitVarianceDescr(
            kwargs=v0_5.ZeroMeanUnitVarianceKwargs(axes=self.mean.axes, eps=self.eps)
        )


@dataclass(frozen=True)
class FixedZeroMeanUnitVariance(SimpleOperator):
    """normalize to zero mean, unit variance with precomputed values."""

    mean: float | tuple[float, ...]
    std: float | tuple[float, ...]

    eps: float = 1e-6
    axis: AxisId | None = None
    """Axis for tuple-valued mean and std."""

    def __post_init__(self):
        if not isinstance(self.mean, (int, float, tuple)) or not isinstance(
            self.std, (int, float, tuple)
        ):
            raise TypeError("mean and std must be scalars or tuples")
        if self.axis is None:
            if isinstance(self.mean, tuple) or isinstance(self.std, tuple):
                raise ValueError("mean and std must be scalars without an axis")
        elif not isinstance(self.mean, tuple) or not isinstance(self.std, tuple):
            raise ValueError("mean and std must be tuples when an axis is specified")

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls,
        descr: v0_5.FixedZeroMeanUnitVarianceDescr,
        member_id: MemberId,
    ) -> Self:
        if isinstance(descr.kwargs, v0_5.FixedZeroMeanUnitVarianceKwargs):
            mean, std, axis = descr.kwargs.mean, descr.kwargs.std, None
        elif isinstance(descr.kwargs, v0_5.FixedZeroMeanUnitVarianceAlongAxisKwargs):
            mean, std, axis = (
                tuple(descr.kwargs.mean),
                tuple(descr.kwargs.std),
                AxisId(descr.kwargs.axis),
            )
        else:
            assert_never(descr.kwargs)

        return cls(
            input=member_id,
            output=member_id,
            mean=mean,
            std=std,
            axis=axis,
        )

    def get_descr(self):
        if self.axis is None:
            assert isinstance(self.mean, (int, float))
            assert isinstance(self.std, (int, float))
            kwargs = v0_5.FixedZeroMeanUnitVarianceKwargs(mean=self.mean, std=self.std)
        else:
            assert isinstance(self.mean, tuple) and isinstance(self.std, tuple)
            kwargs = v0_5.FixedZeroMeanUnitVarianceAlongAxisKwargs(
                axis=self.axis,
                mean=list(self.mean),
                std=list(self.std),
            )

        return v0_5.FixedZeroMeanUnitVarianceDescr(kwargs=kwargs)

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        if self.axis is None:
            assert isinstance(self.mean, (int, float)) and isinstance(
                self.std, (int, float)
            )
            mean = self.mean
            std = self.std
        else:
            assert isinstance(self.mean, tuple) and isinstance(self.std, tuple)
            mean = xr.DataArray(list(self.mean), dims=self.axis)
            std = xr.DataArray(list(self.std), dims=self.axis)

        return cast_and_assert(Tensor, (x - mean) / (std + self.eps))


@dataclass(frozen=True)
class CustomProcessing(SimpleOperator):
    """Execute a user-supplied custom processing callable.

    Two styles are supported — callable class and factory function::

        # Callable class style
        class my_factory:
            def __init__(self, threshold=0.5):
                self.threshold = threshold
            def __call__(self, *arrays):
                return (arrays[0] > self.threshold).astype(np.uint8)

        # Factory function style
        def my_factory(threshold=0.5):
            def run(*arrays):
                return (arrays[0] > threshold).astype(np.uint8)
            return run

    Runtime protocol: ``custom_callable = my_factory(**kwargs)`` once at construction;
    ``result = custom_callable(tensor)`` once per sample.

    Note: The custom callable may not change the shape of the input tensor.
    """

    custom_factory: Callable[..., Callable[[NDArray[Any]], NDArray[Any]]]

    kwargs: Mapping[str, Any] = field(compare=False)
    """Keyword arguments forwarded to the custom factory."""

    _kwargs_key: str = field(init=False, repr=False)

    # Initialised in __post_init__
    _custom_callable: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        kwargs = dict(self.kwargs)
        object.__setattr__(self, "kwargs", MappingProxyType(kwargs))
        object.__setattr__(
            self,
            "_kwargs_key",
            json.dumps(kwargs, sort_keys=True, separators=(",", ":")),
        )
        object.__setattr__(self, "_custom_callable", self.custom_factory(**kwargs))

    def _apply(self, x: Tensor, stat: Stat) -> Tensor:
        return Tensor.from_xarray(
            cast_and_assert(
                xr.DataArray, x.data.copy(data=self._custom_callable(x.to_numpy()))
            )
        )

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls,
        descr: v0_5.CustomProcessingDescr,
        member_id: MemberId,
    ) -> Self:
        factory = import_callable(descr)

        return cls(
            input=member_id,
            output=member_id,
            custom_factory=factory,
            kwargs=dict(descr.kwargs),
        )


ProcDescr = (
    v0_4.PreprocessingDescr
    | v0_4.PostprocessingDescr
    | v0_5.PreprocessingDescr
    | v0_5.PostprocessingDescr
)
Processing = (
    AddKnownDatasetStats
    | Binarize
    | Clip
    | CellposeFlowDynamics
    | CustomProcessing
    | EnsureDtype
    | FixedZeroMeanUnitVariance
    | ScaleLinear
    | ScaleMeanVariance
    | ScaleRange
    | Sigmoid
    | StardistPostprocessing2D
    | StardistPostprocessing3D
    | Softmax
    | UpdateStats
    | ZeroMeanUnitVariance
)


def get_proc(
    proc_descr: ProcDescr,
    tensor_descr: v0_4.InputTensorDescr
    | v0_4.OutputTensorDescr
    | v0_5.InputTensorDescr
    | v0_5.OutputTensorDescr
    | MemberId,
    *,
    tensor_axes: str | None = None,
    preceding_ops: Sequence[Operator] = (),
) -> Processing:
    """Construct an operator from a processing description.

    `tensor_axes` is only needed if `proc_descr` is a v0.4.ZeroMeanUnitVarianceDescr with mode "fixed".
    """
    if isinstance(
        tensor_descr,
        (
            v0_4.InputTensorDescr,
            v0_4.OutputTensorDescr,
            v0_5.InputTensorDescr,
            v0_5.OutputTensorDescr,
        ),
    ):
        member_id = get_member_id(tensor_descr)
        if isinstance(tensor_descr, (v0_4.InputTensorDescr, v0_4.OutputTensorDescr)):
            tensor_axes = tensor_descr.axes
    else:
        member_id = tensor_descr

    if isinstance(proc_descr, (v0_4.BinarizeDescr, v0_5.BinarizeDescr)):
        return Binarize.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, v0_5.CellposeFlowDynamicsDescr):
        return CellposeFlowDynamics.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, (v0_4.ClipDescr, v0_5.ClipDescr)):
        return Clip.from_proc_descr(proc_descr, member_id, preceding_ops)
    elif isinstance(proc_descr, v0_5.CustomProcessingDescr):
        return CustomProcessing.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, v0_5.EnsureDtypeDescr):
        return EnsureDtype.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, v0_5.FixedZeroMeanUnitVarianceDescr):
        return FixedZeroMeanUnitVariance.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, (v0_4.ScaleLinearDescr, v0_5.ScaleLinearDescr)):
        return ScaleLinear.from_proc_descr(proc_descr, member_id)
    elif isinstance(
        proc_descr, (v0_4.ScaleMeanVarianceDescr, v0_5.ScaleMeanVarianceDescr)
    ):
        return ScaleMeanVariance.from_proc_descr(proc_descr, member_id, preceding_ops)
    elif isinstance(proc_descr, (v0_4.ScaleRangeDescr, v0_5.ScaleRangeDescr)):
        return ScaleRange.from_proc_descr(proc_descr, member_id, preceding_ops)
    elif isinstance(proc_descr, (v0_4.SigmoidDescr, v0_5.SigmoidDescr)):
        return Sigmoid.from_proc_descr(proc_descr, member_id)
    elif (
        isinstance(proc_descr, v0_4.ZeroMeanUnitVarianceDescr)
        and proc_descr.kwargs.mode == "fixed"
    ):
        if tensor_axes is None:
            raise TypeError(
                "Expected v0.4 tensor axes for fixed v0.4 processing description"
            )

        v5_proc_descr = _convert_proc(proc_descr, tensor_axes)
        assert isinstance(v5_proc_descr, v0_5.FixedZeroMeanUnitVarianceDescr)
        return FixedZeroMeanUnitVariance.from_proc_descr(v5_proc_descr, member_id)
    elif isinstance(proc_descr, v0_5.SoftmaxDescr):
        return Softmax.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, v0_5.StardistPostprocessingDescr):
        if len(proc_descr.kwargs.grid) == 2:
            return StardistPostprocessing2D.from_proc_descr(proc_descr, member_id)
        elif len(proc_descr.kwargs.grid) == 3:
            return StardistPostprocessing3D.from_proc_descr(proc_descr, member_id)
        else:
            assert_never(proc_descr.kwargs.grid)
    elif isinstance(
        proc_descr,
        (v0_4.ZeroMeanUnitVarianceDescr, v0_5.ZeroMeanUnitVarianceDescr),
    ):
        return ZeroMeanUnitVariance.from_proc_descr(
            proc_descr, member_id, preceding_ops
        )
    else:
        assert_never(proc_descr)


def _processing_tag(value: object) -> str | None:
    if isinstance(value, Mapping):
        tag = cast(Mapping[str, object], value).get("type")
        return tag if isinstance(tag, str) else None
    return type(value).__name__


SerializableProcessing = Annotated[
    Annotated[Binarize, Tag("Binarize")]
    | Annotated[Clip, Tag("Clip")]
    | Annotated[CellposeFlowDynamics, Tag("CellposeFlowDynamics")]
    | Annotated[EnsureDtype, Tag("EnsureDtype")]
    | Annotated[FixedZeroMeanUnitVariance, Tag("FixedZeroMeanUnitVariance")]
    | Annotated[ScaleLinear, Tag("ScaleLinear")]
    | Annotated[ScaleMeanVariance, Tag("ScaleMeanVariance")]
    | Annotated[ScaleRange, Tag("ScaleRange")]
    | Annotated[Sigmoid, Tag("Sigmoid")]
    | Annotated[StardistPostprocessing2D, Tag("StardistPostprocessing2D")]
    | Annotated[StardistPostprocessing3D, Tag("StardistPostprocessing3D")]
    | Annotated[Softmax, Tag("Softmax")]
    | Annotated[ZeroMeanUnitVariance, Tag("ZeroMeanUnitVariance")],
    Discriminator(_processing_tag),
]
_PROCESSING_ADAPTER: TypeAdapter[SerializableProcessing] = TypeAdapter(
    SerializableProcessing
)
_SERIALIZABLE_PROCESSING_TYPES = (
    Binarize,
    Clip,
    CellposeFlowDynamics,
    EnsureDtype,
    FixedZeroMeanUnitVariance,
    ScaleLinear,
    ScaleMeanVariance,
    ScaleRange,
    Sigmoid,
    StardistPostprocessing2D,
    StardistPostprocessing3D,
    Softmax,
    ZeroMeanUnitVariance,
)


class _SerializedUpdateStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["UpdateStats"]
    stats_calculator: _StatsCalculatorReference
    keep_updating_initial_dataset_stats: bool


def serialize_processing(proc: Operator) -> dict[str, Any]:
    """Serialize an operator, omitting mutable statistics calculator state."""
    if type(proc) is AddKnownDatasetStats:
        from .io import serialize_stat

        assert isinstance(proc, AddKnownDatasetStats)
        return {
            "type": "AddKnownDatasetStats",
            "dataset_stats": serialize_stat(proc.dataset_stats),
        }
    if type(proc) is UpdateStats:
        assert isinstance(proc, UpdateStats)
        return _SerializedUpdateStats(
            type="UpdateStats",
            stats_calculator=proc.calculator_reference,
            keep_updating_initial_dataset_stats=proc.keep_updating_initial_dataset_stats,
        ).model_dump(mode="json")
    if (
        not isinstance(proc, _SERIALIZABLE_PROCESSING_TYPES)
        or type(proc) not in _SERIALIZABLE_PROCESSING_TYPES
    ):
        raise TypeError(f"{type(proc).__name__} cannot be serialized")

    return {
        "type": type(proc).__name__,
        **_PROCESSING_ADAPTER.dump_python(proc, mode="json", warnings="error"),
    }


def deserialize_processing(
    data: Mapping[str, Any],
) -> SerializableProcessing | AddKnownDatasetStats | UpdateStats:
    """Restore a tagged operator, rejecting incomplete or unexpected fields."""
    if data.get("type") == "AddKnownDatasetStats":
        from .io import JsonValue, load_stat

        if set(data) != {"type", "dataset_stats"} or not isinstance(
            data["dataset_stats"], list
        ):
            raise ValueError("Invalid serialized AddKnownDatasetStats fields")
        stats = load_stat(cast(list[JsonValue], data["dataset_stats"]))
        if any(not isinstance(measure, DatasetMeasureBase) for measure in stats):
            raise ValueError("AddKnownDatasetStats requires dataset measures")
        return AddKnownDatasetStats(stats)
    if data.get("type") == "UpdateStats":
        payload = _SerializedUpdateStats.model_validate(data)
        return UpdateStats(
            stats_calculator=payload.stats_calculator,
            keep_updating_initial_dataset_stats=payload.keep_updating_initial_dataset_stats,
        )
    proc = _PROCESSING_ADAPTER.validate_python(data)
    expected = {"type", *(f.name for f in fields(proc) if f.init)}
    if set(data) != expected:
        raise ValueError(
            f"Invalid serialized {type(proc).__name__} fields: missing "
            + f"{expected - set(data)}, unexpected {set(data) - expected}"
        )
    return proc
