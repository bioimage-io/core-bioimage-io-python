"""Here pre- and postprocessing operations are implemented according to their definitions in bioimageio.spec:
see https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/preprocessing_spec_latest.md
and https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/postprocessing_spec_latest.md
"""
import collections
from dataclasses import dataclass, field, fields
from typing import Mapping, Optional, Sequence, Set, Tuple, Type, Union

import numpy
import numpy as np
import xarray as xr

from bioimageio.core.statistical_measures import Mean, Measure, Percentile, Std
from bioimageio.spec.model.raw_nodes import PostprocessingName, PreprocessingName
from ._utils import ComputedMeasures, DatasetMode, FIXED, Mode, PER_DATASET, PER_SAMPLE, RequiredMeasures, SampleMode

try:
    from typing import Literal, get_args, TypedDict
except ImportError:
    from typing_extensions import Literal, get_args, TypedDict  # type: ignore


def _get_fixed(
    fixed: Union[float, Sequence[float]], tensor: xr.DataArray, axes: Optional[Sequence[str]]
) -> Union[float, xr.DataArray]:
    if axes is None:
        if isinstance(fixed, float):
            return fixed
        elif isinstance(fixed, collections.Sequence):
            raise TypeError(f"Sequence of fixed values requires axes. Either use scalar fixed value or specify axes.")
        else:
            raise TypeError(type(fixed))

    fixed_shape = tuple(s for d, s in tensor.sizes.items() if d not in axes)
    fixed_dims = tuple(d for d in tensor.dims if d not in axes)
    fixed = np.array(fixed).reshape(fixed_shape)  # type: ignore[assignment]
    return xr.DataArray(fixed, dims=fixed_dims)


TensorName = str

MISSING = "MISSING"


@dataclass
class Processing:
    """base class for all Pre- and Postprocessing transformations."""

    tensor_name: str
    # todo: in python>=3.10 we should use dataclasses.KW_ONLY instead of MISSING (see child classes) to make inheritance work properly
    computed_measures: ComputedMeasures = field(default_factory=dict)
    mode: Mode = FIXED

    def get_required_measures(self) -> RequiredMeasures:
        return {}

    def set_computed_measures(self, computed: ComputedMeasures):
        # check if computed contains all required measures
        for mode, req_per_mode in self.get_required_measures().items():
            for tn, req_per_tn in req_per_mode.items():
                comp_measures = computed.get(mode, {}).get(tn, {})
                for req_measure in req_per_tn:
                    if req_measure not in comp_measures:
                        raise ValueError(f"Missing required {req_measure} for {tn} {mode}.")

        self.computed_measures = computed

    def get_computed_measure(self, tensor_name: TensorName, measure: Measure, *, mode: Optional[Mode] = None):
        """helper to unpack self.computed_measures"""
        mo = mode or self.mode
        assert mo != "fixed"
        ret = self.computed_measures.get(mo, {}).get(tensor_name, {}).get(measure)
        if ret is None:
            raise RuntimeError(f"Missing computed {measure} for {tensor_name} {mode}.")

        return ret

    def __call__(self, tensor: xr.DataArray) -> xr.DataArray:
        return self.apply(tensor)

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        """apply processing"""
        raise NotImplementedError

    def __post_init__(self):
        # validate common kwargs by their annotations
        for f in fields(self):
            # check MISSING
            if getattr(self, f.name) is MISSING:
                raise TypeError(f"missing required argument {f.name}")

            if f.name == "mode":
                # mode is always annotated as literals (or literals of literals)
                valid_modes = get_args(f.type)
                for inner in get_args(f.type):
                    valid_modes += get_args(inner)

                if self.mode not in valid_modes:
                    raise NotImplementedError(f"Unsupported mode {self.mode} for {self.__class__.__name__}")


#
# Pre- and Postprocessing implementations
#


@dataclass
class AssertDtype(Processing):
    """Helper Processing to assert dtype."""

    dtype: Union[str, Sequence[str]] = MISSING
    assert_with: Tuple[Type[numpy.dtype], ...] = field(init=False)

    def __post_init__(self):
        if isinstance(self.dtype, str):
            dtype: Sequence[str] = [self.dtype]
        else:
            dtype = self.dtype

        self.assert_with = tuple(type(numpy.dtype(dt)) for dt in dtype)

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        assert isinstance(tensor.dtype, self.assert_with)
        return tensor


@dataclass
class Binarize(Processing):
    """'output = tensor > threshold'."""

    # make dataclass inheritance work for py<3.10 by using an explicit MISSING value.
    threshold: float = MISSING  # type: ignore[assignment]

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return tensor > self.threshold


@dataclass
class Clip(Processing):
    """Limit tensor values to [min, max]."""

    min: float = MISSING  # type: ignore[assignment]
    max: float = MISSING  # type: ignore[assignment]

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

    gain: Union[float, Sequence[float]] = MISSING  # type: ignore[assignment]
    offset: Union[float, Sequence[float]] = MISSING  # type: ignore[assignment]
    axes: Optional[Sequence[str]] = None

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        if self.axes is not None:
            scale_axes = tuple(ax for ax in tensor.dims if (ax not in self.axes and ax != "b"))
            gain: Union[float, xr.DataArray] = xr.DataArray(np.atleast_1d(self.gain), dims=scale_axes)
            offset: Union[float, xr.DataArray] = xr.DataArray(np.atleast_1d(self.offset), dims=scale_axes)
        else:
            if not isinstance(self.gain, (float, int)):
                raise TypeError(f"axes: None; expected gain to be a scalar, but got type {type(self.gain)}")

            if not isinstance(self.offset, (float, int)):
                raise TypeError(f"axes: None; expected offset to be a scalar, but got type {type(self.offset)}")

            gain = float(self.gain)
            offset = float(self.offset)

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
                self.tensor_name: {Mean(axes=axes), Std(axes=axes)},
                self.reference_tensor: {Mean(axes=axes), Std(axes=axes)},
            }
        }

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        axes = None if self.axes is None else tuple(self.axes)
        assert self.mode in (PER_SAMPLE, PER_DATASET)
        mean = self.get_computed_measure(self.tensor_name, Mean(axes), mode=self.mode)
        std = self.get_computed_measure(self.tensor_name, Std(axes), mode=self.mode)
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
        # todo: not sure why leaving out 'Set[Measure]' here (==Set[Percentile]) gives mypy error,
        #  as Percentile inherits from Measure...?  mypy bug?
        measures: Set[Measure] = {
            Percentile(self.min_percentile, axes=axes),
            Percentile(self.max_percentile, axes=axes),
        }
        return {self.mode: {self.reference_tensor or self.tensor_name: measures}}

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        ref_name = self.reference_tensor or self.tensor_name
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
            return {self.mode: {self.tensor_name: {Mean(axes=axes), Std(axes=axes)}}}

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        axes = None if self.axes is None else tuple(self.axes)
        if self.mode == FIXED:
            assert self.mean is not None and self.std is not None
            mean = _get_fixed(self.mean, tensor, axes)
            std = _get_fixed(self.std, tensor, axes)
        elif self.mode in (PER_SAMPLE, PER_DATASET):
            assert self.mean is None and self.std is None
            mean = self.get_computed_measure(self.tensor_name, Mean(axes), mode=self.mode)
            std = self.get_computed_measure(self.tensor_name, Std(axes), mode=self.mode)
        else:
            raise ValueError(self.mode)

        return (tensor - mean) / (std + self.eps)


class _KnownProcessing(TypedDict):
    pre: Mapping[PreprocessingName, Type[Processing]]
    post: Mapping[PostprocessingName, Type[Processing]]


KNOWN_PROCESSING: _KnownProcessing = dict(
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
