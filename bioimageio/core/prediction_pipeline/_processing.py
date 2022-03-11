from dataclasses import dataclass, field, fields
from typing import Dict, Mapping, Optional, Sequence, Set, Type, Union

import numpy as np
import xarray as xr

from bioimageio.core.statistical_measures import Mean, Measure, Percentile, Std
from bioimageio.core.utils import DatasetMode, FIXED, Mode, PER_DATASET, PER_SAMPLE, SampleMode
from bioimageio.spec.model.raw_nodes import PostprocessingName, PreprocessingName

try:
    from typing import Literal, get_args
except ImportError:
    from typing_extensions import Literal, get_args  # type: ignore


def _get_fixed(
    fixed: Union[float, Sequence[float]], tensor: xr.DataArray, axes: Optional[Sequence[str]]
) -> Union[float, xr.DataArray]:
    if axes is None:
        return fixed

    fixed_shape = tuple(s for d, s in tensor.sizes.items() if d not in axes)
    fixed_dims = tuple(d for d in tensor.dims if d not in axes)
    fixed = np.array(fixed).reshape(fixed_shape)
    return xr.DataArray(fixed, dims=fixed_dims)


TensorName = str

MISSING = "MISSING"


@dataclass
class Processing:
    """base class for all Pre- and Postprocessing transformations"""

    tensor_name: str
    # todo: in python>=3.10 we should use dataclasses.KW_ONLY instead of MISSING (see child classes) to make inheritance work properly
    computed_statistics: Dict[Mode, Dict[TensorName, Dict[Measure, xr.DataArray]]] = field(default_factory=dict)
    mode: Mode = FIXED

    def get_required_statistics(self) -> Dict[Mode, Dict[TensorName, Set[Measure]]]:
        """
        Returns: required measures per tensor for the given scope.
        """
        return {}

    def set_computed_statistics(self, computed: Dict[TensorName, Dict[Measure, xr.DataArray]], *, mode: Mode):
        for tensor_name, req_measures in self.get_required_statistics().get(mode, {}).items():
            comp_measures = computed.get(tensor_name, {})
            for req_measure in req_measures:
                if req_measure not in comp_measures:
                    raise ValueError(f"Missing required {req_measure} for {tensor_name} {mode}.")

        self.computed_statistics[mode] = computed

    def get_computed_statistics(self, tensor_name: TensorName, measure: Measure, *, mode: Optional[Mode] = None):
        """helper to unpack self.computed_statistics"""
        ret = self.computed_statistics.get(mode or self.mode, {}).get(tensor_name, {}).get(measure)
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
                # mode is always annotated as literals
                valid_modes = sum([get_args(inner) for inner in get_args(f.type)], start=tuple())
                if self.mode not in valid_modes:
                    raise NotImplementedError(f"Unsupported mode {self.mode} for {self.__class__.__name__}")


#
# helpers
#
def ensure_dtype(tensor: xr.DataArray, *, dtype) -> xr.DataArray:
    """
    Convert array to a given datatype
    """
    return tensor.astype(dtype)


#
# Pre- and Postprocessing implementations
#


@dataclass
class Binarize(Processing):
    threshold: float = MISSING  # make dataclass inheritance work for py<3.10 by using an explicit MISSING value.

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return ensure_dtype(tensor > self.threshold, dtype="float32")


@dataclass
class Clip(Processing):
    min: float = MISSING
    max: float = MISSING

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return ensure_dtype(tensor.clip(min=self.min, max=self.max), dtype="float32")


@dataclass
class EnsureDtype(Processing):
    dtype: str = MISSING

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return ensure_dtype(tensor, dtype=self.dtype)


@dataclass
class ScaleLinear(Processing):
    """scale the tensor with a fixed multiplicative and additive factor"""

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

        return ensure_dtype(tensor * gain + offset, dtype="float32")

    def __post_init__(self):
        super().__post_init__()
        if self.axes is None:
            assert isinstance(self.gain, (int, float))
            assert isinstance(self.offset, (int, float))


@dataclass
class ScaleMeanVariance(Processing):
    ...


@dataclass
class ScaleRange(Processing):
    mode: Literal[SampleMode, DatasetMode] = PER_SAMPLE
    axes: Optional[Sequence[str]] = None
    min_percentile: float = 0.0
    max_percentile: float = 100.0
    eps: float = 1e-6
    reference_tensor: Optional[TensorName] = None

    def get_required_statistics(self) -> Dict[Mode, Dict[TensorName, Set[Measure]]]:
        axes = None if self.axes is None else tuple(self.axes)
        measures = {Percentile(self.min_percentile, axes=axes), Percentile(self.max_percentile, axes=axes)}
        return {self.mode: {self.reference_tensor or self.tensor_name: measures}}

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        ref_name = self.reference_tensor or self.tensor_name
        axes = None if self.axes is None else tuple(self.axes)
        v_lower = self.get_computed_statistics(ref_name, Percentile(self.min_percentile, axes=axes))
        v_upper = self.get_computed_statistics(ref_name, Percentile(self.max_percentile, axes=axes))

        return ensure_dtype((tensor - v_lower) / (v_upper - v_lower + self.eps), dtype="float32")

    def __post_init__(self):
        super().__post_init__()
        self.axes = None if self.axes is None else tuple(self.axes)  # make sure axes is Tuple[str] or None


@dataclass
class Sigmoid(Processing):
    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return 1.0 / (1.0 + xr.ufuncs.exp(-tensor))


@dataclass
class ZeroMeanUnitVariance(Processing):
    mode: Mode = PER_SAMPLE
    mean: Optional[Union[float, Sequence[float]]] = None
    std: Optional[Union[float, Sequence[float]]] = None
    axes: Optional[Sequence[str]] = None
    eps: float = 1.0e-6

    def get_required_statistics(self) -> Dict[Mode, Dict[TensorName, Set[Measure]]]:
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
            mean = self.get_computed_statistics(self.tensor_name, Mean(axes), mode=self.mode)
            std = self.get_computed_statistics(self.tensor_name, Std(axes), mode=self.mode)
        else:
            raise ValueError(self.mode)

        tensor = (tensor - mean) / (std + self.eps)
        return ensure_dtype(tensor, dtype="float32")


KNOWN_PREPROCESSING: Mapping[PreprocessingName, Type[Processing]] = {
    "binarize": Binarize,
    "clip": Clip,
    "scale_linear": ScaleLinear,
    "scale_range": ScaleRange,
    "sigmoid": Sigmoid,
    "zero_mean_unit_variance": ZeroMeanUnitVariance,
}

KNOWN_POSTPROCESSING: Mapping[PostprocessingName, Type[Processing]] = {
    "binarize": Binarize,
    "clip": Clip,
    "scale_linear": ScaleLinear,
    "scale_mean_variance": ScaleMeanVariance,
    "scale_range": ScaleRange,
    "sigmoid": Sigmoid,
    "zero_mean_unit_variance": ZeroMeanUnitVariance,
}
