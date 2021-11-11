from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Sequence, Set, Union

import numpy as np
import xarray as xr

from bioimageio.core.statistical_measures import Mean, Measure, Percentile, Std

try:
    from typing import Literal, get_args
except ImportError:
    from typing_extensions import Literal, get_args  # type: ignore


@dataclass
class Processing:
    """base class for all Pre- and Postprocessing transformations"""

    tensor_name: str
    computed_dataset_statistics: Dict[str, Dict[Measure, Any]] = field(init=False)
    computed_sample_statistics: Dict[str, Dict[Measure, Any]] = field(init=False)

    def get_required_dataset_statistics(self) -> Dict[str, Set[Measure]]:
        """
        Specifies which dataset measures are required from what tensor.
        Returns: dataset measures required to apply this processing indexed by <tensor_name>.
        """
        return {}

    def get_required_sample_statistics(self) -> Dict[str, Set[Measure]]:
        """
        Specifies which sample measures are required from what tensor.
        Returns: sample measures required to apply this processing indexed by <tensor_name>.
        """
        return {}

    def set_computed_dataset_statistics(self, computed: Dict[str, Dict[Measure, Any]]):
        """helper to set computed statistics and check if they match the requirements"""
        for tensor_name, req_measures in self.get_required_dataset_statistics().items():
            comp_measures = computed.get(tensor_name, {})
            for req_measure in req_measures:
                if req_measure not in comp_measures:
                    raise ValueError(f"Missing required measure {req_measure} for {tensor_name}")
        self.computed_dataset_statistics = computed

    def set_computed_sample_statistics(self, computed: Dict[str, Dict[Measure, Any]]):
        """helper to set computed statistics and check if they match the requirements"""
        for tensor_name, req_measures in self.get_required_sample_statistics().items():
            comp_measures = computed.get(tensor_name, {})
            for req_measure in req_measures:
                if req_measure not in comp_measures:
                    raise ValueError(f"Missing required measure {req_measure} for {tensor_name}")
        self.computed_sample_statistics = computed

    def get_computed_dataset_statistics(self, tensor_name: str, measure: Measure):
        """helper to unpack self.computed_dataset_statistics"""
        ret = self.computed_dataset_statistics.get(tensor_name, {}).get(measure)
        if ret is None:
            raise RuntimeError(f"Missing computed {measure} for {tensor_name} dataset.")

        return ret

    def get_computed_sample_statistics(self, tensor_name: str, measure: Measure):
        """helper to unpack self.computed_sample_statistics"""
        ret = self.computed_sample_statistics.get(tensor_name, {}).get(measure)
        if ret is None:
            raise RuntimeError(f"Missing computed {measure} for {tensor_name} sample.")

        return ret

    def __call__(self, tensor: xr.DataArray) -> xr.DataArray:
        return self.apply(tensor)

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        """apply processing to named tensors"""
        raise NotImplementedError

    def __post_init__(self):
        """validate common kwargs by their annotations"""
        self.computed_dataset_statistics = {}
        self.computed_sample_statistics = {}

        for f in fields(self):
            if f.name == "mode":
                assert hasattr(self, "mode")
                if self.mode not in get_args(f.type):
                    raise NotImplementedError(
                        f"Unsupported mode {self.mode} for {self.__class__.__name__}: {self.mode}"
                    )


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
    threshold: float

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return ensure_dtype(tensor > self.threshold, dtype="float32")


@dataclass
class Clip(Processing):
    min: float
    max: float

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return ensure_dtype(tensor.clip(min=self.min, max=self.max), dtype="float32")


@dataclass
class EnsureDtype(Processing):
    dtype: str

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return ensure_dtype(tensor, dtype=self.dtype)


@dataclass
class ScaleLinear(Processing):
    """scale the tensor with a fixed multiplicative and additive factor"""

    gain: Union[float, Sequence[float]]
    offset: Union[float, Sequence[float]]
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
    mode: Literal["per_sample", "per_dataset"] = "per_sample"
    axes: Optional[Sequence[str]] = None
    min_percentile: float = 0.0
    max_percentile: float = 100.0
    eps: float = 1e-6
    reference_tensor: Optional[str] = None

    def get_required_dataset_statistics(self) -> Dict[str, Set[Measure]]:
        if self.mode == "per_sample":
            return {}
        elif self.mode == "per_dataset":
            measures = {
                Percentile(self.min_percentile, axes=self.axes),
                Percentile(self.max_percentile, axes=self.axes),
            }
            return {self.reference_tensor or self.tensor_name: measures}
        else:
            raise ValueError(self.mode)

    def get_required_sample_statistics(self) -> Dict[str, Set[Measure]]:
        if self.mode == "per_sample":
            measures = {
                Percentile(self.min_percentile, axes=self.axes),
                Percentile(self.max_percentile, axes=self.axes),
            }
            return {self.reference_tensor or self.tensor_name: measures}
        elif self.mode == "per_dataset":
            return {}
        else:
            raise ValueError(self.mode)

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        ref_name = self.reference_tensor or self.tensor_name
        if self.axes:
            axes = tuple(self.axes)
        else:
            axes = None

        if self.mode == "per_sample":
            get_stat = self.get_computed_sample_statistics
        elif self.mode == "per_dataset":
            get_stat = self.get_computed_dataset_statistics
        else:
            raise ValueError(self.mode)

        v_lower = get_stat(ref_name, Percentile(self.min_percentile, axes=axes))
        v_upper = get_stat(ref_name, Percentile(self.max_percentile, axes=axes))

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
    mode: Literal["fixed", "per_sample", "per_dataset"] = "per_sample"
    mean: Optional[float] = None
    std: Optional[float] = None
    axes: Optional[Sequence[str]] = None
    eps: float = 1.0e-6

    def get_required_dataset_statistics(self) -> Dict[str, Set[Measure]]:
        if self.mode == "per_dataset":
            return {self.tensor_name: {Mean(), Std()}}
        else:
            return {}

    def get_required_sample_statistics(self) -> Dict[str, Set[Measure]]:
        if self.mode == "per_sample":
            return {self.tensor_name: {Mean(), Std()}}
        else:
            return {}

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        axes = None if self.axes is None else tuple(self.axes)
        if self.mode == "fixed":
            assert self.mean is not None and self.std is not None
            mean, std = self.mean, self.std
        elif self.mode == "per_sample":
            if axes:
                mean, std = tensor.mean(axes), tensor.std(axes)
            else:
                mean, std = tensor.mean(), tensor.std()
        elif self.mode == "per_dataset":
            mean = self.get_computed_dataset_statistics(self.tensor_name, Mean(axes))
            std = self.get_computed_dataset_statistics(self.tensor_name, Std(axes))
        else:
            raise ValueError(self.mode)

        tensor = (tensor - mean) / (std + self.eps)
        return ensure_dtype(tensor, dtype="float32")
