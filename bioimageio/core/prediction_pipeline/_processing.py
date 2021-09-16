from dataclasses import InitVar, dataclass, field
from itertools import chain
from typing import (
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
)

import numpy as np
import xarray as xr

from bioimageio.core.resource_io import nodes
from ._types import Transform
from .statistical_measures import Mean, Measure, Percentile, Std
from bioimageio.spec.model.v0_3.raw_nodes import PreprocessingName


def ensure_dtype(tensor: xr.DataArray, *, dtype) -> xr.DataArray:
    """
    Convert array to a given datatype
    """
    return tensor.astype(dtype)


@dataclass
class Processing:
    apply_to: str
    computed_statistics: Dict[str, Dict[str, Measure]]

    def get_required_statistics(self) -> Dict[str, Dict[str, Measure]]:
        """
        specifies which dataset measures are required for which purpose from what tensor.
        Returns: dataset measures required to apply this processing indexed by <tensor_name> and <role>:
                 Dict[<tensor_name>, Dict[<role>, Measure]]

        """
        return {}

    def set_computed_statistics(self, computed: Dict[str, Dict[str, Measure]]):
        """helper to set computed statistics and check if they match the requirements"""
        for tensor_measures in computed.values():
            for measure in tensor_measures.values():
                assert measure.value is not None, "encountered uncomputed measure"

        for tensor_name, req_measures in self.get_required_statistics():
            comp_measures = computed.get(tensor_name, {})
            for name, req_measure in req_measures.items():
                comp_measure = comp_measures.get(name)
                assert isinstance(comp_measure, type(req_measure))

        self.computed_statistics = computed

    def get_computed_statistics(self, tensor_name: str, measure_name: str):
        """helper to unpack self.computed_statistics"""
        ret = self.computed_statistics.get(tensor_name, {}).get(measure_name)
        if ret is None:
            raise RuntimeError(f"Missing {measure_name} for {tensor_name} dataset.")

        return ret

    def apply(self, **tensors: xr.DataArray) -> Dict[str, xr.DataArray]:
        """apply processing to named tensors; call 'apply_simple' as default"""
        tensors[self.apply_to] = self.apply_simple(tensors[self.apply_to])
        return tensors

    def apply_simple(self, tensor: xr.DataArray) -> xr.DataArray:
        """apply processing to single tensor"""
        raise NotImplementedError

    def __post_init__(self):
        """validate common kwargs by their annotations"""
        if hasattr(self, "mode"):
            if self.mode not in get_args(self.mode):
                raise NotImplementedError(f"Unsupported mode {self.mode} for {self.__class__.__name__}: {self.mode}")


@dataclass
class ScaleLinear(Processing):
    """scale the tensor with a fixed multiplicative and additive factor"""

    gain: float
    offset: float
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


@dataclass
class ZeroMeanUnitVariance(Processing):
    mode: Literal["fixed", "per_sample", "per_dataset"] = "per_sample"
    mean: Optional[float] = None
    std: Optional[float] = None
    axes: Optional[Sequence[str]] = None
    eps: float = 1.0e-6

    def get_required_statistics(self) -> Dict[str, Dict[str, Measure]]:
        if self.mode == "per_dataset":
            return {self.apply_to: {"mean": Mean(), "std": Std()}}
        else:
            return {}

    def apply(self, **tensors: xr.DataArray) -> Dict[str, xr.DataArray]:
        tensor = tensors[self.apply_to]
        if self.mode == "fixed":
            assert self.mean is not None and self.std is not None
            mean, std = self.mean, self.std
        elif self.mode == "per_sample":
            if self.axes:
                axes = tuple(self.axes)
                mean, std = tensor.mean(axes), tensor.std(axes)
            else:
                mean, std = tensor.mean(), tensor.std()
        elif self.mode == "per_dataset":
            mean = self.get_computed_statistics(self.apply_to, "mean")
            std = self.get_computed_statistics(self.apply_to, "std")
        else:
            raise ValueError(self.mode)

        tensor = (tensor - mean) / (std + self.eps)
        tensors[self.apply_to] = ensure_dtype(tensor, dtype="float32")

        return tensors


@dataclass
class Binarize(Processing):
    threshold: float

    def apply_simple(self, tensor: xr.DataArray) -> xr.DataArray:
        return ensure_dtype(tensor > self.threshold, dtype="float32")


@dataclass
class ScaleRange(Processing):
    mode: Literal["per_sample", "per_dataset"] = "per_sample"
    axes: Optional[Sequence[str]] = None
    min_percentile: float = 0.0
    max_percentile: float = 100.0
    reference_tensor: Optional[str] = None

    def get_required_statistics(self) -> Dict[str, Dict[str, Measure]]:
        if self.mode == "per_sample":
            return {}
        elif self.mode == "per_dataset":
            measures = {"v_lower": Percentile(self.min_percentile), "v_upper": Percentile(self.max_percentile)}
            return {self.reference_tensor or self.apply_to: measures}
        else:
            raise ValueError(self.mode)

    def apply(self, **tensors: xr.DataArray) -> Dict[str, xr.DataArray]:
        ref_name = self.reference_tensor or self.apply_to
        if self.mode == "per_sample":
            ref_tensor = tensors[ref_name]
            if self.axes:
                axes = tuple(self.axes)
            else:
                axes = None

            v_lower = ref_tensor.quantile(self.min_percentile / 100.0, dim=axes)
            v_upper = ref_tensor.quantile(self.max_percentile / 100.0, dim=axes)
        elif self.mode == "per_dataset":
            v_lower = self.get_computed_statistics(ref_name, "v_lower")
            v_upper = self.get_computed_statistics(ref_name, "v_upper")
        else:
            raise ValueError(self.mode)

        tensors[self.apply_to] = ensure_dtype((tensors[self.apply_to] - v_lower) / v_upper, dtype="float32")
        return tensors


# todo: continue here....
@dataclass
class Clip(Processing):
    min: float
    max: float

    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return ensure_dtype(tensor.clip(min=self.min, max=self.max), dtype="float32")


@dataclass
class Sigmoid(Processing):
    def apply(self, tensor: xr.DataArray) -> xr.DataArray:
        return 1.0 / (1.0 + xr.ufuncs.exp(-tensor))


KNOWN_PREPROCESSING: Dict[PreprocessingName, Type[Processing]] = {
    "scale_linear": ScaleLinear,
    "zero_mean_unit_variance": ZeroMeanUnitVariance,
    "binarize": Binarize,
    "clip": Clip,
    "scale_range": ScaleRange,
    "sigmoid": Sigmoid,
}


class CombinedProcessing:
    def __init__(
        self,
        processing_spec: Union[List[nodes.Preprocessing], List[nodes.Postprocessing]],
        input_tensor_names: Sequence[str],
        output_tensor_names: Sequence[str] = tuple(),
    ):
        prep = all(isinstance(ps, nodes.Preprocessing) for ps in processing_spec)
        assert prep or all(isinstance(ps, nodes.Postprocessing) for ps in processing_spec)

        self.tensor_names = input_tensor_names if prep else output_tensor_names
        self.tensor_names = input_tensor_names if prep else output_tensor_names
        self.procs = [KNOWN_PREPROCESSING.get(step.name)(**step.kwargs) for step in processing_spec]
