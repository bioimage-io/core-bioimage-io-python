from __future__ import annotations

import warnings
from itertools import product
from typing import Dict, Hashable, Iterator, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy
import xarray as xr

from bioimageio.core.prediction_pipeline import TensorName
from bioimageio.core.statistical_measures import Mean, Measure, Percentile, Std, Var

try:
    import crick
except ImportError:
    crick = None

MeasureValue = xr.DataArray


class MeasureGroup:
    def reset(self):
        raise NotImplementedError

    def update_with_sample(self, sample: Dict[TensorName, xr.DataArray]):
        raise NotImplementedError

    def finalize(self) -> Dict[Measure, MeasureValue]:
        raise NotImplementedError


class MeanVarStd(MeasureGroup):
    n: int
    mean: Optional[xr.DataArray]
    m2: Optional[xr.DataArray]

    def __init__(self, tensor_name: TensorName, axes: Optional[Tuple[int]]):
        self.axes: Optional[Tuple[str]] = axes
        self.tensor_name = tensor_name
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def update_with_sample(self, sample: Dict[TensorName, xr.DataArray]):
        tensor = sample[self.tensor_name]
        mean_b = tensor.mean(dim=self.axes)
        n_b = numpy.prod(tensor.shape) / numpy.prod(mean_b.shape)  # reduced voxel count
        m2_b = ((tensor - mean_b) ** 2).sum(dim=self.axes)
        if self.n == 0:
            assert self.m2 is None
            self.n = n_b
            self.m2 = m2_b
        else:
            n_a = self.n
            mean_a = self.mean
            m2_a = self.m2
            self.n = n = n_a + n_b
            self.mean = (n_a * mean_a + n_b * mean_b) / n
            d = mean_b - mean_a
            self.m2 = m2_a + m2_b + d ** 2 * n_a * n_b / n

    def finalize(self) -> Dict[Measure, MeasureValue]:
        var = self.m2 / self.n
        return {Mean(axes=self.axes): self.mean, Var(axes=self.axes): var, Std(axes=self.axes): xr.ufuncs.sqrt(var)}


class MeanPercentiles(MeasureGroup):
    count: int
    estimates: Optional[xr.DataArray]

    def __init__(self, tensor_name: TensorName, axes: Optional[Tuple[str]], ns: Sequence[float]):
        assert all(0 <= n <= 100 for n in ns)
        warnings.warn(f"Computing dataset percentiles naively by averaging percentiles of samples.")
        self.ns = ns
        self.qs = [n / 100 for n in ns]
        self.axes = axes
        self.tensor_name = tensor_name
        self.reset()

    def reset(self):
        self.count = 0
        self.estimates = None

    def update_with_sample(self, sample: Dict[TensorName, xr.DataArray]):
        tensor = sample[self.tensor_name]
        sample_estimates = tensor.quantile(self.qs, dim=self.axes)

        n = numpy.prod(tensor.shape) / numpy.prod(sample_estimates.shape[1:])  # reduced voxel count

        if self.count == 0:
            self.estimates = sample_estimates
        else:
            self.estimates = (self.count * self.estimates + n * sample_estimates) / (self.count + n)

        self.count += n

    def finalize(self) -> Dict[Percentile, MeasureValue]:
        assert self.estimates is not None
        return {Percentile(n=n, axes=self.axes): e for n, e in zip(self.ns, self.estimates)}


class TDigestPercentiles(MeasureGroup):
    digest: Optional[Union["crick.TDigest", List["crick.TDigest"]]]
    dims: Optional[Tuple[Hashable, ...]]
    indices: Optional[Iterator[Tuple[int, ...]]]
    reduce_all_axes: bool
    shape: Optional[Tuple[int, ...]]

    def __init__(self, tensor_name: TensorName, axes: Optional[Tuple[str]], ns: Sequence[float]):
        assert all(0 <= n <= 100 for n in ns)
        warnings.warn(f"Computing dataset percentiles with experimental 'crick' library.")
        self.ns = ns
        self.qs = [n / 100 for n in ns]
        self.axes = axes
        self.tensor_name = tensor_name
        self.reset()

    def reset(self):
        self.digest = None
        self.dims = None
        self.indices = None
        self.shape = None

    def _initialize(self, tensor_sizes: Mapping[Hashable, int]):
        reduce_all_axes = self.axes is None
        if reduce_all_axes:
            out_sizes = None
        else:
            out_sizes = {d: s for d, s in tensor_sizes.items() if d not in self.axes}
            if not out_sizes:
                out_sizes = None
                reduce_all_axes = True

        if reduce_all_axes:
            self.digest = crick.TDigest()
        else:
            self.dims, self.shape = zip(*out_sizes.items())
            self.digest = [crick.TDigest() for _ in range(numpy.prod(self.shape))]
            self.indices = product(*map(range, self.shape))

    def update_with_sample(self, sample: Dict[TensorName, xr.DataArray]):
        tensor = sample[self.tensor_name]
        if self.digest is None:
            self._initialize(tensor.sizes)

        assert self.digest is not None
        if isinstance(self.digest, list):
            for i, idx in enumerate(self.indices):
                self.digest[i].update(tensor.isel(dict(zip(self.dims, idx))))
        else:
            self.digest.update(tensor)

    def finalize(self) -> Dict[Measure, MeasureValue]:
        if isinstance(self.digest, list):
            vs = [[d.quantile(q) for d in self.digest] for q in self.qs]
        else:
            vs = [self.digest.quantile(q) for q in self.qs]

        return {Percentile(n=n, axes=self.axes): xr.DataArray(v, dims=self.dims) for n, v in zip(self.ns, vs)}


if crick is None:
    PercentileGroup: Union[Type[MeanPercentiles], Type[TDigestPercentiles]] = MeanPercentiles
else:
    PercentileGroup = TDigestPercentiles
