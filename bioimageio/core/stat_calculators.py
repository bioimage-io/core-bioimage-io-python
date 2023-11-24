from __future__ import annotations

import collections
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import field
from itertools import product
from typing import (
    Any,
    ClassVar,
    DefaultDict,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    OrderedDict,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from bioimageio.core.common import (
    PER_DATASET,
    PER_SAMPLE,
    AxisId,
    DatasetMeasure,
    MeasureVar,
    RequiredMeasure,
    Sample,
    SampleMeasure,
    TensorId,
)
from bioimageio.core.stat_measures import Mean, Measure, Percentile, Std, Var

try:
    import crick  # type: ignore
except ImportError:
    crick = None

MeasureValue = Union[xr.DataArray, float]


class SampleMeasureCalculator(ABC, Generic[MeasureVar]):
    """group of measures for more efficient computation of multiple measures per sample"""

    @abstractmethod
    def compute(self, sample: Sample) -> Mapping[SampleMeasure[MeasureVar], MeasureValue]:
        ...


class DatasetMeasureCalculator(ABC, Generic[MeasureVar]):
    """group of measures for more efficient computation of multiple measures per dataset"""

    @abstractmethod
    def update_with_sample(self, sample: Sample) -> None:
        """update intermediate representation with a data sample"""
        ...

    @abstractmethod
    def finalize(self) -> Mapping[DatasetMeasure[MeasureVar], MeasureValue]:
        """compute statistics from intermediate representation"""
        ...


class MeanCalculator(SampleMeasureCalculator[Mean], DatasetMeasureCalculator[Mean]):
    def __init__(self, tensor_id: TensorId, axes: Optional[Sequence[AxisId]]):
        super().__init__()
        self._axes = None if axes is None else tuple(axes)
        self._tensor_id = tensor_id
        self._n: int = 0
        self._mean: Optional[xr.DataArray] = None

    def compute(self, sample: Sample):
        return {
            SampleMeasure(measure=Mean(axes=self._axes), tensor_id=self._tensor_id): sample[self._tensor_id].mean(
                dim=self._axes
            )
        }

    def update_with_sample(self, sample: Sample):
        tensor = sample[self._tensor_id].astype(np.float64, copy=False)
        mean_b = tensor.mean(dim=self._axes)
        assert mean_b.dtype == np.float64
        n_b = np.prod(list(tensor.shape)) / np.prod(list(mean_b.shape))  # reduced voxel count
        if self._mean is None:
            assert self._n == 0
            self._n = n_b
            self._mean = mean_b
        else:
            assert self._n != 0
            n_a = self._n
            mean_a = self._mean
            self._n = n = n_a + n_b
            self._mean = (n_a * mean_a + n_b * mean_b) / n
            assert self._mean.dtype == np.float64

    def finalize(self) -> Mapping[DatasetMeasure, MeasureValue]:
        if self._mean is None:
            return {}
        else:
            return {DatasetMeasure(measure=Mean(axes=self._axes), tensor_id=self._tensor_id): self._mean}


class MeanVarStdCalculator(SampleMeasureCalculator, DatasetMeasureCalculator):
    def __init__(self, tensor_id: TensorId, axes: Optional[Sequence[AxisId]]):
        super().__init__()
        self._axes = None if axes is None else tuple(axes)
        self._tensor_id = tensor_id
        self._n: int = 0
        self._mean: Optional[xr.DataArray] = None
        self._m2: Optional[xr.DataArray] = None

    def compute(self, sample: Sample):
        tensor = sample[self._tensor_id]
        mean = tensor.mean(dim=self._axes)
        c = tensor - mean
        if self._axes is None:
            n = tensor.size
        else:
            n = int(np.prod([tensor.sizes[d] for d in self._axes]))  # type: ignore  # FIXME: type annotation

        var = xr.dot(c, c, dims=self._axes) / n
        std = np.sqrt(var)
        return {
            SampleMeasure(Mean(axes=self._axes), tensor_id=self._tensor_id): mean,
            SampleMeasure(Var(axes=self._axes), tensor_id=self._tensor_id): var,
            SampleMeasure(Std(axes=self._axes), tensor_id=self._tensor_id): std,
        }

    def update_with_sample(self, sample: Sample):
        tensor = sample[self._tensor_id].astype(np.float64, copy=False)
        mean_b = tensor.mean(dim=self._axes)
        assert mean_b.dtype == np.float64
        # reduced voxel count
        n_b = int(np.prod(tensor.shape) / np.prod(mean_b.shape))  # type: ignore
        m2_b = ((tensor - mean_b) ** 2).sum(dim=self._axes)
        assert m2_b.dtype == np.float64
        if self._mean is None:
            assert self._m2 is None
            self._n = n_b
            self._mean = mean_b
            self._m2 = m2_b
        else:
            n_a = self._n
            mean_a = self._mean
            m2_a = self._m2
            self._n = n = n_a + n_b
            self._mean = (n_a * mean_a + n_b * mean_b) / n
            assert self._mean.dtype == np.float64
            d = mean_b - mean_a
            self._m2 = m2_a + m2_b + d**2 * n_a * n_b / n
            assert self._m2.dtype == np.float64

    def finalize(self) -> Mapping[DatasetMeasure, MeasureValue]:
        if self._mean is None:
            return {}
        else:
            assert self._m2 is not None
            var = self._m2 / self._n
            sqrt: xr.DataArray = np.sqrt(var)  # type: ignore
            return {
                DatasetMeasure(tensor_id=self._tensor_id, measure=Mean(axes=self._axes)): self._mean,
                DatasetMeasure(tensor_id=self._tensor_id, measure=Var(axes=self._axes)): var,
                DatasetMeasure(tensor_id=self._tensor_id, measure=Std(axes=self._axes)): sqrt,
            }


class SamplePercentilesCalculator(SampleMeasureCalculator):
    def __init__(self, tensor_id: TensorId, axes: Optional[Sequence[AxisId]], ns: Sequence[float]):
        super().__init__()
        assert all(0 <= n <= 100 for n in ns)
        self.ns = ns
        self._qs = [n / 100 for n in ns]
        self._axes = None if axes is None else tuple(axes)
        self._tensor_id = tensor_id

    def compute(self, sample: Sample):
        tensor = sample[self._tensor_id]
        ps = tensor.quantile(self._qs, dim=self._axes)  # type:  ignore
        return {
            SampleMeasure(measure=Percentile(n=n, axes=self._axes), tensor_id=self._tensor_id): p
            for n, p in zip(self.ns, ps)
        }


class MeanPercentilesCalculator(DatasetMeasureCalculator):
    def __init__(self, tensor_id: TensorId, axes: Optional[Sequence[AxisId]], ns: Sequence[float]):
        super().__init__()
        assert all(0 <= n <= 100 for n in ns)
        self._ns = ns
        self._qs = np.asarray([n / 100 for n in ns])
        self._axes = None if axes is None else tuple(axes)
        self._tensor_id = tensor_id
        self._n: int = 0
        self._estimates: Optional[xr.DataArray] = None

    def update_with_sample(self, sample: Sample):
        tensor = sample[self._tensor_id]
        sample_estimates = tensor.quantile(self._qs, dim=self._axes).astype(np.float64, copy=False)

        # reduced voxel count
        n = int(np.prod(tensor.shape) / np.prod(sample_estimates.shape[1:]))  # type: ignore

        if self._estimates is None:
            assert self._n == 0
            self._estimates = sample_estimates
        else:
            self._estimates = (self._n * self._estimates + n * sample_estimates) / (self._n + n)
            assert self._estimates.dtype == np.float64

        self._n += n

    def finalize(self) -> Mapping[DatasetMeasure, MeasureValue]:
        if self._estimates is None:
            return {}
        else:
            warnings.warn("Computed dataset percentiles naively by averaging percentiles of samples.")
            return {
                DatasetMeasure(measure=Percentile(n=n, axes=self._axes), tensor_id=self._tensor_id): e
                for n, e in zip(self._ns, self._estimates)
            }


class CrickPercentilesCalculator(DatasetMeasureCalculator):
    def __init__(self, tensor_name: TensorId, axes: Optional[Sequence[AxisId]], ns: Sequence[float]):
        warnings.warn("Computing dataset percentiles with experimental 'crick' library.")
        super().__init__()
        assert all(0 <= n <= 100 for n in ns)
        assert axes is None or "_percentiles" not in axes
        self._ns = ns
        self._qs = [n / 100 for n in ns]
        self._axes = None if axes is None else tuple(axes)
        self._tensor_id = tensor_name
        self._digest: Optional[List[crick.TDigest]] = None
        self._dims: Optional[Tuple[Hashable, ...]] = None
        self._indices: Optional[Iterator[Tuple[int, ...]]] = None
        self._shape: Optional[Tuple[int, ...]] = None

    def _initialize(self, tensor_sizes: Mapping[Hashable, int]):
        assert crick is not None
        out_sizes: OrderedDict[Hashable, int] = collections.OrderedDict(_percentiles=len(self._ns))
        if self._axes is not None:
            for d, s in tensor_sizes.items():
                if d not in self._axes:
                    out_sizes[d] = s

        self._dims, self._shape = zip(*out_sizes.items())
        d = int(np.prod(self._shape[1:]))  # type: ignore
        self._digest = [crick.TDigest() for _ in range(d)]
        self._indices = product(*map(range, self._shape[1:]))

    def update_with_sample(self, sample: Sample):
        tensor = sample[self._tensor_id]
        assert "_percentiles" not in tensor.dims
        if self._digest is None:
            self._initialize(tensor.sizes)

        assert self._digest is not None
        assert self._indices is not None
        assert self._dims is not None
        for i, idx in enumerate(self._indices):
            self._digest[i].update(tensor.isel(dict(zip(self._dims[1:], idx))))

    def finalize(self) -> Dict[DatasetMeasure, MeasureValue]:
        if self._digest is None:
            return {}
        else:
            assert self._dims is not None
            vs: NDArray[Any] = np.asarray([[d.quantile(q) for d in self._digest] for q in self._qs]).reshape(self._shape)  # type: ignore
            return {
                DatasetMeasure(measure=Percentile(n=n, axes=self._axes), tensor_id=self._tensor_id): xr.DataArray(
                    v, dims=self._dims[1:]
                )
                for n, v in zip(self._ns, vs)
            }


if crick is None:
    DatasetPercentileCalculator: Type[
        Union[MeanPercentilesCalculator, CrickPercentilesCalculator]
    ] = MeanPercentilesCalculator
else:
    DatasetPercentileCalculator = CrickPercentilesCalculator


class NaivSampleMeasureCalculator(SampleMeasureCalculator):
    """wrapper for measures to match interface of SampleMeasureGroup"""

    def __init__(self, tensor_id: TensorId, measure: Measure):
        super().__init__()
        self.tensor_name = tensor_id
        self.measure = measure

    def compute(self, sample: Sample) -> Mapping[SampleMeasure, MeasureValue]:
        return {
            SampleMeasure(measure=self.measure, tensor_id=self.tensor_name): self.measure.compute(
                sample[self.tensor_name]
            )
        }


def get_measure_calculators(
    required_measures: Iterable[RequiredMeasure],
) -> Tuple[List[SampleMeasureCalculator], List[DatasetMeasureCalculator]]:
    """determines which calculators are needed to compute the required measures efficiently"""

    sample_calculators: List[SampleMeasureCalculator] = []
    dataset_calculators: List[DatasetMeasureCalculator] = []

    # split required measures into groups
    required_means: Set[RequiredMeasure] = set()
    required_mean_var_std: Set[RequiredMeasure] = set()
    required_percentiles: Set[RequiredMeasure] = set()

    for rm in required_measures:
        if isinstance(rm.measure, Mean):
            required_means.add(rm)
        elif isinstance(rm.measure, (Var, Std)):
            required_mean_var_std.update(
                {
                    RequiredMeasure(measure=msv(rm.measure.axes), tensor_id=rm.tensor_id, mode=rm.mode)
                    for msv in (Mean, Std, Var)
                }
            )
            assert rm in required_mean_var_std
        elif isinstance(rm.measure, Percentile):
            required_percentiles.add(rm)
        elif rm.mode == PER_SAMPLE:
            sample_calculators.append(NaivSampleMeasureCalculator(tensor_id=rm.tensor_id, measure=rm.measure))
        else:
            raise NotImplementedError(f"Computing statistics for {rm.measure} {rm.mode} not yet implemented")

    for rm in required_means:
        if rm in required_mean_var_std:
            # computed togehter with var and std
            continue

        if rm.mode == PER_SAMPLE:
            sample_calculators.append(MeanCalculator(tensor_id=rm.tensor_id, axes=rm.measure.axes))
        # add all mean measures that are not included in a mean/var/std group
        for tn, m in means:
            if (tn, m.axes) not in required_mean_var_std:
                # compute only mean
                if mode == PER_SAMPLE:
                    calculators[mode].append(NaivSampleMeasureCalculator(tensor_id=tn, measure=m))
                elif mode == PER_DATASET:
                    calculators[mode].append(DatasetMeanCalculator(tensor_id=tn, axes=m.axes))
                else:
                    raise NotImplementedError(mode)

        for tn, axes in mean_var_std_groups:
            calculators[mode].append(MeanVarStdCalculator(tensor_id=tn, axes=axes))

        for (tn, axes), ns in required_percentiles.items():
            if mode == PER_SAMPLE:
                calculators[mode].append(SamplePercentilesCalculator(tensor_id=tn, axes=axes, ns=ns))
            elif mode == PER_DATASET:
                calculators[mode].append(DatasetPercentileCalculator(tensor_name=tn, axes=axes, ns=ns))
            else:
                raise NotImplementedError(mode)

    return calculators


def compute_measures(
    measures: RequiredMeasures, *, sample: Optional[Sample] = None, dataset: Iterator[Sample] = ()
) -> ComputedMeasures:
    ms_groups = get_measure_calculators(measures)
    ret = {PER_SAMPLE: {}, PER_DATASET: {}}
    if sample is not None:
        for mg in ms_groups[PER_SAMPLE]:
            assert isinstance(mg, SampleMeasureCalculator)
            ret[PER_SAMPLE].update(mg.compute(sample))

    for sample in dataset:
        for mg in ms_groups[PER_DATASET]:
            assert isinstance(mg, DatasetMeasureCalculator)
            mg.update_with_sample(sample)

    for mg in ms_groups[PER_DATASET]:
        assert isinstance(mg, DatasetMeasureCalculator)
        ret[PER_DATASET].update(mg.finalize())

    return ret
