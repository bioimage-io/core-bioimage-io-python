from __future__ import annotations

import warnings
from collections.abc import Collection, Iterable, Mapping, Sequence
from copy import copy
from itertools import product
from typing import (
    Any,
    Union,
    cast,
)

import dask.array as da
import numpy as np
import xarray as xr
from dask.delayed import Delayed, delayed  # pyright: ignore[reportUnknownVariableType]
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import assert_never

from bioimageio.spec.model.v0_5 import BATCH_AXIS_ID

from ._utils.statistics import compute_quantiles
from .axis import AxisId
from .common import MemberId, QuantileMethod
from .sample import Sample
from .stat_measures import (
    DatasetMean,
    DatasetMeasure,
    DatasetMeasureBase,
    DatasetQuantile,
    DatasetStd,
    DatasetVar,
    Measure,
    MeasureValue,
    SampleMean,
    SampleMeasure,
    SampleQuantile,
    SampleStd,
    SampleVar,
)
from .tensor import Tensor

try:
    import crick  # pyright: ignore[reportMissingTypeStubs]

except Exception:
    crick = None

    class TDigest:
        def update(self, obj: Any):
            pass

        def quantile(self, q: Any) -> Any:
            pass

else:
    TDigest = crick.TDigest  # type: ignore


class MeanCalculator:
    """Build lazy sample and dataset means."""

    def __init__(self, member_id: MemberId, axes: Sequence[AxisId] | None):
        super().__init__()
        self._n: int = 0
        self._mean: Tensor | None = None
        self._axes = None if axes is None else tuple(axes)
        self._member_id = member_id

    def compute(self, sample: Sample) -> dict[SampleMean, MeasureValue]:
        return {
            SampleMean(member_id=self._member_id, axes=self._axes): self._compute_impl(
                sample
            )
        }

    def _compute_impl(self, sample: Sample) -> Tensor:
        tensor = sample.members[self._member_id].as_dask().astype("float64", copy=False)
        return tensor.mean(dim=self._axes)

    def update(self, sample: Sample) -> None:
        mean = self._compute_impl(sample)
        self._update_impl(sample.members[self._member_id], mean)

    def compute_and_update(self, sample: Sample) -> dict[SampleMean, MeasureValue]:
        mean = self._compute_impl(sample)
        self._update_impl(sample.members[self._member_id], mean)
        return {SampleMean(member_id=self._member_id, axes=self._axes): mean}

    def _update_impl(self, tensor: Tensor, tensor_mean: Tensor):
        assert tensor_mean.dtype == "float64"
        # reduced voxel count
        n_b = int(tensor.size / tensor_mean.size)

        if self._mean is None:
            assert self._n == 0
            self._n = n_b
            self._mean = tensor_mean
        else:
            assert self._n != 0
            n_a = self._n
            mean_old = self._mean
            self._n = n_a + n_b
            self._mean = (n_a * mean_old + n_b * tensor_mean) / self._n
            assert self._mean.dtype == "float64"

    def finalize(self) -> dict[DatasetMean, MeasureValue]:
        if self._mean is None:
            return {}
        else:
            return {DatasetMean(member_id=self._member_id, axes=self._axes): self._mean}


class MeanVarStdCalculator:
    """to calculate sample and dataset mean, variance or standard deviation"""

    def __init__(self, member_id: MemberId, axes: Sequence[AxisId] | None):
        super().__init__()
        self._axes = None if axes is None else tuple(map(AxisId, axes))
        self._member_id = member_id
        self._n: int = 0
        self._mean: Tensor | None = None
        self._m2: Tensor | None = None

    def compute(
        self, sample: Sample
    ) -> dict[SampleMean | SampleVar | SampleStd, MeasureValue]:
        tensor = sample.members[self._member_id].as_dask()
        mean = tensor.mean(dim=self._axes)
        c = (tensor - mean).data
        if self._axes is None:
            n = tensor.size
        else:
            n = int(np.prod([tensor.sizes[d] for d in self._axes]))

        if xr.__version__.startswith("2023"):
            var = xr.dot(c, c, dims=self._axes) / n
        else:
            var = xr.dot(c, c, dim=self._axes) / n

        assert isinstance(var, xr.DataArray)
        std = np.sqrt(var)
        assert isinstance(std, xr.DataArray)
        return {
            SampleMean(axes=self._axes, member_id=self._member_id): mean,
            SampleVar(axes=self._axes, member_id=self._member_id): Tensor.from_xarray(
                var
            ),
            SampleStd(axes=self._axes, member_id=self._member_id): Tensor.from_xarray(
                std
            ),
        }

    def update(self, sample: Sample):
        if self._axes is not None and BATCH_AXIS_ID not in self._axes:
            return

        tensor = sample.members[self._member_id].as_dask().astype("float64", copy=False)
        mean_b = tensor.mean(dim=self._axes)
        assert mean_b.dtype == "float64"
        # reduced voxel count
        n_b = int(tensor.size / mean_b.size)
        m2_b = ((tensor - mean_b) ** 2).sum(dim=self._axes)
        assert m2_b.dtype == "float64"
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
            assert self._mean.dtype == "float64"
            d = mean_b - mean_a
            self._m2 = m2_a + m2_b + d**2 * n_a * n_b / n
            assert self._m2.dtype == "float64"

    def finalize(
        self,
    ) -> dict[DatasetMean | DatasetVar | DatasetStd, MeasureValue]:
        if (
            self._axes is not None
            and BATCH_AXIS_ID not in self._axes
            or self._mean is None
        ):
            return {}
        else:
            assert self._m2 is not None
            var = self._m2 / self._n
            sqrt = var**0.5
            if isinstance(sqrt, (int, float)):
                # var and mean are scalar tensors, let's keep it consistent
                sqrt = Tensor.from_xarray(xr.DataArray(sqrt))

            assert isinstance(sqrt, Tensor), type(sqrt)
            return {
                DatasetMean(member_id=self._member_id, axes=self._axes): self._mean,
                DatasetVar(member_id=self._member_id, axes=self._axes): var,
                DatasetStd(member_id=self._member_id, axes=self._axes): sqrt,
            }


class SampleQuantilesCalculator:
    """to calculate sample quantiles"""

    def __init__(
        self,
        member_id: MemberId,
        axes: Sequence[AxisId] | None,
        qs: Collection[float],
        method: QuantileMethod = "linear",
    ):
        super().__init__()
        assert all(0.0 <= q <= 1.0 for q in qs)
        self._qs = sorted(set(qs))
        self._axes = None if axes is None else tuple(axes)
        self._member_id = member_id
        self._method: QuantileMethod = method

    def compute(self, sample: Sample) -> dict[SampleQuantile, MeasureValue]:
        ps = compute_quantiles(
            sample.members[self._member_id], self._qs, self._axes, self._method
        )
        return {
            SampleQuantile(
                q=q, axes=self._axes, member_id=self._member_id, method=self._method
            ): p
            for q, p in zip(self._qs, ps)
        }


class MeanQuantilesCalculator:
    """to calculate dataset quantiles heuristically by averaging across samples

    Note:
        The returned dataset quantiles are an estiamte and **not mathematically correct**
    """

    def __init__(
        self,
        member_id: MemberId,
        axes: Sequence[AxisId] | None,
        qs: Collection[float],
    ):
        super().__init__()
        assert all(0.0 <= q <= 1.0 for q in qs)
        self._qs = sorted(set(qs))
        self._axes = None if axes is None else tuple(axes)
        self._member_id = member_id
        self._n: int = 0
        self._estimates: Tensor | None = None

    def update(self, sample: Sample):
        tensor = sample.members[self._member_id]
        sample_estimates = compute_quantiles(
            tensor, self._qs, self._axes, "linear"
        ).astype("float64", copy=False)

        # reduced voxel count
        n = int(tensor.size / np.prod(sample_estimates.shape_tuple[1:]))

        if self._estimates is None:
            assert self._n == 0
            self._estimates = sample_estimates
        else:
            self._estimates = (self._n * self._estimates + n * sample_estimates) / (
                self._n + n
            )
            assert self._estimates.dtype == "float64"

        self._n += n

    def finalize(self) -> dict[DatasetQuantile, MeasureValue]:
        if self._estimates is None:
            return {}
        else:
            warnings.warn(
                "Computed dataset percentiles naively by averaging percentiles of samples."
            )
            return {
                DatasetQuantile(q=q, axes=self._axes, member_id=self._member_id): e
                for q, e in zip(self._qs, self._estimates)
            }


def _update_crick_digests(
    previous: list[TDigest] | None,
    data: NDArray[Any],
    remaining_shape: tuple[int, ...],
) -> list[TDigest]:
    assert crick is not None
    # A new state keeps earlier results stable across updates and repeated computes.
    digests = (
        [copy(d) for d in previous]
        if previous is not None
        else [TDigest() for _ in range(int(np.prod(remaining_shape)))]
    )
    for digest, idx in zip(digests, product(*map(range, remaining_shape))):
        digest.update(data[idx])
    return digests


def _get_crick_quantiles(
    digests: list[TDigest], qs: tuple[float, ...], remaining_shape: tuple[int, ...]
) -> NDArray[Any]:
    return np.asarray([[d.quantile(q) for d in digests] for q in qs]).reshape(
        (len(qs), *remaining_shape)
    )


class CrickQuantilesCalculator:
    """to calculate dataset quantiles with the experimental [crick libray](https://github.com/dask/crick)"""

    def __init__(
        self,
        member_id: MemberId,
        axes: Sequence[AxisId] | None,
        qs: Collection[float],
    ):
        warnings.warn("Computing dataset quantiles with experimental 'crick' library.")
        super().__init__()
        assert all(0.0 <= q <= 1.0 for q in qs)
        assert axes is None or "_quantiles" not in axes
        self._qs = sorted(set(qs))
        self._axes = None if axes is None else tuple(axes)
        self._member_id = member_id
        self._digest: Delayed | None = None
        self._dims: tuple[AxisId, ...] | None = None
        self._shape: tuple[int, ...] | None = None

    def update(self, part: Sample):
        tensor = part.members[self._member_id].as_dask()
        assert "_quantiles" not in tensor.dims
        dims = tuple(
            d for d in tensor.dims if self._axes is not None and d not in self._axes
        )
        if self._dims is not None and set(dims) == set(self._dims):
            dims = self._dims
        shape = tuple(tensor.sizes[d] for d in dims)
        if self._dims is not None and (dims != self._dims or shape != self._shape):
            raise ValueError("Dataset quantiles require matching unreduced dimensions.")

        self._dims = dims
        self._shape = shape
        array = tensor.transpose((*dims, *(d for d in tensor.dims if d not in dims)))
        data = array.data.data
        assert isinstance(data, da.Array)
        # Preserve upstream keys so statistics and preprocessing share input reads.
        rechunked = cast(da.Array, data.rechunk(data.shape))  # pyright: ignore[reportArgumentType]
        deferred_data = cast(
            Delayed,
            rechunked.to_delayed(optimize_graph=False).item(),  # pyright: ignore[reportAttributeAccessIssue]
        )
        # shortcut: materializes one sample per task; use chunked digests for larger inputs.
        self._digest = delayed(_update_crick_digests, pure=True)(
            self._digest, deferred_data, shape
        )

    def finalize(self) -> dict[DatasetQuantile, MeasureValue]:
        if self._digest is None:
            return {}
        else:
            assert self._dims is not None
            assert self._shape is not None

            vs = cast(
                da.Array,
                da.from_delayed(
                    delayed(_get_crick_quantiles, pure=True)(
                        self._digest, tuple(self._qs), self._shape
                    ),
                    shape=(len(self._qs), *self._shape),
                    dtype="float64",
                ),
            )
            return {
                DatasetQuantile(
                    q=q, axes=self._axes, member_id=self._member_id
                ): Tensor(cast(da.Array, vs[i]), dims=self._dims)
                for i, q in enumerate(self._qs)
            }


if crick is None:
    DatasetQuantilesCalculator: type[
        MeanQuantilesCalculator | CrickQuantilesCalculator
    ] = MeanQuantilesCalculator
else:
    DatasetQuantilesCalculator = CrickQuantilesCalculator


class NaiveSampleMeasureCalculator:
    """wrapper for measures to match interface of other sample measure calculators"""

    def __init__(self, member_id: MemberId, measure: SampleMeasure):
        super().__init__()
        self.tensor_name = member_id
        self.measure = measure

    def compute(self, sample: Sample) -> dict[SampleMeasure, MeasureValue]:
        return {self.measure: self.measure.compute(sample.as_dask())}


SampleMeasureCalculator = Union[
    MeanCalculator,
    MeanVarStdCalculator,
    SampleQuantilesCalculator,
    NaiveSampleMeasureCalculator,
]
DatasetMeasureCalculator = Union[
    MeanCalculator, MeanVarStdCalculator, DatasetQuantilesCalculator
]


class StatsCalculator:
    """Builds lazy dataset and sample statistics without evaluating tensor data."""

    def __init__(
        self,
        measures: Collection[Measure],
        initial_dataset_measures: Mapping[DatasetMeasure, MeasureValue] | None = None,
    ):
        super().__init__()
        self.sample_count = 0
        self.sample_calculators, self.dataset_calculators = get_measure_calculators(
            measures
        )
        if not initial_dataset_measures:
            self._current_dataset_measures: (
                dict[DatasetMeasure, MeasureValue] | None
            ) = None
        else:
            missing_dataset_meas = {
                m
                for m in measures
                if isinstance(m, DatasetMeasureBase)
                and m not in initial_dataset_measures
            }
            if missing_dataset_meas:
                logger.debug(
                    f"ignoring `initial_dataset_measure` as it is missing {missing_dataset_meas}"
                )
                self._current_dataset_measures = None
            else:
                self._current_dataset_measures = dict(initial_dataset_measures)

    @property
    def has_dataset_measures(self):
        return self._current_dataset_measures is not None

    def update(
        self,
        sample: Sample | Iterable[Sample],
    ) -> None:
        _ = self._update(sample)

    def finalize(self) -> dict[DatasetMeasure, MeasureValue]:
        """Return lazy aggregated statistics for the samples registered so far."""
        if self._current_dataset_measures is None:
            self._current_dataset_measures = {}
            for calc in self.dataset_calculators:
                values = calc.finalize()
                self._current_dataset_measures.update(values.items())

        return self._current_dataset_measures

    def update_and_get_all(
        self,
        sample: Sample | Iterable[Sample],
    ) -> dict[Measure, MeasureValue]:
        """Returns sample as well as updated dataset statistics"""
        last_sample = self._update(sample)
        if last_sample is None:
            raise ValueError("`sample` was not a `Sample`, nor did it yield any.")

        return {**self._compute(last_sample), **self.finalize()}

    def skip_update_and_get_all(self, sample: Sample) -> dict[Measure, MeasureValue]:
        """Returns sample as well as previously computed dataset statistics"""
        return {**self._compute(sample), **self.finalize()}

    def _compute(self, sample: Sample) -> dict[SampleMeasure, MeasureValue]:
        ret: dict[SampleMeasure, MeasureValue] = {}
        for calc in self.sample_calculators:
            values = calc.compute(sample)
            ret.update(values.items())

        return ret

    def _update(self, sample: Sample | Iterable[Sample]) -> Sample | None:
        self.sample_count += 1
        samples = [sample] if isinstance(sample, Sample) else sample
        last_sample = None
        for el in samples:
            last_sample = el
            for calc in self.dataset_calculators:
                calc.update(el)

        self._current_dataset_measures = None
        return last_sample


def get_measure_calculators(
    required_measures: Iterable[Measure],
) -> tuple[list[SampleMeasureCalculator], list[DatasetMeasureCalculator]]:
    """determines which calculators are needed to compute the required measures efficiently"""

    sample_calculators: list[SampleMeasureCalculator] = []
    dataset_calculators: list[DatasetMeasureCalculator] = []

    # split required measures into groups
    required_sample_means: set[SampleMean] = set()
    required_dataset_means: set[DatasetMean] = set()
    required_sample_mean_var_std: set[tuple[MemberId, tuple[AxisId, ...] | None]] = (
        set()
    )
    required_dataset_mean_var_std: set[tuple[MemberId, tuple[AxisId, ...] | None]] = (
        set()
    )
    required_sample_quantiles: dict[
        tuple[MemberId, tuple[AxisId, ...] | None, QuantileMethod], set[float]
    ] = {}
    required_dataset_quantiles: dict[
        tuple[MemberId, tuple[AxisId, ...] | None], set[float]
    ] = {}

    for rm in required_measures:
        if isinstance(rm, SampleMean):
            required_sample_means.add(rm)
        elif isinstance(rm, DatasetMean):
            required_dataset_means.add(rm)
        elif isinstance(rm, (SampleVar, SampleStd)):
            required_sample_mean_var_std.add((rm.member_id, rm.axes))
        elif isinstance(rm, (DatasetVar, DatasetStd)):
            required_dataset_mean_var_std.add((rm.member_id, rm.axes))
        elif isinstance(rm, SampleQuantile):
            required_sample_quantiles.setdefault(
                (rm.member_id, rm.axes, rm.method), set()
            ).add(rm.q)
        elif isinstance(rm, DatasetQuantile):
            required_dataset_quantiles.setdefault((rm.member_id, rm.axes), set()).add(
                rm.q
            )
        else:
            assert_never(rm)

    for rm in required_sample_means:
        if (rm.member_id, rm.axes) in required_sample_mean_var_std:
            # computed togehter with var and std
            continue

        sample_calculators.append(MeanCalculator(member_id=rm.member_id, axes=rm.axes))

    for tid, axes in required_sample_mean_var_std:
        sample_calculators.append(MeanVarStdCalculator(member_id=tid, axes=axes))

    for rm in required_dataset_means:
        if (rm.member_id, rm.axes) in required_dataset_mean_var_std:
            # computed togehter with var and std
            continue

        dataset_calculators.append(MeanCalculator(member_id=rm.member_id, axes=rm.axes))

    for tid, axes in required_dataset_mean_var_std:
        dataset_calculators.append(MeanVarStdCalculator(member_id=tid, axes=axes))

    for (tid, axes, m), qs in required_sample_quantiles.items():
        sample_calculators.append(
            SampleQuantilesCalculator(member_id=tid, axes=axes, qs=qs, method=m)
        )

    for (tid, axes), qs in required_dataset_quantiles.items():
        dataset_calculators.append(
            DatasetQuantilesCalculator(member_id=tid, axes=axes, qs=qs)
        )

    return sample_calculators, dataset_calculators


def compute_dataset_measures(
    measures: Iterable[DatasetMeasure], dataset: Iterable[Sample]
) -> dict[DatasetMeasure, MeasureValue]:
    """Build lazy dataset measures, consuming samples without evaluating their data."""
    sample_calculators, calculators = get_measure_calculators(measures)
    assert not sample_calculators

    ret: dict[DatasetMeasure, MeasureValue] = {}

    for sample in dataset:
        for calc in calculators:
            calc.update(sample)

    for calc in calculators:
        ret.update(calc.finalize().items())

    return ret


def compute_sample_measures(
    measures: Iterable[SampleMeasure], sample: Sample
) -> dict[SampleMeasure, MeasureValue]:
    """Build lazy sample measures; convert the returned tensors to NumPy to evaluate."""
    calculators, dataset_calculators = get_measure_calculators(measures)
    assert not dataset_calculators
    ret: dict[SampleMeasure, MeasureValue] = {}

    for calc in calculators:
        ret.update(calc.compute(sample).items())

    return ret


def compute_measures(
    measures: Iterable[Measure], dataset: Iterable[Sample]
) -> dict[Measure, MeasureValue]:
    """Build lazy measures for the dataset and its last sample.

    The dataset iterable is consumed without evaluating tensor data.
    """
    sample_calculators, dataset_calculators = get_measure_calculators(measures)
    ret: dict[Measure, MeasureValue] = {}
    sample = None
    for sample in dataset:
        for calc in dataset_calculators:
            calc.update(sample)
    if sample is None:
        raise ValueError("empty dataset")

    for calc in dataset_calculators:
        ret.update(calc.finalize().items())

    for calc in sample_calculators:
        ret.update(calc.compute(sample).items())

    return ret
