import dataclasses
from itertools import product

import numpy as np
import numpy.testing
import pytest
import xarray as xr

from bioimageio.core import statistical_measures
from bioimageio.core.prediction_pipeline._measure_groups import get_measure_groups
from bioimageio.core.prediction_pipeline._utils import PER_DATASET, PER_SAMPLE
from bioimageio.core.statistical_measures import Mean, Percentile, Std, Var


@pytest.mark.parametrize("name_axes", product(["mean", "var", "std"], [None, ("x", "y")]))
def test_individual_normal_measure(name_axes):
    name, axes = name_axes
    measure = getattr(statistical_measures, name.title())(axes=axes)
    data = xr.DataArray(np.random.random((5, 6, 3)), dims=("x", "y", "c"))

    expected = getattr(data, name)(dim=axes)
    actual = measure.compute(data)
    xr.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("axes_n", product([None, ("x", "y")], [0, 10, 50, 100]))
def test_individual_percentile_measure(axes_n):
    axes, n = axes_n
    measure = statistical_measures.Percentile(axes=axes, n=n)
    data = xr.DataArray(np.random.random((5, 6, 3)), dims=("x", "y", "c"))

    expected = data.quantile(q=n / 100, dim=axes)
    actual = measure.compute(data)
    xr.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize(
    "measures_mode",
    product(
        [
            {"t1": {Mean()}, "t2": {Mean(), Std()}},
            {"t1": {Mean(), Var(), Std()}, "t2": {Std(axes=("x", "y"))}},
            {"t1": {Mean(axes=("x", "y"))}, "t2": {Mean(), Std(axes=("x", "y"))}},
            {
                "t1": {Percentile(n=10), Percentile(n=35), Percentile(n=10, axes=("x", "y"))},
                "t2": {Percentile(n=10, axes=("x", "y")), Percentile(n=35, axes=("x", "y")), Percentile(n=10)},
            },
        ],
        [PER_SAMPLE, PER_DATASET],
    ),
)
def test_measure_groups(measures_mode):
    measures, mode = measures_mode

    def get_sample():
        return {
            "t1": xr.DataArray(np.random.random((2, 500, 600, 3)), dims=("b", "x", "y", "c")),
            "t2": xr.DataArray(np.random.random((1, 500, 600)), dims=("c", "x", "y")),
        }

    sample = get_sample()
    dataset_seq = [sample, get_sample()]
    dataset_full = {tn: xr.concat([s[tn] for s in dataset_seq], dim="dataset") for tn in sample.keys()}

    # compute independently
    expected = {}
    for tn, ms in measures.items():
        for m in ms:
            if mode == PER_SAMPLE:
                expected[(tn, m)] = m.compute(sample[tn])
            elif mode == PER_DATASET:
                if m.axes is None:
                    m_d = m
                else:
                    m_d = dataclasses.replace(m, axes=("dataset",) + m.axes)

                expected[(tn, m)] = m_d.compute(dataset_full[tn])
            else:
                raise NotImplementedError(mode)

    groups = get_measure_groups({mode: measures})[mode]
    actual = {}
    for g in groups:
        if mode == PER_SAMPLE:
            res = g.compute(sample)
        elif mode == PER_DATASET:
            for s in dataset_seq:
                g.update_with_sample(s)

            res = g.finalize()
        else:
            raise NotImplementedError(mode)

        for tn, vs in res.items():
            for m, v in vs.items():
                actual[(tn, m)] = v

    # discard additionally computed measures by groups
    actual = {k: v for k, v in actual.items() if k in expected}

    for k in expected.keys():
        assert k in actual
        numpy.testing.assert_array_almost_equal(expected[k].data, actual[k].data, decimal=2)
