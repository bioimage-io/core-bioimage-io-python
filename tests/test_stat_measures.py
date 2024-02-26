from itertools import product
from typing import Optional, Tuple

import numpy as np
import pytest
import xarray as xr

from bioimageio.core import stat_measures
from bioimageio.core.common import AxisId, Sample, TensorId
from bioimageio.core.stat_calculators import SamplePercentilesCalculator, get_measure_calculators
from bioimageio.core.stat_measures import SamplePercentile


@pytest.mark.parametrize("name, axes", product(["mean", "var", "std"], [None, (AxisId("x"), AxisId("y"))]))
def test_individual_normal_measure(name: str, axes: Optional[Tuple[AxisId, AxisId]]):
    measure = getattr(stat_measures, name.title() + "Measure")(axes=axes)
    data = xr.DataArray(np.random.random((5, 6, 3)), dims=("x", "y", "c"))

    expected = getattr(data, name)(dim=axes)
    actual = measure.compute(data)
    xr.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("axes", [None, (AxisId("x"), AxisId("y"))])
def test_individual_percentile_measure(axes: Optional[Tuple[AxisId, ...]]):
    ns = [0, 10, 50, 100]
    tid = TensorId("tensor")

    measures = [SamplePercentile(tensor_id=tid, axes=axes, n=n) for n in ns]
    calcs, _ = get_measure_calculators(measures)
    assert len(calcs) == 1
    calc = calcs[0]
    assert isinstance(calc, SamplePercentilesCalculator)

    data = xr.DataArray(np.random.random((5, 6, 3)), dims=("x", "y", "c"))
    actual = calc.compute(Sample(data={tid: data}))
    for m in measures:
        expected = data.quantile(q=m.n / 100, dim=m.axes)
        xr.testing.assert_allclose(expected, actual[m])
