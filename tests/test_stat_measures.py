from itertools import product
from typing import Optional, Tuple

import numpy as np
import pytest
import xarray as xr

from bioimageio.core import stat_measures
from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId
from bioimageio.core.sample import Sample
from bioimageio.core.stat_calculators import (
    SamplePercentilesCalculator,
    get_measure_calculators,
)
from bioimageio.core.stat_measures import SamplePercentile
from bioimageio.core.tensor import Tensor


@pytest.mark.parametrize(
    "name,axes",
    product(
        ["mean", "var", "std"],
        [None, (AxisId("c"),), (AxisId("x"), AxisId("y"))],
    ),
)
def test_individual_normal_measure(
    name: str,
    axes: Optional[Tuple[AxisId, AxisId]],
):
    data_id = MemberId("test_data")
    measure = getattr(stat_measures, "Sample" + name.title())(
        axes=axes, member_id=data_id
    )
    data = Tensor(
        np.random.random((5, 6, 3)), dims=(AxisId("x"), AxisId("y"), AxisId("c"))
    )

    expected = getattr(data, name)(dim=axes)
    sample = Sample(members={data_id: data})
    actual = measure.compute(sample)
    xr.testing.assert_allclose(expected.data, actual.data)


@pytest.mark.parametrize("axes", [None, (AxisId("x"), AxisId("y"))])
def test_individual_percentile_measure(axes: Optional[Tuple[AxisId, ...]]):
    qs = [0, 0.1, 0.5, 1.0]
    tid = MemberId("tensor")

    measures = [SamplePercentile(member_id=tid, axes=axes, q=q) for q in qs]
    calcs, _ = get_measure_calculators(measures)
    assert len(calcs) == 1
    calc = calcs[0]
    assert isinstance(calc, SamplePercentilesCalculator)

    data = Tensor(
        np.random.random((5, 6, 3)), dims=(AxisId("x"), AxisId("y"), AxisId("c"))
    )
    actual = calc.compute(Sample(members={tid: data}))
    for m in measures:
        expected = data.quantile(q=m.q, dim=m.axes)
        actual_data = actual[m]
        if isinstance(actual_data, Tensor):
            actual_data = actual_data.data

        xr.testing.assert_allclose(expected.data, actual_data)
