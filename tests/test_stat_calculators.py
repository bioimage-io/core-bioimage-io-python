from typing import Tuple, Union

import numpy as np
import pytest
from xarray.testing import assert_allclose  # pyright: ignore[reportUnknownVariableType]

from bioimageio.core.axis import AxisId
from bioimageio.core.sample import UntiledSample
from bioimageio.core.stat_calculators import MeanVarStdCalculator
from bioimageio.core.stat_measures import (
    DatasetMean,
    DatasetStd,
    DatasetVar,
)
from bioimageio.core.tensor import Tensor, TensorId


def create_random_dataset(tid: TensorId, axes: Tuple[AxisId, ...]):
    n = 3
    sizes = list(range(n, len(axes) + 1))
    data = np.asarray(np.random.rand(*sizes))
    ds = [
        UntiledSample(data={tid: Tensor(data[i : i + 1], dims=axes, id=tid)})
        for i in range(n)
    ]
    return Tensor(data, dims=axes), ds


@pytest.mark.parametrize(
    "axes",
    [
        None,
        ("x", "y"),
        ("channel", "y"),
    ],
)
def test_mean_var_std_calculator(axes: Union[None, str, Tuple[str, ...]]):
    tid = TensorId("tensor")
    axes = tuple(map(AxisId, ("batch", "channel", "x", "y")))
    data, ds = create_random_dataset(tid, axes)
    expected_mean = data.mean(axes)
    expected_var = data.var(axes)
    expected_std = data.std(axes)

    calc = MeanVarStdCalculator(tid, axes=axes)
    for s in ds:
        calc.update(s)

    actual = calc.finalize()
    actual_mean = actual[DatasetMean(tid, axes=axes)]
    actual_var = actual[DatasetVar(tid, axes=axes)]
    actual_std = actual[DatasetStd(tid, axes=axes)]

    assert_allclose(actual_mean, expected_mean)
    assert_allclose(actual_var, expected_var)
    assert_allclose(actual_std, expected_std)
