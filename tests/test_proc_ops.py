from typing import Iterable, Optional, Tuple, Type, TypeVar

import numpy as np
import pytest
import xarray as xr
from typing_extensions import TypeGuard

from bioimageio.core.axis import AxisId
from bioimageio.core.sample import UntiledSample
from bioimageio.core.stat_calculators import compute_measures
from bioimageio.core.stat_measures import SampleMean, SamplePercentile, SampleStd
from bioimageio.core.tensor import TensorId


@pytest.fixture(scope="module")
def tid():
    return TensorId("data123")


def test_scale_linear(tid: TensorId):
    from bioimageio.core.proc_ops import ScaleLinear

    offset = xr.DataArray([1, 2, 42], dims=("c"))
    gain = xr.DataArray([1, 2, 3], dims=("c"))
    data = xr.DataArray(np.arange(6).reshape((1, 2, 3)), dims=("x", "y", "c"))
    sample = UntiledSample(data={tid: data})

    op = ScaleLinear(input=tid, output=tid, offset=offset, gain=gain)
    op(sample)

    expected = xr.DataArray(np.array([[[1, 4, 48], [4, 10, 57]]]), dims=("x", "y", "c"))
    xr.testing.assert_allclose(expected, sample.data[tid])


def test_scale_linear_no_channel(tid: TensorId):
    from bioimageio.core.proc_ops import ScaleLinear

    op = ScaleLinear(tid, tid, offset=1, gain=2)
    data = xr.DataArray(np.arange(6).reshape(2, 3), dims=("x", "y"))
    sample = UntiledSample(data={tid: data})
    op(sample)

    expected = xr.DataArray(np.array([[1, 3, 5], [7, 9, 11]]), dims=("x", "y"))
    xr.testing.assert_allclose(expected, sample.data[tid])


T = TypeVar("T")


def is_iterable(val: Iterable[T], inner: Type[T]) -> TypeGuard[Iterable[T]]:
    """Determines whether all objects in the list are strings"""
    return all(isinstance(x, inner) for x in val)


def test_zero_mean_unit_variance(tid: TensorId):
    from bioimageio.core.proc_ops import ZeroMeanUnitVariance

    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    sample = UntiledSample(data={tid: data})
    m = SampleMean(tid)
    std = SampleStd(tid)
    op = ZeroMeanUnitVariance(tid, tid, m, std)
    req = op.required_measures
    sample.stat = compute_measures(req, [sample])
    op(sample)

    expected = xr.DataArray(
        np.array(
            [
                [-1.54919274, -1.16189455, -0.77459637],
                [-0.38729818, 0.0, 0.38729818],
                [0.77459637, 1.16189455, 1.54919274],
            ]
        ),
        dims=("x", "y"),
    )
    xr.testing.assert_allclose(expected, sample.data[tid])


def test_zero_mean_unit_variance_fixed(tid: TensorId):
    from bioimageio.core.proc_ops import FixedZeroMeanUnitVariance

    op = FixedZeroMeanUnitVariance(
        tid,
        tid,
        mean=xr.DataArray([3, 4, 5], dims=("c")),
        std=xr.DataArray([2.44948974, 2.44948974, 2.44948974], dims=("c")),
    )
    data = xr.DataArray(np.arange(9).reshape((1, 3, 3)), dims=("b", "c", "x"))
    expected = xr.DataArray(
        np.array(
            [
                [
                    [-1.22474487, -0.81649658, -0.40824829],
                    [-0.40824829, 0.0, 0.40824829],
                    [0.40824829, 0.81649658, 1.22474487],
                ]
            ]
        ),
        dims=("b", "c", "x"),
    )
    sample = UntiledSample(data={tid: data})
    op(sample)
    xr.testing.assert_allclose(expected, sample.data[tid])


def test_zero_mean_unit_across_axes(tid: TensorId):
    from bioimageio.core.proc_ops import ZeroMeanUnitVariance

    data = xr.DataArray(np.arange(18).reshape((2, 3, 3)), dims=("c", "x", "y"))

    op = ZeroMeanUnitVariance(
        tid,
        tid,
        SampleMean(tid, (AxisId("x"), AxisId("y"))),
        SampleStd(tid, (AxisId("x"), AxisId("y"))),
    )
    sample = UntiledSample(data={tid: data})
    sample.stat = compute_measures(op.required_measures, [sample])

    expected = xr.concat(
        [(data[i : i + 1] - data[i].mean()) / data[i].std() for i in range(2)], dim="c"
    )
    op(sample)
    xr.testing.assert_allclose(expected, sample.data[tid])


def test_zero_mean_unit_variance_fixed2(tid: TensorId):
    from bioimageio.core.proc_ops import FixedZeroMeanUnitVariance

    np_data = np.arange(9).reshape(3, 3)
    mean = float(np_data.mean())
    std = float(np_data.mean())
    eps = 1.0e-7
    op = FixedZeroMeanUnitVariance(tid, tid, mean=mean, std=std, eps=eps)

    data = xr.DataArray(np_data, dims=("x", "y"))
    sample = UntiledSample(data={tid: data})
    expected = xr.DataArray((np_data - mean) / (std + eps), dims=("x", "y"))
    op(sample)
    xr.testing.assert_allclose(expected, sample.data[tid])


def test_binarize(tid: TensorId):
    from bioimageio.core.proc_ops import Binarize

    op = Binarize(tid, tid, threshold=14)
    data = xr.DataArray(np.arange(30).reshape((2, 3, 5)), dims=("x", "y", "c"))
    sample = UntiledSample(data={tid: data})
    expected = xr.zeros_like(data)
    expected[{"x": slice(1, None)}] = 1
    op(sample)
    xr.testing.assert_allclose(expected, sample.data[tid])


def test_binarize2(tid: TensorId):
    from bioimageio.core.proc_ops import Binarize

    shape = (3, 32, 32)
    axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)

    threshold = 0.5
    exp = xr.DataArray(np_data > threshold, dims=axes)

    sample = UntiledSample(data={tid: data})
    binarize = Binarize(tid, tid, threshold=threshold)
    binarize(sample)
    xr.testing.assert_allclose(exp, sample.data[tid])


def test_clip(tid: TensorId):
    from bioimageio.core.proc_ops import Clip

    op = Clip(tid, tid, min=3, max=5)
    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    sample = UntiledSample(data={tid: data})

    expected = xr.DataArray(
        np.array([[3, 3, 3], [3, 4, 5], [5, 5, 5]]), dims=("x", "y")
    )
    op(sample)
    xr.testing.assert_equal(expected, sample.data[tid])


def test_combination_of_op_steps_with_dims_specified(tid: TensorId):
    from bioimageio.core.proc_ops import ZeroMeanUnitVariance

    data = xr.DataArray(np.arange(18).reshape((2, 3, 3)), dims=("c", "x", "y"))
    sample = UntiledSample(data={tid: data})
    op = ZeroMeanUnitVariance(
        tid,
        tid,
        SampleMean(
            tid,
            (AxisId("x"), AxisId("y")),
        ),
        SampleStd(
            tid,
            (AxisId("x"), AxisId("y")),
        ),
    )
    sample.stat = compute_measures(op.required_measures, [sample])

    expected = xr.DataArray(
        np.array(
            [
                [
                    [-1.54919274, -1.16189455, -0.77459637],
                    [-0.38729818, 0.0, 0.38729818],
                    [0.77459637, 1.16189455, 1.54919274],
                ],
                [
                    [-1.54919274, -1.16189455, -0.77459637],
                    [-0.38729818, 0.0, 0.38729818],
                    [0.77459637, 1.16189455, 1.54919274],
                ],
            ]
        ),
        dims=("c", "x", "y"),
    )

    op(sample)
    xr.testing.assert_allclose(expected, sample.data[tid])


@pytest.mark.parametrize(
    "axes",
    [
        None,
        tuple(map(AxisId, "cy")),
        tuple(map(AxisId, "cyx")),
        tuple(map(AxisId, "x")),
    ],
)
def test_scale_mean_variance(tid: TensorId, axes: Optional[Tuple[AxisId, ...]]):
    from bioimageio.core.proc_ops import ScaleMeanVariance

    shape = (3, 32, 46)
    ipt_axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    ipt_data = xr.DataArray(np_data, dims=ipt_axes)
    ref_data = xr.DataArray((np_data * 2) + 3, dims=ipt_axes)

    op = ScaleMeanVariance(tid, tid, reference_tensor=TensorId("ref_name"), axes=axes)
    sample = UntiledSample(data={tid: ipt_data, TensorId("ref_name"): ref_data})
    sample.stat = compute_measures(op.required_measures, [sample])
    op(sample)
    xr.testing.assert_allclose(ref_data, sample.data[tid])


@pytest.mark.parametrize(
    "axes_str",
    [None, "cy", "y", "yx"],
)
def test_scale_mean_variance_per_channel(tid: TensorId, axes_str: Optional[str]):
    from bioimageio.core.proc_ops import ScaleMeanVariance

    axes = None if axes_str is None else tuple(map(AxisId, axes_str))

    shape = (3, 32, 46)
    ipt_axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    ipt_data = xr.DataArray(np_data, dims=ipt_axes)

    # set different mean, std per channel
    np_ref_data = np.stack([d * i + i for i, d in enumerate(np_data, start=2)])
    ref_data = xr.DataArray(np_ref_data, dims=ipt_axes)

    op = ScaleMeanVariance(tid, tid, reference_tensor=TensorId("ref_name"), axes=axes)
    sample = UntiledSample(data={tid: ipt_data, TensorId("ref_name"): ref_data})
    sample.stat = compute_measures(op.required_measures, [sample])
    op(sample)

    if axes is not None and AxisId("c") not in axes:
        # mean,std per channel should match exactly
        xr.testing.assert_allclose(ref_data, sample.data[tid])
    else:
        # mean,std across channels should not match
        with pytest.raises(AssertionError):
            xr.testing.assert_allclose(ref_data, sample.data[tid])


def test_scale_range(tid: TensorId):
    from bioimageio.core.proc_ops import ScaleRange

    op = ScaleRange(tid, tid)
    np_data = np.arange(9).reshape(3, 3).astype("float32")
    data = xr.DataArray(np_data, dims=("x", "y"))
    sample = UntiledSample(data={tid: data})
    sample.stat = compute_measures(op.required_measures, [sample])

    eps = 1.0e-6
    mi, ma = np_data.min(), np_data.max()
    exp_data = (np_data - mi) / (ma - mi + eps)
    expected = xr.DataArray(exp_data, dims=("x", "y"))

    op(sample)
    # NOTE xarray.testing.assert_allclose compares irrelavant properties here and fails although the result is correct
    np.testing.assert_allclose(expected, sample.data[tid])


def test_scale_range_axes(tid: TensorId):
    from bioimageio.core.proc_ops import ScaleRange

    lower_percentile = SamplePercentile(tid, 1, axes=(AxisId("x"), AxisId("y")))
    upper_percentile = SamplePercentile(tid, 100, axes=(AxisId("x"), AxisId("y")))
    op = ScaleRange(tid, tid, lower_percentile, upper_percentile)

    np_data = np.arange(18).reshape((2, 3, 3)).astype("float32")
    data = xr.DataArray(np_data, dims=("c", "x", "y"))
    sample = UntiledSample(data={tid: data})
    sample.stat = compute_measures(op.required_measures, [sample])

    eps = 1.0e-6
    p_low = np.percentile(np_data, lower_percentile.n, axis=(1, 2), keepdims=True)
    p_up = np.percentile(np_data, upper_percentile.n, axis=(1, 2), keepdims=True)
    exp_data = (np_data - p_low) / (p_up - p_low + eps)
    expected = xr.DataArray(exp_data, dims=("c", "x", "y"))

    op(sample)
    # NOTE xarray.testing.assert_allclose compares irrelavant properties here and fails although the result is correct
    np.testing.assert_allclose(expected, sample.data[tid])


def test_sigmoid(tid: TensorId):
    from bioimageio.core.proc_ops import Sigmoid

    shape = (3, 32, 32)
    axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)
    sample = UntiledSample(data={tid: data})
    sigmoid = Sigmoid(tid, tid)
    sigmoid(sample)

    exp = xr.DataArray(1.0 / (1 + np.exp(-np_data)), dims=axes)
    xr.testing.assert_allclose(exp, sample.data[tid])
