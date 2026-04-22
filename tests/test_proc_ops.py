from pathlib import Path
from typing import Iterable, Optional, Tuple, Type, TypeVar

import numpy as np
import pytest
import scipy  # pyright: ignore[reportMissingTypeStubs]
import xarray as xr
from typing_extensions import TypeGuard

from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId
from bioimageio.core.sample import Sample
from bioimageio.core.stat_calculators import compute_measures
from bioimageio.core.stat_measures import SampleMean, SampleQuantile, SampleStd
from bioimageio.core.tensor import Tensor


@pytest.fixture(scope="module")
def tid():
    return MemberId("data123")


def test_scale_linear(tid: MemberId):
    from bioimageio.core.proc_ops import ScaleLinear

    offset = xr.DataArray([1, 2, 42], dims=("channel",))
    gain = xr.DataArray([1, 2, 3], dims=("channel",))
    data = xr.DataArray(np.arange(6).reshape((1, 2, 3)), dims=("x", "y", "channel"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)

    op = ScaleLinear(input=tid, output=tid, offset=offset, gain=gain)
    op(sample)

    expected = xr.DataArray(
        np.array([[[1, 4, 48], [4, 10, 57]]]), dims=("x", "y", "channel")
    )
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_scale_linear_no_channel(tid: MemberId):
    from bioimageio.core.proc_ops import ScaleLinear

    op = ScaleLinear(tid, tid, offset=1, gain=2)
    data = xr.DataArray(np.arange(6).reshape(2, 3), dims=("x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    op(sample)

    expected = xr.DataArray(np.array([[1, 3, 5], [7, 9, 11]]), dims=("x", "y"))
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


T = TypeVar("T")


def is_iterable(val: Iterable[T], inner: Type[T]) -> TypeGuard[Iterable[T]]:
    """Determines whether all objects in the list are strings"""
    return all(isinstance(x, inner) for x in val)


def test_zero_mean_unit_variance(tid: MemberId):
    from bioimageio.core.proc_ops import ZeroMeanUnitVariance

    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    m = SampleMean(member_id=tid)
    std = SampleStd(member_id=tid)
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
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_zero_mean_unit_variance_fixed(tid: MemberId):
    from bioimageio.core.proc_ops import FixedZeroMeanUnitVariance

    op = FixedZeroMeanUnitVariance(
        tid,
        tid,
        mean=xr.DataArray([3, 4, 5], dims=("channel",)),
        std=xr.DataArray([2.44948974, 2.44948974, 2.44948974], dims=("channel",)),
    )
    data = xr.DataArray(np.arange(9).reshape((1, 3, 3)), dims=("batch", "channel", "x"))
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
        dims=("batch", "channel", "x"),
    )
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    op(sample)
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_zero_mean_unit_variance_fixed2(tid: MemberId):
    from bioimageio.core.proc_ops import FixedZeroMeanUnitVariance

    np_data = np.arange(9).reshape(3, 3)
    mean = float(np_data.mean())
    std = float(np_data.mean())
    eps = 1.0e-7
    op = FixedZeroMeanUnitVariance(tid, tid, mean=mean, std=std, eps=eps)

    data = xr.DataArray(np_data, dims=("x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    expected = xr.DataArray((np_data - mean) / (std + eps), dims=("x", "y"))
    op(sample)
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_zero_mean_unit_across_axes(tid: MemberId):
    from bioimageio.core.proc_ops import ZeroMeanUnitVariance

    data = xr.DataArray(np.arange(18).reshape((2, 3, 3)), dims=("channel", "x", "y"))

    op = ZeroMeanUnitVariance(
        tid,
        tid,
        SampleMean(member_id=tid, axes=(AxisId("x"), AxisId("y"))),
        SampleStd(member_id=tid, axes=(AxisId("x"), AxisId("y"))),
    )
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    sample.stat = compute_measures(op.required_measures, [sample])

    expected = xr.concat(
        [(data[i : i + 1] - data[i].mean()) / data[i].std() for i in range(2)],
        dim="channel",
    )
    op(sample)
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_binarize(tid: MemberId):
    from bioimageio.core.proc_ops import Binarize

    op = Binarize(tid, tid, threshold=14)
    data = xr.DataArray(np.arange(30).reshape((2, 3, 5)), dims=("x", "y", "channel"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    expected = xr.zeros_like(data)
    expected[{"x": slice(1, None)}] = 1
    op(sample)
    xr.testing.assert_allclose(expected, sample.members[tid].data)


def test_binarize2(tid: MemberId):
    from bioimageio.core.proc_ops import Binarize

    shape = (3, 32, 32)
    axes = ("channel", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)

    threshold = 0.5
    exp = xr.DataArray(np_data > threshold, dims=axes)

    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    binarize = Binarize(tid, tid, threshold=threshold)
    binarize(sample)
    xr.testing.assert_allclose(exp, sample.members[tid].data)


def test_clip(tid: MemberId):
    from bioimageio.core.proc_ops import Clip

    op = Clip(tid, tid, min=3, max=5)
    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)

    expected = xr.DataArray(
        np.array([[3, 3, 3], [3, 4, 5], [5, 5, 5]]), dims=("x", "y")
    )
    op(sample)
    xr.testing.assert_equal(expected, sample.members[tid].data)


def test_clip_percentiles():
    from bioimageio.core.proc_ops import Clip
    from bioimageio.core.stat_measures import SampleQuantile
    from bioimageio.spec.model.v0_5 import AxisId, ClipDescr, ClipKwargs

    descr = ClipDescr(
        kwargs=ClipKwargs(min_percentile=30, max_percentile=70, axes=(AxisId("x"),))
    )
    op = Clip.from_proc_descr(
        descr,
        member_id=MemberId("data"),
    )
    assert op.required_measures == {
        SampleQuantile(
            member_id=MemberId("data"),
            scope="sample",
            name="quantile",
            q=0.3,
            axes=(AxisId("x"),),
            method="inverted_cdf",
        ),
        SampleQuantile(
            member_id=MemberId("data"),
            scope="sample",
            name="quantile",
            q=0.7,
            axes=(AxisId("x"),),
            method="inverted_cdf",
        ),
    }

    data = xr.DataArray(np.arange(15).reshape(3, 5), dims=("channel", "x"))
    sample = Sample(
        members={MemberId("data"): Tensor.from_xarray(data)}, stat={}, id=None
    )
    sample.stat = compute_measures(op.required_measures, [sample])

    expected = xr.DataArray(
        np.array([[1, 1, 2, 3, 3], [6, 6, 7, 8, 8], [11, 11, 12, 13, 13]]),
        dims=("channel", "x"),
    )
    op(sample)
    xr.testing.assert_equal(expected, sample.members[MemberId("data")].data)


def test_combination_of_op_steps_with_dims_specified(tid: MemberId):
    from bioimageio.core.proc_ops import ZeroMeanUnitVariance

    data = xr.DataArray(np.arange(18).reshape((2, 3, 3)), dims=("channel", "x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    op = ZeroMeanUnitVariance(
        tid,
        tid,
        SampleMean(
            member_id=tid,
            axes=(AxisId("x"), AxisId("y")),
        ),
        SampleStd(
            member_id=tid,
            axes=(AxisId("x"), AxisId("y")),
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
        dims=("channel", "x", "y"),
    )

    op(sample)
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "axes",
    [
        None,
        tuple(map(AxisId, "cy")),
        tuple(map(AxisId, "cyx")),
        tuple(map(AxisId, "x")),
    ],
)
def test_scale_mean_variance(tid: MemberId, axes: Optional[Tuple[AxisId, ...]]):
    from bioimageio.core.proc_ops import ScaleMeanVariance

    shape = (3, 32, 46)
    ipt_axes = ("channel", "y", "x")
    np_data = np.random.rand(*shape)
    ipt_data = xr.DataArray(np_data, dims=ipt_axes)
    ref_data = xr.DataArray((np_data * 2) + 3, dims=ipt_axes)

    op = ScaleMeanVariance(tid, tid, reference_tensor=MemberId("ref_name"), axes=axes)
    sample = Sample(
        members={
            tid: Tensor.from_xarray(ipt_data),
            MemberId("ref_name"): Tensor.from_xarray(ref_data),
        },
        stat={},
        id=None,
    )
    sample.stat = compute_measures(op.required_measures, [sample])
    op(sample)
    xr.testing.assert_allclose(ref_data, sample.members[tid].data, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "axes_str",
    [None, "cy", "y", "yx"],
)
def test_scale_mean_variance_per_channel(tid: MemberId, axes_str: Optional[str]):
    from bioimageio.core.proc_ops import ScaleMeanVariance

    axes = None if axes_str is None else tuple(map(AxisId, axes_str))

    shape = (3, 32, 46)
    ipt_axes = ("channel", "y", "x")
    np_data = np.random.rand(*shape)
    ipt_data = xr.DataArray(np_data, dims=ipt_axes)

    # set different mean, std per channel
    np_ref_data = np.stack([d * i + i for i, d in enumerate(np_data, start=2)])
    ref_data = xr.DataArray(np_ref_data, dims=ipt_axes)

    op = ScaleMeanVariance(tid, tid, reference_tensor=MemberId("ref_name"), axes=axes)
    sample = Sample(
        members={
            tid: Tensor.from_xarray(ipt_data),
            MemberId("ref_name"): Tensor.from_xarray(ref_data),
        },
        stat={},
        id=None,
    )
    sample.stat = compute_measures(op.required_measures, [sample])
    op(sample)

    if axes is not None and AxisId("channel") not in axes:
        # mean,std per channel should match exactly
        xr.testing.assert_allclose(
            ref_data, sample.members[tid].data, rtol=1e-5, atol=1e-7
        )
    else:
        # mean,std across channels should not match
        with pytest.raises(AssertionError):
            xr.testing.assert_allclose(
                ref_data, sample.members[tid].data, rtol=1e-5, atol=1e-7
            )


def test_scale_range(tid: MemberId):
    from bioimageio.core.proc_ops import ScaleRange

    op = ScaleRange(tid, tid)
    np_data = np.arange(9).reshape(3, 3).astype("float32")
    data = xr.DataArray(np_data, dims=("x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    sample.stat = compute_measures(op.required_measures, [sample])

    eps = 1.0e-6
    mi, ma = np_data.min(), np_data.max()
    exp_data = (np_data - mi) / (ma - mi + eps)
    expected = xr.DataArray(exp_data, dims=("x", "y"))

    op(sample)
    # NOTE xarray.testing.assert_allclose compares irrelavant properties here and fails although the result is correct
    np.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_scale_range_axes(tid: MemberId):
    from bioimageio.core.proc_ops import ScaleRange

    eps = 1.0e-6

    lower_quantile = SampleQuantile(
        member_id=tid, q=0.1, axes=(AxisId("x"), AxisId("y")), method="linear"
    )
    upper_quantile = SampleQuantile(
        member_id=tid, q=0.9, axes=(AxisId("x"), AxisId("y")), method="linear"
    )
    op = ScaleRange(tid, tid, lower_quantile, upper_quantile, eps=eps)

    np_data = np.arange(18).reshape((2, 3, 3)).astype("float32")
    data = Tensor.from_xarray(xr.DataArray(np_data, dims=("channel", "x", "y")))
    sample = Sample(members={tid: data}, stat={}, id=None)

    p_low_direct = lower_quantile.compute(sample)
    p_up_direct = upper_quantile.compute(sample)

    p_low_expected = np.quantile(np_data, lower_quantile.q, axis=(1, 2), keepdims=True)
    p_up_expected = np.quantile(np_data, upper_quantile.q, axis=(1, 2), keepdims=True)

    np.testing.assert_allclose(p_low_expected.squeeze(), p_low_direct)
    np.testing.assert_allclose(p_up_expected.squeeze(), p_up_direct)

    sample.stat = compute_measures(op.required_measures, [sample])

    np.testing.assert_allclose(p_low_expected.squeeze(), sample.stat[lower_quantile])
    np.testing.assert_allclose(p_up_expected.squeeze(), sample.stat[upper_quantile])

    exp_data = (np_data - p_low_expected) / (p_up_expected - p_low_expected + eps)
    expected = xr.DataArray(exp_data, dims=("channel", "x", "y"))

    op(sample)
    # NOTE xarray.testing.assert_allclose compares irrelavant properties here and fails although the result is correct
    np.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_sigmoid(tid: MemberId):
    from bioimageio.core.proc_ops import Sigmoid

    shape = (3, 32, 32)
    axes = ("channel", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    sigmoid = Sigmoid(tid, tid)
    sigmoid(sample)

    exp = xr.DataArray(1.0 / (1 + np.exp(-np_data)), dims=axes)
    xr.testing.assert_allclose(exp, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_softmax(tid: MemberId):
    from bioimageio.core.proc_ops import Softmax

    shape = (3, 32, 32)
    axes = ("channel", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    softmax = Softmax(tid, tid, axis=AxisId("channel"))
    softmax(sample)

    exp = xr.DataArray(
        np.exp(np_data - np.max(np_data, axis=0, keepdims=True))
        / np.sum(np.exp(np_data - np.max(np_data, axis=0, keepdims=True)), axis=0),
        dims=axes,
    )
    xr.testing.assert_allclose(exp, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_softmax_with_scipy(tid: MemberId):
    from bioimageio.core.proc_ops import Softmax

    shape = (3, 32, 32)
    axes = ("channel", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    softmax = Softmax(tid, tid, axis=AxisId("channel"))
    softmax(sample)

    exp = xr.DataArray(
        scipy.special.softmax(np_data, axis=0),
        dims=axes,
    )
    xr.testing.assert_allclose(exp, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_softmax_from_spec_descr(tid: MemberId, tmp_path: Path):
    """Verify the full spec→runtime path for softmax postprocessing.

    SoftmaxDescr (bioimageio.spec v0.5.9+) must be:
      1. Accepted as a valid postprocessing descriptor by the spec.
      2. Correctly dispatched by get_proc() to the Softmax operator.
      3. Numerically identical to scipy.special.softmax over the channel axis.

    This test guards against regressions where softmax is defined in the spec
    but silently dropped or misrouted in the core runtime. It also documents
    that softmax postprocessing IS supported end-to-end without needing to
    embed it inside the model's forward() method.
    """
    from bioimageio.core.proc_ops import Softmax, get_proc
    from bioimageio.spec.model import v0_5

    shape = (3, 16, 16)
    axes = ("channel", "y", "x")
    np_data = np.random.rand(*shape).astype(np.float32)
    data = xr.DataArray(np_data, dims=axes)
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)

    # Write a dummy test tensor so FileDescr validates
    test_npy = tmp_path / "test.npy"
    np.save(test_npy, np_data)

    # Build spec descriptor — SoftmaxDescr requires bioimageio.spec >= 0.5.9
    softmax_descr = v0_5.SoftmaxDescr(
        kwargs=v0_5.SoftmaxKwargs(axis=AxisId("channel"))
    )
    tensor_descr = v0_5.OutputTensorDescr(
        id=v0_5.TensorId(str(tid)),
        axes=[
            v0_5.ChannelAxis(
                channel_names=[
                    v0_5.Identifier("c0"),
                    v0_5.Identifier("c1"),
                    v0_5.Identifier("c2"),
                ]
            ),
            v0_5.SpaceOutputAxis(id=AxisId("y"), size=16),
            v0_5.SpaceOutputAxis(id=AxisId("x"), size=16),
        ],
        test_tensor=v0_5.FileDescr(source=test_npy),
        postprocessing=[softmax_descr],
    )

    # Dispatch through get_proc — must return a Softmax instance
    op = get_proc(softmax_descr, tensor_descr)
    assert isinstance(op, Softmax), f"Expected Softmax, got {type(op)}"

    # Apply and verify numerical correctness
    op(sample)
    expected = scipy.special.softmax(np_data, axis=0)
    xr.testing.assert_allclose(
        xr.DataArray(expected, dims=axes),
        sample.members[tid].data,
        rtol=1e-5,
        atol=1e-7,
    )

    # Output must be a valid probability distribution (sums to 1 over channel axis)
    out_data = sample.members[tid].data
    assert out_data is not None
    channel_sums = out_data.sum(dim="channel")
    xr.testing.assert_allclose(
        channel_sums, xr.ones_like(channel_sums), atol=1e-6, rtol=0
    )


def test_custom_postprocessing_callable_class(tid: MemberId) -> None:
    """CustomPostprocessing loads a callable class from source bytes and applies it."""
    from bioimageio.core.proc_ops import CustomPostprocessing

    source_code = b"""
import numpy as np

class double_values:
    def __init__(self, scale: float = 2.0) -> None:
        self.scale = scale
    def __call__(self, *arrays):
        return (arrays[0] * self.scale).astype(np.float32)
"""
    data = xr.DataArray(np.ones((2, 3), dtype=np.float32), dims=("y", "x"))
    out_id = MemberId("out")
    sample = Sample(members={out_id: Tensor.from_xarray(data)}, stat={}, id=None)

    op = CustomPostprocessing(
        output_id=out_id,
        input_ids=[out_id],
        callable_name="double_values",
        source_code=source_code,
        kwargs={"scale": 3.0},
    )
    op(sample)

    expected = xr.DataArray(np.full((2, 3), 3.0, dtype=np.float32), dims=("y", "x"))
    xr.testing.assert_allclose(expected, sample.members[out_id].data, rtol=1e-5, atol=1e-7)


def test_custom_postprocessing_factory_function(tid: MemberId) -> None:
    """CustomPostprocessing also works with a factory-function style callable."""
    from bioimageio.core.proc_ops import CustomPostprocessing

    source_code = b"""
import numpy as np

def threshold_op(threshold: float = 0.5):
    def run(*arrays):
        return (arrays[0] > threshold).astype(np.uint8)
    return run
"""
    np_data = np.array([[0.1, 0.6], [0.4, 0.9]], dtype=np.float32)
    data = xr.DataArray(np_data, dims=("y", "x"))
    out_id = MemberId("out2")
    sample = Sample(members={out_id: Tensor.from_xarray(data)}, stat={}, id=None)

    op = CustomPostprocessing(
        output_id=out_id,
        input_ids=[out_id],
        callable_name="threshold_op",
        source_code=source_code,
        kwargs={"threshold": 0.5},
    )
    op(sample)

    expected = xr.DataArray(np.array([[0, 1], [0, 1]], dtype=np.uint8), dims=("y", "x"))
    xr.testing.assert_equal(expected, sample.members[out_id].data)
