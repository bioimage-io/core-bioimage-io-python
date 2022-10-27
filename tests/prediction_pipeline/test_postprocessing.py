import numpy as np
import pytest
import xarray as xr

from bioimageio.core.prediction_pipeline._measure_groups import compute_measures


def test_binarize():
    from bioimageio.core.prediction_pipeline._processing import Binarize

    shape = (3, 32, 32)
    axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)

    threshold = 0.5
    exp = xr.DataArray(np_data > threshold, dims=axes)

    binarize = Binarize("data_name", threshold=threshold)
    res = binarize(data)
    xr.testing.assert_allclose(res, exp)


@pytest.mark.parametrize("axes", [None, tuple("cy"), tuple("cyx"), tuple("x")])
def test_scale_mean_variance(axes):
    from bioimageio.core.prediction_pipeline._processing import ScaleMeanVariance

    shape = (3, 32, 46)
    ipt_axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    ipt_data = xr.DataArray(np_data, dims=ipt_axes)
    ref_data = xr.DataArray((np_data * 2) + 3, dims=ipt_axes)

    scale_mean_variance = ScaleMeanVariance("data_name", reference_tensor="ref_name", axes=axes)
    required = scale_mean_variance.get_required_measures()
    computed = compute_measures(required, sample={"data_name": ipt_data, "ref_name": ref_data})
    scale_mean_variance.set_computed_measures(computed)

    res = scale_mean_variance(ipt_data)
    xr.testing.assert_allclose(res, ref_data)


@pytest.mark.parametrize("axes", [None, tuple("cy"), tuple("y"), tuple("yx")])
def test_scale_mean_variance_per_channel(axes):
    from bioimageio.core.prediction_pipeline._processing import ScaleMeanVariance

    shape = (3, 32, 46)
    ipt_axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    ipt_data = xr.DataArray(np_data, dims=ipt_axes)

    # set different mean, std per channel
    np_ref_data = np.stack([d * i + i for i, d in enumerate(np_data, start=2)])
    print(np_ref_data.shape)
    ref_data = xr.DataArray(np_ref_data, dims=ipt_axes)

    scale_mean_variance = ScaleMeanVariance("data_name", reference_tensor="ref_name", axes=axes)
    required = scale_mean_variance.get_required_measures()
    computed = compute_measures(required, sample={"data_name": ipt_data, "ref_name": ref_data})
    scale_mean_variance.set_computed_measures(computed)

    res = scale_mean_variance(ipt_data)

    if axes is not None and "c" not in axes:
        # mean,std per channel should match exactly
        xr.testing.assert_allclose(res, ref_data)
    else:
        # mean,std across channels should not match
        with pytest.raises(AssertionError):
            xr.testing.assert_allclose(res, ref_data)
