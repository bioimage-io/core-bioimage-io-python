import numpy as np
import xarray as xr


def test_binarize_postprocessing():
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
