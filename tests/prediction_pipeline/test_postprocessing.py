import numpy as np
import xarray as xr
from bioimageio.core.resource_io.nodes import Postprocessing
from bioimageio.core.prediction_pipeline._combined_processing import make_postprocessing


def test_binarize_postprocessing():
    shape = (3, 32, 32)
    axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)

    threshold = 0.5
    exp = xr.DataArray(np_data > threshold, dims=axes)

    for dtype in ("float32", "float64", "uint8", "uint16"):
        binarize = make_postprocessing(spec=[Postprocessing("binarize", kwargs={"threshold": threshold})], dtype=dtype)
        res = binarize(data)
        assert np.dtype(res.dtype) == np.dtype(dtype)
        xr.testing.assert_allclose(res, exp.astype(dtype))
