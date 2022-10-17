import numpy as np
import xarray as xr

from bioimageio.core.resource_io import nodes


def test_postprocessing_dtype():
    from bioimageio.core.prediction_pipeline._combined_processing import CombinedProcessing

    shape = [3, 32, 32]
    axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)

    threshold = 0.5
    exp = xr.DataArray(np_data > threshold, dims=axes)

    for dtype in ("float32", "float64", "uint8", "uint16"):
        outputs = [
            nodes.OutputTensor(
                "out1",
                data_type=dtype,
                axes=axes,
                shape=shape,
                postprocessing=[nodes.Postprocessing("binarize", dict(threshold=threshold))],
            )
        ]
        com_proc = CombinedProcessing.from_tensor_specs(outputs)

        sample = {"out1": data}
        com_proc.apply(sample, {})
        res = sample["out1"]
        assert np.dtype(res.dtype) == np.dtype(dtype)
        xr.testing.assert_allclose(res, exp.astype(dtype))
