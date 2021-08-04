import numpy as np
import pytest
import xarray as xr
from bioimageio.spec.model.nodes import Preprocessing

from bioimageio.core.prediction_pipeline._preprocessing import make_preprocessing


def test_scale_linear():
    spec = Preprocessing(name="scale_linear", kwargs={"offset": [1, 2, 42], "gain": [1, 2, 3], "axes": "yx"})
    data = xr.DataArray(np.arange(6).reshape(1, 2, 3), dims=("x", "y", "c"))
    expected = xr.DataArray(np.array([[[1, 4, 48], [4, 10, 57]]]), dims=("x", "y", "c"))
    preprocessing = make_preprocessing([spec])
    result = preprocessing(data)
    xr.testing.assert_allclose(expected, result)


def test_scale_linear_no_channel():
    spec = Preprocessing(name="scale_linear", kwargs={"offset": 1, "gain": 2, "axes": "yx"})
    data = xr.DataArray(np.arange(6).reshape(2, 3), dims=("x", "y"))
    expected = xr.DataArray(np.array([[1, 3, 5], [7, 9, 11]]), dims=("x", "y"))
    preprocessing = make_preprocessing([spec])
    result = preprocessing(data)
    xr.testing.assert_allclose(expected, result)


def test_zero_mean_unit_variance_preprocessing():
    zero_mean_spec = Preprocessing(name="zero_mean_unit_variance", kwargs={})
    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
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
    preprocessing = make_preprocessing([zero_mean_spec])
    result = preprocessing(data)
    xr.testing.assert_allclose(expected, result)


def test_zero_mean_unit_across_axes():
    zero_mean_spec = Preprocessing(name="zero_mean_unit_variance", kwargs={"axes": ("x", "y")})
    data = xr.DataArray(np.arange(18).reshape(2, 3, 3), dims=("c", "x", "y"))
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
    preprocessing = make_preprocessing([zero_mean_spec])
    result = preprocessing(data)
    xr.testing.assert_allclose(expected, result[dict(c=0)])


def test_binarize():
    binarize_spec = Preprocessing(name="binarize", kwargs={"threshold": 14})
    data = xr.DataArray(np.arange(30).reshape(2, 3, 5), dims=("x", "y", "c"))
    expected = xr.zeros_like(data)
    expected[{"x": slice(1, None)}] = 1
    preprocessing = make_preprocessing([binarize_spec])
    result = preprocessing(data)
    xr.testing.assert_allclose(expected, result)


def test_clip_preprocessing():
    clip_spec = Preprocessing(name="clip", kwargs={"min": 3, "max": 5})
    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    expected = xr.DataArray(np.array([[3, 3, 3], [3, 4, 5], [5, 5, 5]]), dims=("x", "y"))
    preprocessing = make_preprocessing([clip_spec])
    result = preprocessing(data)
    xr.testing.assert_equal(expected, result)


def test_unknown_preprocessing_should_raise():
    mypreprocessing = Preprocessing(name="mycoolpreprocessing", kwargs={"axes": ("x", "y")})
    with pytest.raises(NotImplementedError):
        make_preprocessing([mypreprocessing])


def test_combination_of_preprocessing_steps_with_dims_specified():
    zero_mean_spec = Preprocessing(name="zero_mean_unit_variance", kwargs={"axes": ("x", "y")})
    data = xr.DataArray(np.arange(18).reshape(2, 3, 3), dims=("c", "x", "y"))

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

    preprocessing = make_preprocessing([zero_mean_spec])
    result = preprocessing(data)
    xr.testing.assert_allclose(expected, result[dict(c=0)])
