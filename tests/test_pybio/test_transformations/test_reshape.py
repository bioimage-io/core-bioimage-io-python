import numpy
import pytest

from pybio.transformations.reshape import Reshape

testdata = [
    ((12,), (3, 4), (12,)),
    ((-1,), (3, 4), (12,)),
    ((-2, -2), (3, 4), (3, 4)),
    ((-2, -1), (3, 4), (3, 4)),
    ((-2, 6, 1, -2), (1, 2, 3, 4), (1, 6, 1, 4)),
    ((-2, 6, -1), (1, 2, 3, 4), (1, 6, 4)),
    ((6, -1), (1, 2, 3, 4), (6, 4)),
    ((-2, 1, 3), (4, 3), (4, 1, 3)),
]


@pytest.mark.parametrize("shape,ipt_shape,out_shape", testdata)
def test_reshape(shape, ipt_shape, out_shape):
    trf = Reshape(shape=shape)
    ipt = numpy.empty(ipt_shape)
    out = trf.apply([ipt])[0]
    assert out.shape == out_shape


testdata_value_error_on_call = [
    ((-1, -1), (3, 4)),
    ((-1, 6, -1), (1, 2, 3, 4)),
    ((3, 4, -2), (3, 4)),
    ((1, 1, -2), (2, 3)),
]


@pytest.mark.parametrize("shape,ipt_shape", testdata_value_error_on_call)
def test_reshape_raises_value_error_on_call(shape, ipt_shape):
    trf = Reshape(shape=shape)
    ipt = numpy.empty(ipt_shape)
    with pytest.raises(ValueError):
        trf.apply([ipt])
