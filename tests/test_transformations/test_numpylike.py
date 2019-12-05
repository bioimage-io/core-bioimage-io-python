import numpy
import pytest

from pybio.transformations.numpylike import Reshape, Transpose


reshape_testdata = [((12,), (3, 4), (12,)), ((-1,), (3, 4), (12,)), ((6, -1), (1, 2, 3, 4), (6, 4))]


@pytest.mark.parametrize("newshape,ipt_shape,out_shape", reshape_testdata)
def test_reshape(newshape, ipt_shape, out_shape):
    trf = Reshape(newshape=newshape)
    ipt = numpy.empty(ipt_shape)
    out = trf([ipt])[0]
    assert out.shape == out_shape


transpose_testdata = [(None, (3, 4), (4, 3)), ((1, 0), (3, 4), (4, 3)), ((0, 1), (3, 4), (3, 4))]


@pytest.mark.parametrize("axes,ipt_shape,out_shape", transpose_testdata)
def test_transpose(axes, ipt_shape, out_shape):
    trf = Transpose(axes=axes)
    ipt = numpy.empty(ipt_shape)
    out = trf([ipt])[0]
    assert out.shape == out_shape


def test_transpose_no_kwargs():
    trf = Transpose()

    ipt = numpy.empty((3, 4))
    out = trf([ipt])[0]
    assert out.shape == (4, 3)
