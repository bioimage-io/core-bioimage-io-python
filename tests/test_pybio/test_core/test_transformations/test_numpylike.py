import numpy
import pytest

from pybio.core.transformations.numpylike import Transpose


transpose_testdata = [(None, (3, 4), (4, 3)), ((1, 0), (3, 4), (4, 3)), ((0, 1), (3, 4), (3, 4))]


@pytest.mark.parametrize("axes,ipt_shape,out_shape", transpose_testdata)
def test_transpose(axes, ipt_shape, out_shape):
    trf = Transpose(axes=axes)
    ipt = numpy.empty(ipt_shape)
    out = trf.apply(ipt)[0]
    assert out.shape == out_shape


def test_transpose_no_kwargs():
    trf = Transpose()

    ipt = numpy.empty((3, 4))
    out = trf.apply(ipt)[0]
    assert out.shape == (4, 3)
