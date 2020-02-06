import numpy
import pytest

from pybio.core.transformations.normalizations import NormalizeZeroMeanUnitVariance


normalize_testdata = [
    [[0], [numpy.arange(12)], [3], [4], [(numpy.arange(12) - 3) / 4]],
    [None, [numpy.arange(12)], [3], [4], [(numpy.arange(12) - 3) / 4]],
    [[0], [numpy.arange(12), numpy.arange(3)], [3], [4], [(numpy.arange(12) - 3) / 4, numpy.arange(3)]],
    [[1], [numpy.arange(12), numpy.arange(3)], [3], [4], [numpy.arange(12), (numpy.arange(3) - 3) / 4]],
]


@pytest.mark.parametrize("apply_to,ipt,means,stds,expected_out_list", normalize_testdata)
def test_NormalizeZeroMeanUnitVariance_1array(apply_to, ipt, means, stds, expected_out_list):
    trf = NormalizeZeroMeanUnitVariance(apply_to=apply_to, means=means, stds=stds)
    out_list = trf.apply(*ipt)
    assert len(out_list) == len(expected_out_list)
    for out, expected_out in zip(out_list, expected_out_list):
        assert numpy.isclose(out, expected_out).all()


def test_NormalizeZeroMeanUnitVariance_no_kwargs_1array():
    trf = NormalizeZeroMeanUnitVariance()

    ipt = numpy.arange(12).reshape((3, 4))
    out = trf.apply(ipt)[0]
    assert out.shape == (3, 4)
    expected = numpy.array(
        [
            [-1.59325455, -1.30357191, -1.01388926, -0.72420661],
            [-0.43452397, -0.14484132, 0.14484132, 0.43452397],
            [0.72420661, 1.01388926, 1.30357191, 1.59325455],
        ]
    )
    assert numpy.isclose(out, expected).all()


def test_NormalizeZeroMeanUnitVariance_no_kwargs_2arrays():
    trf = NormalizeZeroMeanUnitVariance()

    ipt = [numpy.arange(12).reshape((3, 4)), numpy.arange(12, 24).reshape((3, 4))]
    out = trf.apply(*ipt)
    assert out[0].shape == (3, 4)
    assert out[1].shape == (3, 4)
    assert numpy.isclose(
        out[0],
        numpy.array(
            [
                [-1.59325455, -1.30357191, -1.01388926, -0.72420661],
                [-0.43452397, -0.14484132, 0.14484132, 0.43452397],
                [0.72420661, 1.01388926, 1.30357191, 1.59325455],
            ]
        ),
    ).all()
    assert numpy.equal(out[1], numpy.arange(12, 24).reshape((3, 4))).all()
