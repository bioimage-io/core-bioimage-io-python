from pathlib import Path

import numpy

from pybio.spec import utils, load_model


def test_RandomForestClassifierBroadNucleusDataBinarized():
    spec_path = (
        Path(__file__).parent
        / "../../../../specs/models/sklearnbased/RandomForestClassifierBroadNucleusDataBinarized.model.yaml"
    )
    pybio_model = load_model(spec_path.as_posix(), kwargs={"c_indices": [None]})
    model = utils.train(pybio_model)

    ipt = [numpy.arange(24).reshape((2, 3, 4))]
    out = model(ipt)
    assert len(out) == 1
    assert out[0].shape == ipt[0].shape
