from pathlib import Path

import numpy

from pybio.spec import utils, load_model


def test_RandomForestClassifierBroadNucleusDataBinarized(cache_path):
    spec_path = (
        Path(__file__).parent
        / "../../../specs/models/sklearnbased/RandomForestClassifierBroadNucleusDataBinarized.model.yaml"
    )
    pybio_model = load_model(str(spec_path), kwargs={"c_indices": [None]}, cache_path=cache_path)
    model = utils.train(pybio_model)

    ipt = [numpy.arange(24).reshape((2, 3, 4))]
    out = model(ipt)
    assert len(out) == 1
    assert out[0].shape == ipt[0].shape
