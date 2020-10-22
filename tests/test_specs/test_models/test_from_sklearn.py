from pathlib import Path

import numpy

from pybio.core.training.classic_fit import classic_fit
from pybio.spec import load_model


def test_RandomForestClassifier(cache_path):
    spec_path = Path(__file__).parent / "../../../specs/models/sklearnbased/RandomForestClassifier.model.yaml"
    pybio_model = load_model(str(spec_path))
    model = classic_fit(pybio_model)

    ipt = [numpy.arange(24).reshape((2, 3, 4))]
    out = model(ipt)
    assert len(out) == 1
    assert out[0].shape == ipt[0].shape
