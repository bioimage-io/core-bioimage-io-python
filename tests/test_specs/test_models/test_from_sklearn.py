from pathlib import Path

import numpy

from pybio.sklearn.training.classic_fit import classic_fit
from pybio.spec.utils.transformers import load_and_resolve_spec


def test_RandomForestClassifier():
    spec_path = Path(__file__).parent / "../../../specs/models/sklearn/RandomForestClassifier.model.yaml"
    spec = load_and_resolve_spec(spec_path)
    model = classic_fit(spec)

    ipt = numpy.arange(24)[:, None]
    out = model(ipt)
    assert isinstance(out, numpy.ndarray)
    assert out.shape == ipt.shape
