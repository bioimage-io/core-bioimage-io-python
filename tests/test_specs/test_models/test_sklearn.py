import numpy

from bioimageio.sklearn.training.classic_fit import classic_fit
from bioimageio.spec.utils.transformers import load_and_resolve_spec


def test_RandomForestClassifier(rf_config_path):
    spec = load_and_resolve_spec(rf_config_path)
    model = classic_fit(spec)

    ipt = numpy.arange(24)[:, None]
    out = model(ipt)
    assert isinstance(out, numpy.ndarray)
    assert out.shape == ipt.shape
