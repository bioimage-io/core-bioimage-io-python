from pathlib import Path

import numpy

from pybio import load_spec


def test_RandomForestClassifier():
    spec_path = Path(__file__).parent / "../../specs/models/sklearnbased/RandomForestClassifier0.model.yaml"
    loaded_spec = load_spec(spec_path.as_posix(), kwargs={"c_indices": [None]})

    # model = loaded_spec.get_instance()
    model = loaded_spec.train()

    # model.train()
    ipt = [numpy.arange(24).reshape((2, 3, 4))]
    out = model(ipt)
    assert len(out) == 1
    assert out[0].shape == ipt[0].shape
