import pickle
from pathlib import Path

import numpy

from bioimageio.core.datasets.broad_nucleus_data import BroadNucleusDataBinarized
from bioimageio.sklearn.models import RandomForestClassifier
from bioimageio.spec.nodes import Model
from bioimageio.spec.utils import get_instance
from bioimageio.spec.utils.transformers import load_and_resolve_spec


test_input_path = Path(__file__).parent / "../../../specs/models/sklearn/test_input_raw.npy"
test_output_path = Path(__file__).parent / "../../../specs/models/sklearn/test_output_raw.npy"


def classic_fit(spec: Model):
    model: RandomForestClassifier = get_instance(spec)

    dataset = BroadNucleusDataBinarized()
    batch = dataset[(slice(None), slice(100, 110), slice(100, 110))]

    model.fit(batch["x"].reshape((-1, 1)), batch["y"].reshape((-1, 1)))

    write_test_inputs_and_outputs = False
    if write_test_inputs_and_outputs:
        test_input = dataset[(slice(0, 1), slice(100, 200), slice(100, 200))]["x"].reshape((-1, 1))
        numpy.save(test_input_path, test_input)
        test_output = model(test_input)
        numpy.save(test_output_path, test_output)

    return model


def create_rf_weights():
    spec = load_and_resolve_spec(
        Path(__file__).parent / "../../../specs/models/sklearn/RandomForestClassifier.model.yaml"
    )
    model = classic_fit(spec)
    with (Path(__file__).parent / "../../../specs/models/sklearn/rf_v0.pickle").open("wb") as f:
        pickle.dump(model, f)


def restore_rf_from_weights():
    spec = load_and_resolve_spec(
        Path(__file__).parent / "../../../specs/models/sklearn/RandomForestClassifier.model.yaml"
    )
    with spec.weights["pickle"].source.open("br") as f:
        model = pickle.load(f)


if __name__ == "__main__":
    create_rf_weights()
    restore_rf_from_weights()

