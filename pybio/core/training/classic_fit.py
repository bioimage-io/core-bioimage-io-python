from os import getenv
from pathlib import Path

from pybio.core.datasets.broad_nucleus_data import BroadNucleusDataBinarized
from pybio.core.models.sklearnbased import RandomForestClassifier
from pybio.spec.nodes import Model
from pybio.spec.utils import get_instance, load_model_config


def classic_fit(pybio_model: Model):
    model: RandomForestClassifier = get_instance(pybio_model)

    dataset = BroadNucleusDataBinarized(cache_path=Path(getenv("PYBIO_CACHE_PATH", "pybio_cache")))
    roi = (slice(None), slice(100, 200), slice(100, 200))
    X, y = dataset[roi, roi]

    model.fit([X], [y])
    return model
    # return model
    # # todo: save/return model weights/checkpoint?!?


def train_rf():
    pybio_model = load_model_config(
        str((Path(__file__).parent / "../../../specs/models/sklearnbased/RandomForestClassifier.model.yaml").resolve())
    )
    rf = classic_fit(pybio_model)
    weight = rf.get_weights()
    Path("/repos/python-bioimage-io/rf_v0.pickle").write_bytes(weight)


def load_rf_weight_from_weights():
    pybio_model = load_model_config(
        str((Path(__file__).parent / "../../../specs/models/sklearnbased/RandomForestClassifier.model.yaml").resolve())
    )
    p: Path = pybio_model.spec.weights[0].source  # noqa
    with p.open("rb") as f:
        rf.set_weights(f.read())

    return rf


if __name__ == "__main__":
    rf = load_rf_weight()
