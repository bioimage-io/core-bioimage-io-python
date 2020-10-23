from pathlib import Path

import pytest

from pybio.spec import load_model_config


@pytest.fixture
def rf_config():
    return load_model_config(Path(__file__) / "../../specs/models/sklearn/RandomForestClassifier.model.yaml").spec
