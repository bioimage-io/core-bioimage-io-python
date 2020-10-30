from pathlib import Path

import pytest
import yaml

from pybio.spec import load_model_config


@pytest.fixture
def rf_config_path_v0_1():
    return Path(__file__) / "../../specs/models/sklearn/RandomForestClassifier_v0_1.model.yaml"


@pytest.fixture
def rf_config_path():
    return Path(__file__) / "../../specs/models/sklearn/RandomForestClassifier.model.yaml"


@pytest.fixture
def rf_config(rf_config_path):
    return load_model_config(rf_config_path).spec


@pytest.fixture
def rf_config_v0_1(rf_config_path_v0_1):
    return load_model_config(rf_config_path_v0_1).spec


@pytest.fixture
def rf_model_data_v0_1(rf_config_path_v0_1):
    with rf_config_path_v0_1.open() as f:
        return yaml.safe_load(f)


@pytest.fixture
def rf_model_data(rf_config_path):
    with rf_config_path.open() as f:
        return yaml.safe_load(f)
