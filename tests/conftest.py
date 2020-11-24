from pathlib import Path

import pytest
from ruamel.yaml import YAML

from pybio.spec import load_spec

yaml = YAML(typ="safe")


@pytest.fixture
def rf_config_path_v0_1():
    return Path(__file__) / "../../specs/models/sklearn/RandomForestClassifier_v0_1.model.yaml"


@pytest.fixture
def rf_config_path():
    return Path(__file__) / "../../specs/models/sklearn/RandomForestClassifier.model.yaml"


@pytest.fixture
def rf_config(rf_config_path):
    return load_spec(rf_config_path)


@pytest.fixture
def rf_config_v0_1(rf_config_path_v0_1):
    return load_spec(rf_config_path_v0_1)


@pytest.fixture
def rf_model_data_v0_1(rf_config_path_v0_1):
    return yaml.load(rf_config_path_v0_1)


@pytest.fixture
def rf_model_data(rf_config_path):
    return yaml.load(rf_config_path)
