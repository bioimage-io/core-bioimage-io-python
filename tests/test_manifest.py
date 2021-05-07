from pathlib import Path

import pytest
import yaml

from bioimageio.spec import load_spec, utils

MANIFEST_PATH = Path(__file__).parent / "../manifest.yaml"


def pytest_generate_tests(metafunc):
    if "category" in metafunc.fixturenames and "spec_path" in metafunc.fixturenames:
        with MANIFEST_PATH.open() as f:
            manifest = yaml.safe_load(f)

        categories_and_spec_paths = [
            (category, spec_path) for category, spec_paths in manifest.items() for spec_path in spec_paths
        ]
        metafunc.parametrize("category,spec_path", categories_and_spec_paths)


@pytest.fixture
def required_spec_kwargs():
    kwargs = yaml.safe_load(
        """
specs/models/sklearnbased/RandomForestClassifierBroadNucleusDataBinarized.model.yaml:
    kwargs:
        channel_indices: [1]
    """
    )

    # testing the test data...
    for spec_path in kwargs:
        if not (MANIFEST_PATH.parent / spec_path).exists():
            raise FileNotFoundError(MANIFEST_PATH.parent / spec_path)

    return kwargs


def test_load_specs_from_manifest(category, spec_path, required_spec_kwargs):
    spec_path = MANIFEST_PATH.parent / spec_path
    assert spec_path.exists()

    loaded_spec = load_spec(str(spec_path))
    instance = utils.get_instance(loaded_spec)
    assert instance
