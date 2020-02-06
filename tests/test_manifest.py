from pathlib import Path

import pytest
import yaml

from pybio.spec import load_spec_and_kwargs, utils

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
def required_kwargs():
    local_pybio_path = Path(__file__).parent.parent
    kwargs = yaml.safe_load(
        f"""
specs/transformations/Reshape.transformation.yaml:
    shape: [-1]
specs/models/sklearnbased/RandomForestClassifierBroadNucleusDataBinarized.model.yaml: 
    c_indices: [1]
specs/samplers/SequentialSamplerAlongDimension.sampler.yaml:
    readers: [uri: {str(local_pybio_path / "specs/readers/BroadNucleusDataBinarized.reader.yaml")}]    
    sample_dimensions: [0, 0]
specs/transformations/Cast.transformation.yaml:
    dtype: float32
specs/readers/SimpleConcatenatedReader.reader.yaml:
    # readers: # todo: local relative paths for uri in kwargs
    #     - Reader: {{spec: ../specs/readers/BroadNucleusDataBinarized.reader.yaml)}}
    readers: [uri: {str(local_pybio_path / "specs/readers/BroadNucleusDataBinarized.reader.yaml")}]    
    dims: [0, 0]
    """
    )

    # testing the test data...
    for spec_path in kwargs:
        if not (MANIFEST_PATH.parent / spec_path).exists():
            raise FileNotFoundError(MANIFEST_PATH.parent / spec_path)

    return kwargs


def test_load_specs_from_manifest(category, spec_path, required_kwargs):
    kwargs = required_kwargs.get(spec_path, {})

    spec_path = MANIFEST_PATH.parent / spec_path
    assert spec_path.exists()

    loaded_spec = load_spec_and_kwargs(str(spec_path), kwargs=kwargs)
    instance = utils.get_instance(loaded_spec)
    assert instance
