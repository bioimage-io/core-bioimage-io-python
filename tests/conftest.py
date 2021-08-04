import pytest
from bioimageio.spec import export_resource_package


# set 'skip_<FRAMEWORK>' flags as global pytest variables,
# to deselect tests that require frameworks not available in current env
def pytest_configure():
    try:
        import tensorflow
    except ImportError:
        tensorflow = None
    pytest.skip_tf = tensorflow is None

    try:
        import torch
    except ImportError:
        torch = None
    pytest.skip_torch = torch is None

    try:
        import onnxruntime
    except ImportError:
        onnxruntime = None
    pytest.skip_onnx = onnxruntime is None


@pytest.fixture
def unet2d_nuclei_broad_model():
    url = (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/"
        "models/unet2d_nuclei_broad/rdf.yaml"
    )
    cached_path = export_resource_package(url)
    return cached_path


@pytest.fixture
def FruNet_model():
    url = "https://raw.githubusercontent.com/deepimagej/models/master/fru-net_sev_segmentation/model.yaml"
    cached_path = export_resource_package(url)
    return cached_path
