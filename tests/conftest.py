import pytest
from bioimageio.spec import export_resource_package


@pytest.fixture
def unet2d_nuclei_broad_model():
    url = ("https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/"
           "models/unet2d_nuclei_broad/rdf.yaml")
    cached_path = export_resource_package(url)
    return cached_path


@pytest.fixture
def FruNet_model():
    url = "https://raw.githubusercontent.com/deepimagej/models/master/fru-net_sev_segmentation/model.yaml"
    cached_path = export_resource_package(url)
    return cached_path
