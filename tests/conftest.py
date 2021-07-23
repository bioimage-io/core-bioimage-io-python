import pytest


@pytest.fixture
def unet2d_nuclei_broad_model_url():
    return "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/unet2d_nuclei_broad/rdf.yaml"


@pytest.fixture
def FruNet_model_url():
    return "https://raw.githubusercontent.com/deepimagej/models/master/fru-net_sev_segmentation/model.yaml"
