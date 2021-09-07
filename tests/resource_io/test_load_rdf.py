import pathlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from marshmallow import ValidationError

from bioimageio.core.resource_io.utils import resolve_uri


def test_load_non_existing_rdf():
    from bioimageio.core import load_resource_description

    spec_path = Path("some/none/existing/path/to/spec.model.yaml")

    with pytest.raises(FileNotFoundError):
        load_resource_description(spec_path)


def test_load_non_valid_rdf_name_no_suffix():
    from bioimageio.core import load_resource_description

    with NamedTemporaryFile() as f:
        spec_path = pathlib.Path(f.name)

        with pytest.raises(ValidationError):
            load_resource_description(spec_path)


def test_load_non_valid_rdf_name_invalid_suffix():
    from bioimageio.core import load_resource_description

    with NamedTemporaryFile(suffix=".invalid_suffix") as f:
        spec_path = pathlib.Path(f.name)

        with pytest.raises(ValidationError):
            load_resource_description(spec_path)


def test_load_raw_model(unet2d_nuclei_broad_model):
    from bioimageio.core import load_raw_resource_description

    raw_model = load_raw_resource_description(unet2d_nuclei_broad_model)
    assert raw_model


def test_load_raw_model_from_package(unet2d_nuclei_broad_model):
    from bioimageio.core import load_raw_resource_description

    raw_model = load_raw_resource_description(unet2d_nuclei_broad_model)
    assert raw_model


def test_load_model(unet2d_nuclei_broad_model):
    from bioimageio.core import load_resource_description

    model = load_resource_description(unet2d_nuclei_broad_model)
    assert model


def test_load_model_from_package(unet2d_nuclei_broad_model):
    from bioimageio.core import load_resource_description

    model = load_resource_description(unet2d_nuclei_broad_model)
    assert model


def test_load_remote_model_with_folders():
    from bioimageio.core import load_resource_description, load_raw_resource_description
    from bioimageio.core.resource_io import nodes
    from bioimageio.spec.model import raw_nodes

    # todo: point to real model with nested folders, not this temporary sandbox one
    rdf_url = "https://sandbox.zenodo.org/record/892199/files/rdf.yaml"
    raw_model = load_raw_resource_description(rdf_url)
    assert isinstance(raw_model, raw_nodes.Model)
    model = load_resource_description(rdf_url)
    assert isinstance(model, nodes.Model)
    assert resolve_uri(raw_model.documentation) == model.documentation