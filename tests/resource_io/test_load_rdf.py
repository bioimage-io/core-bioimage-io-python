import os.path
import pathlib
from pathlib import Path

import pytest

from bioimageio.core.resource_io.utils import resolve_source


def test_load_non_existing_rdf():
    from bioimageio.core import load_resource_description

    spec_path = Path("some/none/existing/path/to/spec.model.yaml")

    with pytest.raises(FileNotFoundError):
        load_resource_description(spec_path)


def test_load_raw_model(any_model):
    from bioimageio.core import load_raw_resource_description

    raw_model = load_raw_resource_description(any_model)
    assert raw_model


def test_load_model(any_model):
    from bioimageio.core import load_resource_description

    model = load_resource_description(any_model)
    assert model


def test_load_model_with_abs_path_source(unet2d_nuclei_broad_model):
    from bioimageio.core.resource_io import load_raw_resource_description, load_resource_description

    raw_rd = load_raw_resource_description(unet2d_nuclei_broad_model)
    path_source = (raw_rd.root_path / "rdf.yaml").absolute()
    assert path_source.is_absolute()
    model = load_resource_description(path_source)
    assert model


def test_load_model_with_rel_path_source(unet2d_nuclei_broad_model):
    from bioimageio.core.resource_io import load_raw_resource_description, load_resource_description

    raw_rd = load_raw_resource_description(unet2d_nuclei_broad_model)
    path_source = pathlib.Path(os.path.relpath(raw_rd.root_path / "rdf.yaml", os.curdir))
    assert not path_source.is_absolute()
    model = load_resource_description(path_source)
    assert model


def test_load_model_with_abs_str_source(unet2d_nuclei_broad_model):
    from bioimageio.core.resource_io import load_raw_resource_description, load_resource_description

    raw_rd = load_raw_resource_description(unet2d_nuclei_broad_model)
    path_source = (raw_rd.root_path / "rdf.yaml").absolute()
    assert path_source.is_absolute()
    model = load_resource_description(str(path_source))
    assert model


def test_load_model_with_rel_str_source(unet2d_nuclei_broad_model):
    from bioimageio.core.resource_io import load_raw_resource_description, load_resource_description

    raw_rd = load_raw_resource_description(unet2d_nuclei_broad_model)
    path_source = pathlib.Path(os.path.relpath(raw_rd.root_path / "rdf.yaml", os.curdir))
    assert not path_source.is_absolute()
    model = load_resource_description(str(path_source))
    assert model


@pytest.mark.skipif(pytest.skip_torch, reason="requires pytorch")
def test_load_remote_model_with_folders():
    from bioimageio.core import load_resource_description, load_raw_resource_description
    from bioimageio.core.resource_io import nodes
    from bioimageio.spec.model import raw_nodes

    # todo: point to real model with nested folders, not this temporary sandbox one
    rdf_url = "https://sandbox.zenodo.org/record/892199/files/rdf.yaml"
    raw_model = load_raw_resource_description(rdf_url, update_to_format="latest")
    assert isinstance(raw_model, raw_nodes.Model)
    model = load_resource_description(rdf_url)
    assert isinstance(model, nodes.Model)
    assert resolve_source(raw_model.documentation) == model.documentation
