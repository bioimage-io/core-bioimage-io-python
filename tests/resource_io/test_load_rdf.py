import os.path
import pathlib
from pathlib import Path

import pytest


def test_load_model_with_abs_path_source(unet2d_nuclei_broad_model: Path):
    from bioimageio.spec import load_description

    raw_rd = load_description(unet2d_nuclei_broad_model)
    path_source = (raw_rd.root / "rdf.yaml").absolute()
    assert path_source.is_absolute()
    model = load_description(path_source)
    assert model


def test_load_model_with_rel_path_source(unet2d_nuclei_broad_model: Path):
    from bioimageio.core.resource_io import load_description, load_raw_resource_description

    raw_rd = load_raw_resource_description(unet2d_nuclei_broad_model)
    path_source = pathlib.Path(os.path.relpath(raw_rd.root_path / "rdf.yaml", os.curdir))
    assert not path_source.is_absolute()
    model = load_description(path_source)
    assert model


def test_load_model_with_abs_str_source(unet2d_nuclei_broad_model: Path):
    from bioimageio.core.resource_io import load_description, load_raw_resource_description

    raw_rd = load_raw_resource_description(unet2d_nuclei_broad_model)
    path_source = (raw_rd.root_path / "rdf.yaml").absolute()
    assert path_source.is_absolute()
    model = load_description(str(path_source))
    assert model


def test_load_model_with_rel_str_source(unet2d_nuclei_broad_model: Path):
    from bioimageio.core.resource_io import load_description, load_raw_resource_description

    raw_rd = load_raw_resource_description(unet2d_nuclei_broad_model)
    path_source = pathlib.Path(os.path.relpath(raw_rd.root_path / "rdf.yaml", os.curdir))
    assert not path_source.is_absolute()
    model = load_description(str(path_source))
    assert model


def test_load_remote_rdf(unet2d_nuclei_broad_model: Path):
    # remote model is a pytorch model, needing unet2d_nuclei_broad_model skips the test when needed
    _ = unet2d_nuclei_broad_model
    from bioimageio.core import load_description
    from bioimageio.core.resource_io import nodes

    remote_rdf = "https://zenodo.org/api/files/63b44f05-a187-4fc9-81c8-c4568535531b/rdf.yaml"
    model = load_description(remote_rdf)
    assert isinstance(model, nodes.Model)


@pytest.mark.skipif(True, reason="No suitable test model available yet")
def test_load_remote_rdf_with_folders():
    from bioimageio.core import load_description, load_raw_resource_description
    from bioimageio.core.resource_io import nodes
    from bioimageio.spec.model import raw_nodes

    rdf_doi = "<doi to rdf with local folders>"
    raw_model = load_raw_resource_description(rdf_doi, update_to_format="latest")
    assert isinstance(raw_model, raw_nodes.Model)
    model = load_description(rdf_doi)
    assert isinstance(model, nodes.Model)

    # test for field value with folder, e.g.
    assert resolve_source(raw_model.documentation) == model.documentation
