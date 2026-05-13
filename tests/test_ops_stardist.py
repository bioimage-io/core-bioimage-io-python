# type: ignore
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["2D_versatile_fluo"])
def test_stardist_export(name: str, tmp_path: Path):
    """test case analog to the example in example/export_stardist_model"""
    try:
        import stardist.data
        from stardist.models import StarDist2D
    except ImportError:
        pytest.mark.skip("compatible stardist version not installed")

    from bioimageio.core import test_model

    model = StarDist2D.from_pretrained(name)

    img = stardist.data.test_image_nuclei_2d()
    img_axes = "yx"

    output_path = stardist.export_bioimageio(
        model, tmp_path / f"stardist_bioimageio_{name}.zip", img, img_axes
    )

    s = test_model(output_path)
    assert s.status == "passed", s.display()
