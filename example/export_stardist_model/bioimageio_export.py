"""Run original cellpose model and save an analog input and output for bioimageio tests"""

from pathlib import Path

import stardist.data
from stardist.models import StarDist2D

from bioimageio.core import test_model

if __name__ == "__main__":
    name = "2D_versatile_fluo"
    model = StarDist2D.from_pretrained(name)

    img = stardist.data.test_image_nuclei_2d()
    img_axes = "yx"

    output_path = stardist.export_bioimageio(
        model, Path(f"output/stardist_bioimageio_{name}.zip"), img, img_axes
    )

    s = test_model(output_path)
    s.display()
