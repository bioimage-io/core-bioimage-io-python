import json

from bioimageio.core.io import load_description as load_description
from bioimageio.core.io import load_description_and_validate as load_description_and_validate
from bioimageio.core.io import read_description as read_description
from bioimageio.core.io import read_description_and_validate as read_description_and_validate
from bioimageio.core.io import resolve_source as resolve_source
from bioimageio.core.io import write_description as write_description
from bioimageio.core.io import write_package as write_package
from bioimageio.core.io import write_package_as_folder as write_package_as_folder
from bioimageio.core.utils import files

with files("bioimageio.core").joinpath("VERSION").open("r", encoding="utf-8") as f:
    __version__: str = json.load(f)["version"]
    assert isinstance(__version__, str)

# from .prediction import predict_image, predict_images, predict_with_padding, predict_with_tiling
from .prediction_pipeline import create_prediction_pipeline

# from .resource_tests import check_input_shape, check_output_shape, test_resource
