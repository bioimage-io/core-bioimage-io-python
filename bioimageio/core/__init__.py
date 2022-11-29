import json
import pathlib

__version__ = json.loads((pathlib.Path(__file__).parent / "VERSION").read_text())["version"]
assert isinstance(__version__, str)

from .resource_io import (
    export_resource_package,
    load_raw_resource_description,
    load_resource_description,
    save_raw_resource_description,
    serialize_raw_resource_description,
)
from .prediction_pipeline import create_prediction_pipeline
from .prediction import predict_image, predict_images, predict_with_padding, predict_with_tiling
from .resource_tests import check_input_shape, check_output_shape, test_resource
