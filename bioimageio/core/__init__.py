import json

from bioimageio.core.utils import files

with files("bioimageio.core").joinpath("VERSION").open("r", encoding="utf-8") as f:
    __version__: str = json.load(f)["version"]
    assert isinstance(__version__, str)

from bioimageio.spec import build_description as build_description
from bioimageio.spec import dump_description as dump_description
from bioimageio.spec import load_description as load_description
from bioimageio.spec import load_description_and_validate_format_only as load_description_and_validate_format_only
from bioimageio.spec import save_bioimageio_package as save_bioimageio_package
from bioimageio.spec import save_bioimageio_package_as_folder as save_bioimageio_package_as_folder
from bioimageio.spec import save_bioimageio_yaml_only as save_bioimageio_yaml_only
from bioimageio.spec import validate_format as validate_format

from ._prediction_pipeline import create_prediction_pipeline as create_prediction_pipeline
from ._resource_tests import load_description_and_test as load_description_and_test
from ._resource_tests import test_description as test_description
from ._resource_tests import test_model as test_model

test_resource = test_description
