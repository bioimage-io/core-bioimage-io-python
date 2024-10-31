"""
.. include:: ../../README.md
"""

from bioimageio.core.stat_measures import Stat
from bioimageio.spec import (
    build_description,
    dump_description,
    load_dataset_description,
    load_description,
    load_description_and_validate_format_only,
    load_model_description,
    save_bioimageio_package,
    save_bioimageio_package_as_folder,
    save_bioimageio_yaml_only,
    validate_format,
)

from . import digest_spec
from ._prediction_pipeline import PredictionPipeline, create_prediction_pipeline
from ._resource_tests import load_description_and_test, test_description, test_model
from ._settings import settings
from .axis import Axis, AxisId
from .block_meta import BlockMeta
from .common import MemberId
from .prediction import predict, predict_many
from .sample import Sample
from .stat_calculators import compute_dataset_measures
from .tensor import Tensor
from .utils import VERSION

__version__ = VERSION


# aliases
test_resource = test_description
load_resource = load_description
load_model = load_model_description

__all__ = [
    "__version__",
    "Axis",
    "AxisId",
    "BlockMeta",
    "build_description",
    "compute_dataset_measures",
    "create_prediction_pipeline",
    "digest_spec",
    "dump_description",
    "load_dataset_description",
    "load_description_and_test",
    "load_description_and_validate_format_only",
    "load_description",
    "load_model_description",
    "load_model",
    "load_resource",
    "MemberId",
    "predict_many",
    "predict",
    "PredictionPipeline",
    "Sample",
    "save_bioimageio_package_as_folder",
    "save_bioimageio_package",
    "save_bioimageio_yaml_only",
    "settings",
    "Stat",
    "Tensor",
    "test_description",
    "test_model",
    "test_resource",
    "validate_format",
]
