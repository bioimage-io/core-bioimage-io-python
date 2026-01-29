"""bioimageio.core --- core functionality for BioImage.IO resources

The main focus on this library is to provide functionality to run prediction with
BioImage.IO models, including standardized pre- and postprocessing operations.
The BioImage.IO models (and other resources) are described by---and can be loaded with---the bioimageio.spec package.

See `predict` and `predict_many` for straight-forward model inference
and `create_prediction_pipeline` for finer control of the inference process.

Other notable bioimageio.core functionalities include:
- Testing BioImage.IO resources beyond format validation, e.g. by generating model outputs from test inputs.
  See `test_model` or for arbitrary resource types `test_description`.
- Extending available model weight formats by converting existing ones, see `add_weights`.
- Creating and manipulating `Sample`s consisting of tensors with associated statistics.
- Computing statistics on datasets (represented as sequences of samples), see `compute_dataset_measures`.
"""
# ruff: noqa: E402

__version__ = "0.9.6"
from loguru import logger

logger.disable("bioimageio.core")

import bioimageio.spec

from . import axis as axis
from . import backends as backends
from . import block_meta as block_meta
from . import cli as cli
from . import commands as commands
from . import common as common
from . import digest_spec as digest_spec
from . import io as io
from . import model_adapters as model_adapters
from . import prediction as prediction
from . import proc_ops as proc_ops
from . import proc_setup as proc_setup
from . import sample as sample
from . import stat_calculators as stat_calculators
from . import stat_measures as stat_measures
from . import tensor as tensor
from . import weight_converters as weight_converters
from ._prediction_pipeline import PredictionPipeline as PredictionPipeline
from ._prediction_pipeline import (
    create_prediction_pipeline as create_prediction_pipeline,
)
from ._resource_tests import enable_determinism as enable_determinism
from ._resource_tests import load_description_and_test as load_description_and_test
from ._resource_tests import test_description as test_description
from ._resource_tests import test_model as test_model
from ._settings import Settings as Settings
from ._settings import settings as settings

# reexports from bioimageio.spec
build_description = bioimageio.spec.build_description
dump_description = bioimageio.spec.dump_description
load_dataset_description = bioimageio.spec.load_dataset_description
load_description = bioimageio.spec.load_description
load_description_and_validate_format_only = (
    bioimageio.spec.load_description_and_validate_format_only
)
load_model_description = bioimageio.spec.load_model_description
save_bioimageio_package = bioimageio.spec.save_bioimageio_package
save_bioimageio_package_as_folder = bioimageio.spec.save_bioimageio_package_as_folder
save_bioimageio_yaml_only = bioimageio.spec.save_bioimageio_yaml_only
validate_format = bioimageio.spec.validate_format
ValidationSummary = bioimageio.spec.ValidationSummary


# reexports from bioimageio.core submodules
add_weights = weight_converters.add_weights
Axis = axis.Axis
AxisId = axis.AxisId
BlockMeta = block_meta.BlockMeta
compute_dataset_measures = stat_calculators.compute_dataset_measures
create_model_adapter = backends.create_model_adapter
MemberId = common.MemberId
predict = prediction.predict
predict_many = prediction.predict_many
Sample = sample.Sample
Stat = stat_measures.Stat
Tensor = tensor.Tensor

# aliases
test_resource = test_description
"""alias of `test_description`"""
load_resource = load_description
"""alias of `load_description`"""
load_model = load_model_description
"""alias of `load_model_description`"""
