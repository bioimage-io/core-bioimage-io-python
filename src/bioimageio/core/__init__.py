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

__version__ = "0.11.0"
from loguru import logger

logger.disable("bioimageio.core")


from bioimageio.spec import ValidationSummary as ValidationSummary
from bioimageio.spec import build_description as build_description
from bioimageio.spec import dump_description as dump_description
from bioimageio.spec import load_dataset_description as load_dataset_description
from bioimageio.spec import load_description as load_description
from bioimageio.spec import (
    load_description_and_validate_format_only as load_description_and_validate_format_only,
)
from bioimageio.spec import load_model_description as load_model_description
from bioimageio.spec import save_bioimageio_package as save_bioimageio_package
from bioimageio.spec import (
    save_bioimageio_package_as_folder as save_bioimageio_package_as_folder,
)
from bioimageio.spec import save_bioimageio_yaml_only as save_bioimageio_yaml_only
from bioimageio.spec import validate_format as validate_format

from . import axis as axis
from . import backends as backends
from . import block_meta as block_meta
from . import cli as cli
from . import commands as commands
from . import common as common
from . import digest_spec as digest_spec
from . import io as io
from . import prediction as prediction
from . import proc_ops as proc_ops
from . import proc_setup as proc_setup
from . import sample as sample
from . import stat_calculators as stat_calculators
from . import stat_measures as stat_measures
from . import tensor as tensor
from . import weight_converters as weight_converters
from ._prediction_pipeline import IntermediatePrediction as IntermediatePrediction
from ._prediction_pipeline import PredictionPipeline as PredictionPipeline
from ._prediction_pipeline import RemotePredictionPipeline as RemotePredictionPipeline
from ._prediction_pipeline import (
    create_prediction_pipeline as create_prediction_pipeline,
)
from ._prediction_pipeline import (
    create_remote_prediction_pipeline as create_remote_prediction_pipeline,
)
from ._resource_tests import enable_determinism as enable_determinism
from ._resource_tests import load_description_and_test as load_description_and_test
from ._resource_tests import test_description as test_description
from ._resource_tests import test_model as test_model
from ._sample_serializer import SampleSerializer as SampleSerializer
from ._settings import Settings as Settings
from ._settings import settings as settings

# reexports from bioimageio.core submodules
from .axis import Axis as Axis
from .axis import AxisId as AxisId
from .backends import create_model_adapter as create_model_adapter
from .block_meta import BlockMeta as BlockMeta
from .common import MemberId as MemberId
from .prediction import predict as predict
from .prediction import predict_many as predict_many
from .sample import Sample as Sample
from .sample import SampleBlock as SampleBlock
from .sample import SampleBlockMeta as SampleBlockMeta
from .stat_calculators import compute_dataset_measures as compute_dataset_measures
from .stat_calculators import compute_measures as compute_measures
from .stat_calculators import compute_sample_measures as compute_sample_measures
from .stat_measures import Stat as Stat
from .tensor import Tensor as Tensor
from .weight_converters import add_weights as add_weights

# aliases
test_resource = test_description
"""alias of `test_description`"""
load_resource = load_description
"""alias of `load_description`"""
load_model = load_model_description
"""alias of `load_model_description`"""
