import logging
import os
import subprocess
import warnings

import pytest

os.environ["BIOIMAGEIO_COUNT_RDF_DOWNLOADS"] = "false"  # disable tracking before bioimageio imports
from bioimageio.core import export_resource_package
from bioimageio.spec import __version__ as bioimageio_spec_version


logger = logging.getLogger(__name__)
warnings.warn(f"testing with bioimageio.spec {bioimageio_spec_version}")

# test models for various frameworks
torch_models = [
    "unet2d_fixed_shape",
    "unet2d_multi_tensor",
    "unet2d_nuclei_broad_model",
    "unet2d_diff_output_shape",
    "shape_change",
    "multichannel_output",
]
torchscript_models = ["unet2d_multi_tensor", "unet2d_nuclei_broad_model"]
onnx_models = ["unet2d_multi_tensor", "unet2d_nuclei_broad_model", "hpa_densenet"]
tensorflow1_models = ["stardist"]
tensorflow2_models = ["unet2d_keras_tf2"]
keras_tf1_models = ["unet2d_keras"]
keras_tf2_models = ["unet2d_keras_tf2"]
tensorflow_js_models = []


model_sources = {
    "unet2d_keras": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "unet2d_keras_tf/rdf.yaml"
    ),
    "unet2d_keras_tf2": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "unet2d_keras_tf2/rdf.yaml"
    ),
    "unet2d_nuclei_broad_model": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "unet2d_nuclei_broad/rdf.yaml"
    ),
    "unet2d_expand_output_shape": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "unet2d_nuclei_broad/rdf_expand_output_shape.yaml"
    ),
    "unet2d_fixed_shape": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "unet2d_fixed_shape/rdf.yaml"
    ),
    "unet2d_multi_tensor": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "unet2d_multi_tensor/rdf.yaml"
    ),
    "unet2d_diff_output_shape": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "unet2d_diff_output_shape/rdf.yaml"
    ),
    "hpa_densenet": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/hpa-densenet/rdf.yaml"
    ),
    "stardist": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models"
        "/stardist_example_model/rdf.yaml"
    ),
    "stardist_wrong_shape": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "stardist_example_model/rdf_wrong_shape.yaml"
    ),
    "stardist_wrong_shape2": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "stardist_example_model/rdf_wrong_shape2.yaml"
    ),
    "shape_change": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "upsample_test_model/rdf.yaml"
    ),
    "multichannel_output": "10.5281/zenodo.6079314",
}

try:
    import torch

    torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
    logger.warning(f"detected torch version {torch_version}.x")
except ImportError:
    torch = None
    torch_version = None
skip_torch = torch is None

try:
    import onnxruntime
except ImportError:
    onnxruntime = None
skip_onnx = onnxruntime is None

try:
    import tensorflow

    tf_major_version = int(tensorflow.__version__.split(".")[0])
except ImportError:
    tensorflow = None
    tf_major_version = None
skip_tensorflow = tensorflow is None
skip_tensorflow_js = True  # TODO: add a tensorflow_js example model

# load all model packages we need for testing
load_model_packages = set()
if not skip_torch:
    load_model_packages |= set(torch_models + torchscript_models)

if not skip_onnx:
    load_model_packages |= set(onnx_models)

if not skip_tensorflow:
    load_model_packages |= set(tensorflow_js_models)
    if tf_major_version == 1:
        load_model_packages |= set(keras_tf1_models)
        load_model_packages |= set(tensorflow1_models)
        load_model_packages.add("stardist_wrong_shape")
        load_model_packages.add("stardist_wrong_shape2")
    elif tf_major_version == 2:
        load_model_packages |= set(keras_tf2_models)
        load_model_packages |= set(tensorflow2_models)


def pytest_configure():
    # explicit skip flags needed for some tests
    pytest.skip_torch = skip_torch
    pytest.skip_onnx = skip_onnx

    # load all model packages used in tests
    pytest.model_packages = {name: export_resource_package(model_sources[name]) for name in load_model_packages}

    pytest.mamba_cmd = "micromamba"
    try:
        subprocess.run(["which", pytest.mamba_cmd], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.mamba_cmd = "mamba"
        try:
            subprocess.run(["which", pytest.mamba_cmd], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.mamba_cmd = None


#
# model groups of the form any_<weight format>_model that include all models providing a specific weight format
#


@pytest.fixture(params=[] if skip_torch else torch_models)
def any_torch_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_torch else torchscript_models)
def any_torchscript_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_onnx else onnx_models)
def any_onnx_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_tensorflow else tensorflow1_models if tf_major_version == 1 else tensorflow2_models)
def any_tensorflow_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_tensorflow else keras_tf1_models if tf_major_version == 1 else keras_tf2_models)
def any_keras_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_tensorflow_js else tensorflow_js_models)
def any_tensorflow_js_model(request):
    return pytest.model_packages[request.param]


# fixture to test with all models that should run in the current environment
# we exclude stardist_wrong_shape here because it is not a valid model
# and included only to test that validation for this model fails
@pytest.fixture(params=load_model_packages - {"stardist_wrong_shape", "stardist_wrong_shape2"})
def any_model(request):
    return pytest.model_packages[request.param]


# TODO it would be nice to just generate fixtures for all the individual models dynamically
#
# temporary fixtures to test not with all, but only a manual selection of models
# (models/functionality should be improved to get rid of this specific model group)
#


@pytest.fixture(params=[] if skip_torch else ["unet2d_nuclei_broad_model", "unet2d_fixed_shape"])
def unet2d_fixed_shape_or_not(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_torch else ["unet2d_nuclei_broad_model", "unet2d_multi_tensor"])
def unet2d_multi_tensor_or_not(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_tensorflow else ["unet2d_keras" if tf_major_version == 1 else "unet2d_keras_tf2"])
def unet2d_keras(request):
    return pytest.model_packages[request.param]


# written as model group to automatically skip on missing torch
@pytest.fixture(params=[] if skip_torch else ["unet2d_nuclei_broad_model"])
def unet2d_nuclei_broad_model(request):
    return pytest.model_packages[request.param]


# written as model group to automatically skip on missing torch
@pytest.fixture(params=[] if skip_torch else ["unet2d_diff_output_shape"])
def unet2d_diff_output_shape(request):
    return pytest.model_packages[request.param]


# written as model group to automatically skip on missing torch
@pytest.fixture(params=[] if skip_torch else ["unet2d_expand_output_shape"])
def unet2d_expand_output_shape(request):
    return pytest.model_packages[request.param]


# written as model group to automatically skip on missing torch
@pytest.fixture(params=[] if skip_torch else ["unet2d_fixed_shape"])
def unet2d_fixed_shape(request):
    return pytest.model_packages[request.param]


# written as model group to automatically skip on missing torch
@pytest.fixture(params=[] if skip_torch else ["shape_change"])
def shape_change_model(request):
    return pytest.model_packages[request.param]


# written as model group to automatically skip on missing tensorflow 1
@pytest.fixture(params=[] if skip_tensorflow or tf_major_version != 1 else ["stardist_wrong_shape"])
def stardist_wrong_shape(request):
    return pytest.model_packages[request.param]


# written as model group to automatically skip on missing tensorflow 1
@pytest.fixture(params=[] if skip_tensorflow or tf_major_version != 1 else ["stardist_wrong_shape2"])
def stardist_wrong_shape2(request):
    return pytest.model_packages[request.param]


# written as model group to automatically skip on missing tensorflow 1
@pytest.fixture(params=[] if skip_tensorflow or tf_major_version != 1 else ["stardist"])
def stardist(request):
    return pytest.model_packages[request.param]
