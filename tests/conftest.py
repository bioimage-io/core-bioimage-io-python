import pytest
from bioimageio.core import export_resource_package

# test models for various frameworks
torch_models = ["unet2d_fixed_shape", "unet2d_multi_tensor", "unet2d_nuclei_broad_model"]
torchscript_models = ["unet2d_multi_tensor", "unet2d_nuclei_broad_model"]
onnx_models = ["unet2d_multi_tensor", "unet2d_nuclei_broad_model"]
tensorflow1_models = ["FruNet_model"]
tensorflow2_models = []
keras_models = ["FruNet_model"]
tensorflow_js_models = ["FruNet_model"]

model_sources = {
    "FruNet_model": "https://sandbox.zenodo.org/record/894498/files/rdf.yaml",
    # "FruNet_model": "https://raw.githubusercontent.com/deepimagej/models/master/fru-net_sev_segmentation/model.yaml",
    "unet2d_nuclei_broad_model": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "unet2d_nuclei_broad/rdf.yaml"
    ),
    "unet2d_fixed_shape": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "unet2d_fixed_shape/rdf.yaml"
    ),
    "unet2d_multi_tensor": (
        "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
        "unet2d_multi_tensor/rdf.yaml"
    ),
}

try:
    import torch
except ImportError:
    torch = None
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
skip_tensorflow = True  # todo: update FruNet and remove this
skip_tensorflow_js = True  # todo: update FruNet and figure out how to test tensorflow_js weights in python

try:
    import keras
except ImportError:
    keras = None
skip_keras = keras is None
skip_keras = True  # FruNet requires update

# load all model packages we need for testing
load_model_packages = set()
if not skip_torch:
    load_model_packages |= set(torch_models + torchscript_models)

if not skip_onnx:
    load_model_packages |= set(onnx_models)

if not skip_tensorflow:
    load_model_packages |= set(keras_models)
    load_model_packages |= set(tensorflow_js_models)
    if tf_major_version == 1:
        load_model_packages |= set(tensorflow1_models)
    elif tf_major_version == 2:
        load_model_packages |= set(tensorflow2_models)


def pytest_configure():

    # explicit skip flags needed for some tests
    pytest.skip_torch = skip_torch
    pytest.skip_onnx = skip_onnx

    # load all model packages used in tests
    pytest.model_packages = {name: export_resource_package(model_sources[name]) for name in load_model_packages}


#
# model groups of the form any_<weight format>_model that include all models providing a specific weight format
#

# written as model group to automatically skip on missing torch
@pytest.fixture(params=[] if skip_torch else ["unet2d_nuclei_broad_model"])
def unet2d_nuclei_broad_model(request):
    return pytest.model_packages[request.param]


# written as model group to automatically skip on missing tensorflow 1
@pytest.fixture(params=[] if skip_tensorflow or tf_major_version != 1 else ["FruNet_model"])
def FruNet_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_torch else torch_models)
def any_torch_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_torch else torchscript_models)
def any_torchscript_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_onnx else onnx_models)
def any_onnx_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_tensorflow else (set(tensorflow1_models) | set(tensorflow2_models)))
def any_tensorflow_model(request):
    name = request.param
    if (tf_major_version == 1 and name in tensorflow1_models) or (tf_major_version == 2 and name in tensorflow2_models):
        return pytest.model_packages[name]


@pytest.fixture(params=[] if skip_keras else keras_models)
def any_keras_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=[] if skip_tensorflow_js else tensorflow_js_models)
def any_tensorflow_js_model(request):
    return pytest.model_packages[request.param]


# fixture to test with all models that should run in the current environment
@pytest.fixture(params=load_model_packages)
def any_model(request):
    return pytest.model_packages[request.param]


# temporary fixture to test not with all, but only a manual selection of models
# (models/functionality should be improved to get rid of this specific model group)
@pytest.fixture(params=[] if skip_torch else ["unet2d_nuclei_broad_model", "unet2d_fixed_shape"])
def unet2d_fixed_shape_or_not(request):
    return pytest.model_packages[request.param]
