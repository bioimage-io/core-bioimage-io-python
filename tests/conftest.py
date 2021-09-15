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

# set 'skip_<FRAMEWORK>' flags as global pytest variables,
# to deselect tests that require frameworks not available in current env
def pytest_configure():
    try:
        import torch
    except ImportError:
        torch = None
    pytest.skip_torch = torch is None

    try:
        import onnxruntime
    except ImportError:
        onnxruntime = None
    pytest.skip_onnx = onnxruntime is None

    try:
        import tensorflow

        pytest.tf_major_version = int(tensorflow.__version__.split(".")[0])
    except ImportError:
        tensorflow = None
    pytest.skip_tensorflow = tensorflow is None
    pytest.skip_tensorflow = True  # todo: update FruNet and remove this

    try:
        import keras
    except ImportError:
        keras = None
    pytest.skip_keras = keras is None
    pytest.skip_keras = True  # FruNet requires update

    # load all model packages we need for testing
    load_packages = {"unet2d_nuclei_broad_model"}  # always load unet2d_nuclei_broad_model
    if not pytest.skip_torch:
        load_packages |= set(torch_models + torchscript_models)

    if not pytest.skip_onnx:
        load_packages |= set(onnx_models)

    if not pytest.skip_tensorflow:
        load_packages |= set(keras_models)
        load_packages |= set(tensorflow_js_models)
        if pytest.tf_major_version == 1:
            load_packages |= set(tensorflow1_models)
        elif pytest.tf_major_version == 2:
            load_packages |= set(tensorflow2_models)

    pytest.model_packages = {name: export_resource_package(model_sources[name]) for name in load_packages}


@pytest.fixture
def unet2d_nuclei_broad_model():
    return pytest.model_packages["unet2d_nuclei_broad_model"]


@pytest.fixture
def unet2d_multi_tensor():
    return pytest.model_packages["unet2d_multi_tensor"]


@pytest.fixture
def FruNet_model():
    return pytest.model_packages["FruNet_model"]


#
# model groups
#


@pytest.fixture(params=torch_models)
def any_torch_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=torchscript_models)
def any_torchscript_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=onnx_models)
def any_onnx_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=set(tensorflow1_models) | set(tensorflow2_models))
def any_tensorflow_model(request):
    name = request.param
    if (pytest.tf_major_version == 1 and name in tensorflow1_models) or (
        pytest.tf_major_version == 2 and name in tensorflow2_models
    ):
        return pytest.model_packages[name]


@pytest.fixture(params=keras_models)
def any_keras_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=tensorflow_js_models)
def any_tensorflow_js_model(request):
    return pytest.model_packages[request.param]


@pytest.fixture(params=model_sources.keys())
def any_model(request):
    return pytest.model_packages[request.param]
