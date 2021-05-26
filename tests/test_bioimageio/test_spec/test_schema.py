import pytest
from datetime import datetime

from bioimageio.spec import nodes, schema


def test_tensor_schema_preprocessing():
    data = {
        "name": "input_1",
        "description": "Input 1",
        "data_type": "float32",
        "axes": "xyc",
        "shape": [128, 128, 3],
        "preprocessing": [
            {
                "name": "scale_range",
                "kwargs": {"max_percentile": 99, "min_percentile": 5, "mode": "per_sample", "axes": "xy"},
            }
        ],
    }
    validated_data = schema.InputTensor().load(data)
    assert isinstance(validated_data, nodes.InputTensor)
    assert validated_data.name == data["name"]
    assert validated_data.description == data["description"]
    assert validated_data.data_type == data["data_type"]
    assert validated_data.axes == data["axes"]
    assert validated_data.shape == data["shape"]

    assert isinstance(validated_data.preprocessing, list)
    assert len(validated_data.preprocessing) == 1
    preprocessing = validated_data.preprocessing[0]
    assert preprocessing.name == "scale_range"


@pytest.mark.parametrize(
    "data",
    [
        {
            "name": "input_1",
            "description": "Input 1",
            "data_type": "float32",
            "axes": "xyc",
            "shape": [128, 128, 3],
            "preprocessing": [],
        },
        {
            "name": "input_1",
            "description": "Input 1",
            "data_type": "float32",
            "axes": "xyc",
            "shape": [128, 128, 3],
        },
    ],
)
def test_tensor_schema_no_preprocessing(data):
    validated_data = schema.InputTensor().load(data)
    assert isinstance(validated_data.preprocessing, list)
    assert len(validated_data.preprocessing) == 0


@pytest.mark.parametrize("schema_instance", [schema.InputTensor(), schema.OutputTensor()])
def test_tensor_schema_optional_description(schema_instance):
    data = {
        "name": "input_1",
        "data_type": "float32",
        "axes": "xyc",
        "shape": [128, 128, 3],
    }
    validated_data = schema_instance.load(data)
    assert validated_data.description is None


@pytest.fixture
def model_dict():
    """
    Valid model dict fixture
    """
    return {
        "documentation": "./docs.md",
        "license": "MIT",
        "framework": "pytorch",
        "language": "python",
        "source": "somesrc",
        "git_repo": "https://github.com/bioimage-io/python-bioimage-io",
        "format_version": "0.3.0",
        "description": "description",
        "authors": ["Author 1", "Author 2"],
        "timestamp": datetime.now(),
        "cite": [
            {
                "text": "Paper title",
                "doi": "doi",
            }
        ],
        "inputs": [
            {
                "name": "input_1",
                "description": "Input 1",
                "data_type": "float32",
                "axes": "xyc",
                "shape": [128, 128, 3],
            }
        ],
        "outputs": [
            {
                "name": "output_1",
                "description": "Output 1",
                "data_type": "float32",
                "axes": "xyc",
                "shape": [128, 128, 3],
            }
        ],
        "name": "Model",
        "tags": [],
        "weights": {},
        "test_inputs": [],
        "test_outputs": [],
    }


def test_model_schema_accepts_run_mode(model_dict):
    model_schema = schema.Model()
    model_dict.update({"run_mode": {"name": "special_run_mode", "kwargs": dict(marathon=True)}})
    validated_data = model_schema.load(model_dict)
    assert validated_data


@pytest.mark.parametrize(
    "format",
    [
        "pickle",
        "pytorch_state_dict",
        "pytorch_script",
        "keras_hdf5",
        "tensorflow_js",
        "tensorflow_saved_model_bundle",
        "onnx",
    ],
)
def test_model_schema_accepts_valid_weight_formats(model_dict, format):
    model_schema = schema.Model()
    model_dict.update({"weights": {format: {"source": "local_weights"}}})
    validated_data = model_schema.load(model_dict)
    assert validated_data
