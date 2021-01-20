import pytest

from pybio.spec import nodes, schema


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
