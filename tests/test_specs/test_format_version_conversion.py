from dataclasses import asdict

from pybio.spec import schema
from pybio.spec.utils import FormatConverter


def test_model_nodes_format_0_1_to_0_3(rf_model_data_v0_1, rf_model_data):
    expected = asdict(schema.Model().load(rf_model_data))
    converted_data = FormatConverter()(rf_model_data_v0_1)
    actual = asdict(schema.Model().load(converted_data))

    # expect converted description
    for ipt in expected["inputs"]:
        ipt["description"] = ipt["name"]

    for out in expected["outputs"]:
        out["description"] = out["name"]

    for key, item in expected.items():
        assert key in actual
        assert actual[key] == item

    for key, item in actual.items():
        assert key in expected
        assert expected[key] == item
