import json
from pathlib import Path

from marshmallow_jsonschema import JSONSchema  # todo: add marshmallow_jsonschema to dependencies

from pybio.spec.schema import Model

JSON_SPEC_PATH = Path(__file__).parent / "../json_specs"


def generate_json_model_spec():
    model_schema = Model()

    with (JSON_SPEC_PATH / "model_spec.json").open("w") as f:
        json_schema = JSONSchema().dump(model_schema)
        json.dump(json_schema, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    generate_json_model_spec()  # todo: use marshmallow union for fields.Shap
