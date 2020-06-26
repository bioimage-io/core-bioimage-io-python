import json
from pathlib import Path

from marshmallow_jsonschema import JSONSchema  # todo: add marshmallow_jsonschema to dependencies

from pybio.spec.schema import ModelSpec

JSON_SPEC_PATH = Path(__file__).parent / "../json_specs"

model_schema = ModelSpec()

with (JSON_SPEC_PATH / "model_spec.json").open("w") as f:
    json_schema = JSONSchema().dump(model_schema)
    # drop kwargs for now as discussed in https://github.com/bioimage-io/python-bioimage-io/issues/27
    json_schema["definitions"]["ModelSpec"]["properties"].pop("optional_kwargs")
    json_schema["definitions"]["ModelSpec"]["properties"].pop("required_kwargs")
    json.dump(json_schema, f, indent=4, sort_keys=True)
