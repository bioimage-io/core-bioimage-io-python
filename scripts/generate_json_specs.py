import json
from pathlib import Path

from marshmallow_jsonschema import JSONSchema

from pybio.spec.schema import ModelSpec

JSON_SPEC_PATH = Path(__file__).parent / "../json_specs"

model_schema = ModelSpec()

with (JSON_SPEC_PATH / "model_spec.json").open("w") as f:
    json.dump(JSONSchema().dump(model_schema), f)
