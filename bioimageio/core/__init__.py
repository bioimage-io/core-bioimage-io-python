import json
import pathlib

from .resource_io import (
    export_resource_package,
    load_raw_resource_description,
    load_resource_description,
    save_raw_resource_description,
    serialize_raw_resource_description,
)


VERSION = json.loads((pathlib.Path(__file__).parent / "VERSION").read_text())["version"]
