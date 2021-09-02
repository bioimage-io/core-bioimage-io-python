import json
import pathlib

VERSION = json.loads((pathlib.Path(__file__).parent / "VERSION").read_text())["version"]
