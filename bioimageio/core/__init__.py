import json
import pathlib

__version__ = json.loads((pathlib.Path(__file__).parent / "VERSION").read_text())["version"]
