import json
import sys
from pathlib import Path

if sys.version_info < (3, 9):

    def files(package_name: str):
        assert package_name == "bioimageio.core"
        return Path(__file__).parent.parent

else:
    from importlib.resources import files as files


with files("bioimageio.core").joinpath("VERSION").open("r", encoding="utf-8") as f:
    VERSION = json.load(f)["version"]
    assert isinstance(VERSION, str)
