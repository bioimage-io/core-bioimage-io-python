import sys
from pathlib import Path

from ._import_callable import import_callable as import_callable
from ._tensor_io import get_test_inputs as get_test_inputs
from ._tensor_io import get_test_outputs as get_test_outputs

if sys.version_info < (3, 9):

    def files(package_name: str):
        assert package_name == "bioimageio.core"
        return Path(__file__).parent.parent

else:
    from importlib.resources import files as files
