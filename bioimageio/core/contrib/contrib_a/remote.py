import sys
from pathlib import Path

from bioimageio.core.contrib.utils import RemoteContrib

remote_module = RemoteContrib(Path(__file__).parent.stem)
__all__ = remote_module.__all__
sys.modules[__name__] = remote_module  # noqa
