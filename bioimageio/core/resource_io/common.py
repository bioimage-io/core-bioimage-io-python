import os
import pathlib
import tempfile

from bioimageio.spec.shared.common import yaml

assert yaml is not None, "missing yaml dependency?!"


BIOIMAGEIO_CACHE_PATH = pathlib.Path(
    os.getenv("BIOIMAGEIO_CACHE_PATH", pathlib.Path(tempfile.gettempdir()) / "bioimageio_cache")
)
