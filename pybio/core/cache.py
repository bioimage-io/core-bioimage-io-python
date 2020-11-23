from os import getenv
from pathlib import Path

PYBIO_CACHE_PATH_ENV = getenv("PYBIO_CACHE_PATH")
if PYBIO_CACHE_PATH_ENV is None:
    PYBIO_CACHE_PATH = Path.home() / "pybio_cache"
else:
    PYBIO_CACHE_PATH = Path(PYBIO_CACHE_PATH_ENV)
