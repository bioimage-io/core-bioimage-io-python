from os import getenv
from pathlib import Path

PYBIO_CACHE_PATH = getenv("PYBIO_CACHE_PATH")
if PYBIO_CACHE_PATH is None:
    cache_path = Path.home() / "pybio_cache"
else:
    cache_path = Path(PYBIO_CACHE_PATH)
