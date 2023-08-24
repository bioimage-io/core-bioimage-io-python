import getpass
import os
import tempfile
import warnings
from pathlib import Path

from bioimageio.spec.types import ValidationSummary


class TestSummary(ValidationSummary):
    bioimageio_core_version: str


# BIOIMAGEIO_CACHE_PATH = Path(
#     os.getenv("BIOIMAGEIO_CACHE_PATH", Path(tempfile.gettempdir()) / getpass.getuser() / "bioimageio_cache")
# )

# BIOIMAGEIO_USE_CACHE = os.getenv("BIOIMAGEIO_USE_CACHE", "true").lower() in ("1", "yes", "true")
# if (env_val := os.getenv("BIOIMAGEIO_USE_CACHE", "true").lower()) not in ("0", "1", "no", "yes", "false", "true"):
#     warnings.warn(f"Unrecognized BIOIMAGEIO_USE_CACHE environment value '{env_val}'")
