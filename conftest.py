from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def pytest_ignore_collect(collection_path: Path, config: Any) -> bool:
    if sys.version_info >= (3, 10):
        return False

    path = str(collection_path).replace("\\", "/")
    return "/src/bioimageio/core/remote_backends/gradio/" in path
