import json
import subprocess
from typing import Optional

import pytest
from bioimageio.spec._internal.gh_utils import set_github_warning
from packaging.version import Version


def test_bioimageio_spec_version(conda_cmd: Optional[str]):
    if conda_cmd is None:
        pytest.skip("requires conda")

    from importlib.metadata import metadata

    # get latest released bioimageio.spec version
    conda_search = subprocess.run(
        f"{conda_cmd} search --json -f conda-forge::bioimageio.spec>=0.5.3.2".split(),
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    result = json.loads(conda_search.stdout)
    latest_spec = max(result["bioimageio.spec"], key=lambda entry: entry["timestamp"])
    released = Version(latest_spec["version"])

    # get currently pinned bioimageio.spec version
    meta = metadata("bioimageio.core")
    req = meta["Requires-Dist"]
    valid_starts = ("bioimageio.spec ==", "bioimageio.spec==")
    for start in valid_starts:
        if req.startswith(start):
            spec_ver = req[len(start) :]
            break
    else:
        raise ValueError(
            f"Expected bioimageio.sepc pin to start with any of {valid_starts}"
        )

    assert spec_ver.count(".") == 3
    pinned = Version(spec_ver)
    if pinned != released:
        set_github_warning(
            "spec pin mismatch",
            f"bioimageio.spec pinned to {pinned}, while latest version on conda-forge is {released}",
        )
