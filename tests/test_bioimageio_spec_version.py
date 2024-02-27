import json
import subprocess
from typing import Optional

import pytest
from packaging.version import Version


def test_bioimageio_spec_version(mamba_cmd: Optional[str]):
    if mamba_cmd is None:
        pytest.skip("requires mamba")

    from importlib.metadata import metadata

    # get latest released bioimageio.spec version
    mamba_repoquery = subprocess.run(
        f"{pytest.mamba_cmd} repoquery search -c conda-forge --json bioimageio.spec".split(" "),
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    full_out = mamba_repoquery.stdout  # full output includes mamba banner
    search = json.loads(full_out[full_out.find("{") :])  # json output starts at '{'
    rmaj, rmin, rpatch, *_ = search["result"]["pkgs"][0]["version"].split(".")
    released = Version(f"{rmaj}.{rmin}.{rpatch}")

    # get currently pinned bioimageio.spec version
    meta = metadata("bioimageio.core")
    req = meta["Requires-Dist"]
    assert req.startswith("bioimageio.spec ==")
    spec_ver = req[len("bioimageio.spec ==") :]
    assert spec_ver.count(".") == 3
    pmaj, pmin, ppatch, post = spec_ver.split(".")
    assert (
        pmaj.isdigit() and pmin.isdigit() and ppatch.isdigit() and post == "*"
    ), "bioimageio.spec version should be pinned down to patch, e.g. '0.4.9.*'"

    pinned = Version(f"{pmaj}.{pmin}.{ppatch}")
    assert pinned == released, "bioimageio.spec not pinned to the latest version"
