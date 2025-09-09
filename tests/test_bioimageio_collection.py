import os
from itertools import chain
from pathlib import Path
from typing import Iterable, Mapping, Tuple

import pytest
from pydantic import HttpUrl

from bioimageio.spec import InvalidDescr, settings
from bioimageio.spec.common import Sha256
from tests.utils import ParameterSet, expensive_test

TEST_RDF_SOURCES: Mapping[str, Tuple[HttpUrl, Sha256]] = {
    "affable-shark": (
        HttpUrl(
            "https://hypha.aicell.io/bioimage-io/artifacts/affable-shark/files/rdf.yaml?version=v0"
        ),
        Sha256("b74944b4949591d3eaf231cf9ab259f91dec679863020178e6c3ddadd52a019c"),
    ),
    "ambitious-sloth": (
        HttpUrl(
            "https://hypha.aicell.io/bioimage-io/artifacts/ambitious-sloth/files/rdf.yaml?version=v0"
        ),
        Sha256("caf162e847a0812fb7704e7848b1ee68f46383278d8b74493553fe96750d1e39"),
    ),
}


def yield_bioimageio_yaml_urls() -> Iterable[ParameterSet]:
    for key, (descr_url, sha) in TEST_RDF_SOURCES.items():
        yield pytest.param(descr_url, sha, key, id=key)


def get_directory_size(path: Path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


@pytest.mark.parametrize("descr_url,sha,key", list(yield_bioimageio_yaml_urls()))
def test_rdf_format_to_populate_cache(
    descr_url: HttpUrl,
    sha: Sha256,
    key: str,
):
    """this test is redundant if `test_rdf` runs, but is used in the CI to populate the cache"""
    from bioimageio.spec._internal.gh_utils import set_github_warning

    if os.environ.get("BIOIMAGEIO_POPULATE_CACHE") != "1":
        pytest.skip("BIOIMAGEIO_POPULATE_CACHE != 1")

    if (cache_size := get_directory_size(settings.cache_path)) > 7e9:
        msg = f"Reached 7GB cache size limit ({cache_size / 1e9:.2f} GB)"
        set_github_warning("Reached cache size limit", msg)
        pytest.skip(msg)

    from bioimageio.core import load_description

    _ = load_description(descr_url, sha256=sha, perform_io_checks=True)


@expensive_test
@pytest.mark.parametrize("descr_url,sha,key", list(yield_bioimageio_yaml_urls()))
def test_rdf(
    descr_url: HttpUrl,
    sha: Sha256,
    key: str,
):
    from bioimageio.core import load_description, load_description_and_test
    from bioimageio.spec import get_conda_env
    from bioimageio.spec.model import ModelDescr

    descr = load_description(
        descr_url, sha256=sha, format_version="latest", perform_io_checks=True
    )
    assert not isinstance(descr, InvalidDescr), descr.validation_summary.display() or [
        e.msg for e in descr.validation_summary.errors
    ]

    if (
        isinstance(descr, ModelDescr)
        and descr.weights.pytorch_state_dict is not None
        and descr.weights.pytorch_state_dict.dependencies is not None
    ):
        conda_env = get_conda_env(entry=descr.weights.pytorch_state_dict)

        def depends_on(dep: str) -> bool:
            return any(
                chain(
                    (d.startswith(dep) for d in conda_env.get_pip_deps()),
                    (cd for cd in conda_env.dependencies if isinstance(cd, str)),
                )
            )

        for skip_if_depends_on in (
            "biapy",
            "git+https://github.com/CAREamics/careamics.git",
            "careamics",
            "inferno",
            "plantseg",
        ):
            if depends_on(skip_if_depends_on):
                pytest.skip(f"requires {skip_if_depends_on}")

    descr = load_description_and_test(
        descr,
        format_version="latest",
        sha256=sha,
        stop_early=True,
    )

    assert not isinstance(descr, InvalidDescr), descr.validation_summary.display() or [
        e.msg for e in descr.validation_summary.errors
    ]
    assert descr.validation_summary.status == "passed", (
        descr.validation_summary.display()
        or [e.msg for e in descr.validation_summary.errors]
    )
