import os
from typing import Any, Collection, Dict, Iterable, Mapping, Tuple

import httpx
import pytest
from pydantic import HttpUrl

from bioimageio.spec import InvalidDescr
from bioimageio.spec.common import Sha256
from tests.utils import ParameterSet, expensive_test

BASE_URL = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/"


def _get_latest_rdf_sources():
    entries: Any = httpx.get(BASE_URL + "all_versions.json").json()["entries"]
    ret: Dict[str, Tuple[HttpUrl, Sha256]] = {}
    for entry in entries:
        version = entry["versions"][0]
        ret[f"{entry['concept']}/{version['v']}"] = (
            HttpUrl(version["source"]),  # pyright: ignore[reportCallIssue]
            Sha256(version["sha256"]),
        )

    return ret


ALL_LATEST_RDF_SOURCES: Mapping[str, Tuple[HttpUrl, Sha256]] = _get_latest_rdf_sources()


def yield_bioimageio_yaml_urls() -> Iterable[ParameterSet]:
    for descr_url, sha in ALL_LATEST_RDF_SOURCES.values():
        key = (
            str(descr_url)
            .replace(BASE_URL, "")
            .replace("/files/rdf.yaml", "")
            .replace("/files/bioimageio.yaml", "")
        )
        yield pytest.param(descr_url, sha, key, id=key)


KNOWN_INVALID: Collection[str] = {
    "affectionate-cow/0.1.0",  # custom dependencies
    "ambitious-sloth/1.2",  # requires inferno
    "committed-turkey/1.2",  # error deserializing VarianceScaling
    "creative-panda/1",  # error deserializing Conv2D
    "dazzling-spider/0.1.0",  # requires careamics
    "discreet-rooster/1",  # error deserializing VarianceScaling
    "discreete-rooster/1",  # error deserializing VarianceScaling
    "dynamic-t-rex/1",  # needs update to 0.5 for scale_linear with axes processing
    "easy-going-sauropod/1",  # CPU implementation of Conv3D currently only supports the NHWC tensor format.
    "efficient-chipmunk/1",  # needs plantseg
    "emotional-cricket/1.1",  # sporadic 403 responses from  https://elifesciences.org
    "famous-fish/0.1.0",  # list index out of range `fl[3]`
    "greedy-whale/1",  # batch size is actually limited to 1
    "happy-elephant/0.1.0",  # list index out of range `fl[3]`
    "happy-honeybee/0.1.0",  # requires biapy
    "heroic-otter/0.1.0",  # requires biapy
    "humorous-crab/1",  # batch size is actually limited to 1
    "humorous-fox/0.1.0",  # requires careamics
    "humorous-owl/1",  # error deserializing GlorotUniform
    "idealistic-turtle/0.1.0",  # requires biapy
    "impartial-shark/1",  # error deserializing VarianceScaling
    "intelligent-lion/0.1.0",  # requires biapy
    "joyful-deer/1",  # needs update to 0.5 for scale_linear with axes processing
    "merry-water-buffalo/0.1.0",  # requires biapy
    "naked-microbe/1",  # unknown layer Convolution2D
    "noisy-ox/1",  # batch size is actually limited to 1
    "non-judgemental-eagle/1",  # error deserializing GlorotUniform
    "straightforward-crocodile/1",  # needs update to 0.5 for scale_linear with axes processing
    "stupendous-sheep/1.1",  # requires relativ import of attachment
    "stupendous-sheep/1.2",
    "venomous-swan/0.1.0",  # requires biapy
    "wild-rhino/0.1.0",  # requires careamics
}


@pytest.mark.parametrize("descr_url,sha,key", list(yield_bioimageio_yaml_urls()))
def test_rdf_format_to_populate_cache(
    descr_url: HttpUrl,
    sha: Sha256,
    key: str,
):
    """this test is redundant if `test_rdf` runs, but is used in the CI to populate the cache"""
    if os.environ.get("BIOIMAGEIO_POPULATE_CACHE") != "1":
        pytest.skip("only runs in CI to populate cache")

    if key in KNOWN_INVALID:
        pytest.skip("known failure")

    from bioimageio.core import load_description

    _ = load_description(descr_url, sha256=sha, perform_io_checks=True)


@expensive_test
@pytest.mark.parametrize("descr_url,sha,key", list(yield_bioimageio_yaml_urls()))
def test_rdf(
    descr_url: HttpUrl,
    sha: Sha256,
    key: str,
):
    if key in KNOWN_INVALID:
        pytest.skip("known failure")

    from bioimageio.core import load_description_and_test

    descr = load_description_and_test(descr_url, sha256=sha, stop_early=True)

    assert not isinstance(descr, InvalidDescr), descr.validation_summary.display()
    assert (
        descr.validation_summary.status == "passed"
    ), descr.validation_summary.display()
