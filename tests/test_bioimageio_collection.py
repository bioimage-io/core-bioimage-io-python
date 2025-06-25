import os
from itertools import chain
from typing import Any, Dict, Iterable, Mapping, Tuple

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
            HttpUrl(version["source"]),
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


KNOWN_INVALID: Mapping[str, str] = {
    "affectionate-cow/0.1.0": "custom dependencies",
    "ambitious-sloth/1.2": "requires inferno",
    "appealing-popcorn/1": "missing license",
    "appetizing-eggplant/1": "missing license",
    "appetizing-peach/1": "missing license",
    "authoritative-ballet-shoes/1.13.1": "invalid id",
    "biapy/biapy/1": "invalid github user arratemunoz and lmescu",
    "bitter-hot-dog/1": "missing license",
    "bold-shorts/1.13": "invalid id",
    "brisk-scarf/1.16.2": "missing license",
    "buttery-apple/1": "missing cite",
    "buttery-sandwich/1": "missing license",
    "cheerful-cap/1.15.3": "missing license",
    "chewy-garlic/1": "missing license",
    "classy-googles/1": "missing license",
    "committed-turkey/1.2": "error deserializing VarianceScaling",
    "convenient-purse/1.14.1": "missing license",
    "convenient-t-shirt/1.14.1": "missing license",
    "cozy-hiking-boot/1.16.2": "missing license",
    "creative-panda/1": "error deserializing Conv2D",
    "crunchy-cookie/1": "missing license",
    "dazzling-spider/0.1.0": "requires careamics",
    "delectable-eggplant/1": "missing license",
    "delicious-cheese/1": "missing license",
    "determined-hedgehog/1": "wrong output shape?",
    "discreet-rooster/1": "error deserializing VarianceScaling",
    "discreete-rooster/1": "error deserializing VarianceScaling",
    "divine-paella/1": "missing license",
    "dl4miceverywhere/DL4MicEverywhere/1": "invalid id",
    "dynamic-t-rex/1": "needs update to 0.5 for scale_linear with axes processing",
    "easy-going-sauropod/1": (
        "CPU implementation of Conv3D currently only supports the NHWC tensor format."
    ),
    "efficient-chipmunk/1": "needs plantseg",
    "emotional-cricket/1.1": "sporadic 403 responses from  https://elifesciences.org",
    "exciting-backpack/1.19.1": "missing license",
    "exquisite-curry/1": "missing license",
    "famous-fish/0.1.0": "list index out of range `fl[3]`",
    "fiji/Fiji/1": "invalid id",
    "flattering-bikini/1.13.2": "missing license",
    "flexible-helmet/1.14.1": "missing license",
    "fluffy-popcorn/1": "missing license",
    "fluid-glasses/1.17.2": "missing license",
    "fruity-sushi/1": "missing license",
    "fun-high-heels/1.15.2": "missing license",
    "funny-butterfly/1": "Do not specify an axis for scalar gain and offset values.",
    "greedy-whale/1": "batch size is actually limited to 1",
    "happy-elephant/0.1.0": "list index out of range `fl[3]`",
    "happy-honeybee/0.1.0": "requires biapy",
    "heroic-otter/0.1.0": "requires biapy",
    "hpa/HPA-Classification/1": "invalid id",
    "humorous-crab/1": "batch size is actually limited to 1",
    "humorous-fox/0.1.0": "requires careamics",
    "humorous-owl/1": "error deserializing GlorotUniform",
    "icy/icy/1": "invalid github user 'None'",
    "idealistic-turtle/0.1.0": "requires biapy",
    "imjoy/BioImageIO-Packager/1": "invalid id",
    "imjoy/GenericBioEngineApp/1": "invalid documentation suffix",
    "imjoy/HPA-Single-Cell/1": "invalid documentation suffix",
    "imjoy/ImageJ.JS/1": "invalid documentation suffix",
    "imjoy/ImJoy/1": "invalid documentation suffix",
    "imjoy/vizarr/1": "invalid documentation suffix",
    "impartial-shark/1": "error deserializing VarianceScaling",
    "indulgent-sandwich/1": "missing license",
    "inspiring-sandal/1.13.3": "missing license",
    "intelligent-lion/0.1.0": "requires biapy",
    "irresistible-swimsuit/1.14.1": "missing license",
    "joyful-deer/1": "needs update to 0.5 for scale_linear with axes processing",
    "joyful-top-hat/2.2.1": "missing license",
    "juicy-peanut/1": "missing license",
    "light-swimsuit/1.13": "missing license",
    "limited-edition-crown/1.14.1": "missing license",
    "lively-t-shirt/1.13": "missing license",
    "luscious-tomato/1": "missing license",
    "mellow-broccoli/1": "missing license",
    "mellow-takeout/1": "missing cite",
    "merry-water-buffalo/0.1.0": "requires biapy",
    "mesmerizing-shoe/1.14.1": "missing license",
    "naked-microbe/1": "unknown layer Convolution2D",
    "nice-peacock/1": "invalid id",
    "noisy-ox/1": "batch size is actually limited to 1",
    "non-judgemental-eagle/1": "error deserializing GlorotUniform",
    "nutty-burrito/1": "missing license",
    "nutty-knuckle/1": "missing license",
    "opalescent-ribbon/1.15.3": "missing license",
    "palatable-curry/1": "missing license",
    "polished-t-shirt/1.16.2": "missing license",
    "powerful-sandal/1": "missing license",
    "regal-ribbon/1.14.1": "missing license",
    "resourceful-potato/1": "missing license",
    "resplendent-ribbon/2.2.1": "missing license",
    "rich-burrito/1": "missing license",
    "rich-cheese/1": "missing license",
    "savory-cheese/1": "missing license",
    "silky-shorts/1.13": "missing license",
    "slinky-bikini/1.15.1": "missing license",
    "smooth-graduation-hat/1.15.0": "missing license",
    "smooth-hat/1.1.0": "invalid id",
    "smooth-safety-vest/1.14.1": "missing license, invalid id",
    "smooth-scarf/1": "invalid id",
    "sparkling-sari/1.0.0": "missing license, invalid id",
    "straightforward-crocodile/1": (
        "needs update to 0.5 for scale_linear with axes processing"
    ),
    "striking-necktie/1.14.1": "invalid id",
    "stupendous-sheep/1.1": "requires relativ import of attachment",
    "tempting-pizza/1": "missing license",
    "timeless-running-shirt/1.13.2": "invalid id, missing license",
    "uplifting-backpack/1.14.1": "invalid id, missing license",
    "venomous-swan/0.1.0": "requires biapy",
    "whimsical-helmet/2.1.2": "invalid id",
    "wild-rhino/0.1.0": "requires careamics",
    "zero/notebook_preview/1": "missing authors",
}


@pytest.mark.parametrize("descr_url,sha,key", list(yield_bioimageio_yaml_urls()))
def test_rdf_format_to_populate_cache(
    descr_url: HttpUrl,
    sha: Sha256,
    key: str,
):
    """this test is redundant if `test_rdf` runs, but is used in the CI to populate the cache"""
    if os.environ.get("BIOIMAGEIO_POPULATE_CACHE") != "1":
        pytest.skip("BIOIMAGEIO_POPULATE_CACHE != 1")

    if key in KNOWN_INVALID:
        pytest.skip(KNOWN_INVALID[key])

    from bioimageio.core import load_description

    _ = load_description(descr_url, sha256=sha, perform_io_checks=True)


@expensive_test
@pytest.mark.parametrize("descr_url,sha,key", list(yield_bioimageio_yaml_urls()))
def test_rdf(
    descr_url: HttpUrl,
    sha: Sha256,
    key: str,
):
    from bioimageio.spec import get_conda_env
    from bioimageio.spec.model import ModelDescr

    if key in KNOWN_INVALID:
        pytest.skip(KNOWN_INVALID[key])

    from bioimageio.core import load_description, load_description_and_test

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
    assert (
        descr.validation_summary.status == "passed"
    ), descr.validation_summary.display() or [
        e.msg for e in descr.validation_summary.errors
    ]
