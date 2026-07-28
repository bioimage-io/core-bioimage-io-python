"""Collection utilities for bioimageio"""

import json
import warnings
from functools import cache
from typing import Literal, Mapping, Sequence, TypedDict

from bioimageio.spec.common import Sha256
from bioimageio.spec.utils import get_reader

from ._settings import settings


class IndexItemVersion(TypedDict):
    version: str
    source: str
    sha256: Sha256


class IndexItem(TypedDict):
    id: str
    type: str
    versions: Sequence[IndexItemVersion]


class Index(TypedDict):
    items: Sequence[IndexItem]
    total: int
    count_per_type: Mapping[str, int]
    timestamp: str


class IdPartsEntry(TypedDict):
    nouns: Mapping[str, str]
    adjectives: Sequence[str]


class CollectionConfig(TypedDict):
    id_parts: Mapping[Literal["model", "dataset", "notebook"], IdPartsEntry]
    reviewers: Mapping[str, Sequence[str]]


@cache
def load_json(url: str):
    reader = get_reader(url)
    return json.load(reader)


def load_index() -> Index:
    return load_json(settings.collection_index_url)


def load_collection_config() -> CollectionConfig:
    return load_json(settings.collection_config_url)


@cache
def load_hyphened_nouns() -> set[str]:
    """get all nouns with hyphens that could be part of a nickname, e.g. 't-rex'"""
    return {
        noun
        for id_parts in load_collection_config()["id_parts"].values()
        for noun in id_parts.get("nouns", [])
        if "-" in noun
    }


def lookup_from_index(source: str) -> tuple[str, dict[Literal["sha256"], Sha256]]:
    index = load_index()
    if source.startswith("bioimage-io/") and source.count("/") == 2:
        version = source.split("/")[-1]
        source = source[: -(len(version) + 1)]
    else:
        version = None

    for item in index["items"]:
        if item["id"] in (source, f"bioimage-io/{source}"):
            v = item["versions"][-1]
            if version is not None:
                for v in item["versions"]:
                    if v["version"] == version:
                        break
                else:
                    warnings.warn(
                        f"Version {version} not found in index, using latest version."
                    )
        else:
            continue

        return v["source"], {"sha256": v["sha256"]}

    return source, {}


def get_resource_icon(nickname: str, rtype: str) -> str:
    """Get emoji for a resource, matching to its nickname noun. nicknames are of the form "{adjective}-{noun}", e.g. "affable-shark"."""
    if "-" not in nickname:
        return " "

    # remove hyphen from noun part of nickname, e.g. "laid-back-t-rex" -> "laid-back-trex"
    for hyphened_noun in load_hyphened_nouns():
        if nickname.endswith(hyphened_noun):
            nickname = nickname[: -len(hyphened_noun)] + hyphened_noun.replace("-", "")

    # last hyphen now sparates adjective and noun, e.g. "laid-back-trex" -> "laid-back" and "trex"
    noun = nickname[nickname.rfind("-") + 1 :].replace("-", "")
    try:
        ret = {
            k.replace("-", ""): v
            for k, v in load_collection_config()["id_parts"][
                rtype if rtype in ("model", "dataset", "notebook") else "notebook"
            ]
            .get("nouns", {})
            .items()
        }.get(noun, " ")
    except Exception as e:
        warnings.warn(f"Error getting icon for {rtype} {nickname}: {e}")
        ret = " "

    return ret
