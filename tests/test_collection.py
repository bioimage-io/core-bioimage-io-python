import json
from io import StringIO
from unittest.mock import Mock

import pytest

from bioimageio.core import _collection as collection
from bioimageio.core._settings import settings
from bioimageio.spec.common import Sha256


@pytest.fixture(autouse=True)
def clear_collection_caches():
    collection.load_json.cache_clear()
    collection.load_hyphened_nouns.cache_clear()
    yield
    collection.load_json.cache_clear()
    collection.load_hyphened_nouns.cache_clear()


def test_json_loaders_cache_per_url(monkeypatch: pytest.MonkeyPatch):
    def read_json(url: str):
        return StringIO(json.dumps({"url": url}))

    reader = Mock(side_effect=read_json)
    monkeypatch.setattr(collection, "get_reader", reader)
    assert collection.load_index() == {"url": settings.collection_index_url}
    assert collection.load_index() == {"url": settings.collection_index_url}
    assert collection.load_collection_config() == {
        "url": settings.collection_config_url
    }
    assert reader.call_count == 2


@pytest.mark.parametrize(
    "source,version",
    [
        ("kind-shark", "v2"),
        ("bioimage-io/kind-shark", "v2"),
        ("bioimage-io/kind-shark/v1", "v1"),
    ],
)
def test_index_lookup(monkeypatch: pytest.MonkeyPatch, source: str, version: str):
    versions = [
        {"version": "v1", "source": "first.yaml", "sha256": Sha256("1" * 64)},
        {"version": "v2", "source": "last.yaml", "sha256": Sha256("2" * 64)},
    ]
    monkeypatch.setattr(
        collection,
        "load_index",
        Mock(
            return_value={
                "items": [
                    {"id": "other"},
                    {"id": "bioimage-io/kind-shark", "versions": versions},
                ]
            }
        ),
    )
    selected = versions[0 if version == "v1" else 1]
    assert collection.lookup_from_index(source) == (
        selected["source"],
        {"sha256": selected["sha256"]},
    )
    assert collection.lookup_from_index("unknown") == ("unknown", {})
    with pytest.warns(UserWarning, match="Version missing not found"):
        assert collection.lookup_from_index("bioimage-io/kind-shark/missing") == (
            "last.yaml",
            {"sha256": Sha256("2" * 64)},
        )


def test_resource_icons(monkeypatch: pytest.MonkeyPatch):
    config = Mock(
        return_value={
            "id_parts": {
                "model": {"nouns": {"t-rex": "🦖", "shark": "🦈"}},
                "dataset": {"nouns": {}},
                "notebook": {"nouns": {"owl": "🦉"}},
            }
        }
    )
    monkeypatch.setattr(collection, "load_collection_config", config)
    assert collection.load_hyphened_nouns() == {"t-rex"}
    assert collection.load_hyphened_nouns() == {"t-rex"}
    config.assert_called_once()
    assert collection.get_resource_icon("plain", "model") == " "
    assert collection.get_resource_icon("laid-back-t-rex", "model") == "🦖"
    assert collection.get_resource_icon("kind-shark", "model") == "🦈"
    assert collection.get_resource_icon("kind-owl", "unknown") == "🦉"
    assert collection.get_resource_icon("kind-shark", "dataset") == " "
    config.side_effect = RuntimeError("unavailable")
    with pytest.warns(UserWarning, match="Error getting icon"):
        assert collection.get_resource_icon("kind-shark", "model") == " "
