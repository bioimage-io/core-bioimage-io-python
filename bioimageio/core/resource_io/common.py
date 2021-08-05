import os
import pathlib
import tempfile
from collections import UserDict
from typing import Optional
from bioimageio.spec.shared.common import yaml

assert yaml is not None, "missing yaml dependency?!"

try:
    from typing import Literal, get_args, get_origin, Protocol
except ImportError:
    from typing_extensions import Literal, get_args, get_origin, Protocol  # type: ignore


BIOIMAGEIO_CACHE_PATH = pathlib.Path(
    os.getenv("BIOIMAGEIO_CACHE_PATH", pathlib.Path(tempfile.gettempdir()) / "bioimageio_cache")
)


class NoOverridesDict(UserDict):
    def __init__(self, *args, key_exists_error_msg: Optional[str] = None, allow_if_same_value: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_exists_error_message = (
            "key {key} already exists!" if key_exists_error_msg is None else key_exists_error_msg
        )
        self.allow_if_same_value = allow_if_same_value

    def __setitem__(self, key, value):
        if key in self and (not self.allow_if_same_value or value != self[key]):
            raise ValueError(self.key_exists_error_message.format(key=key, value=value))

        super().__setitem__(key, value)


def nested_default_dict_as_nested_dict(nested_dd):
    return {
        key: (nested_default_dict_as_nested_dict(value) if isinstance(value, dict) else value)
        for key, value in nested_dd.items()
    }
