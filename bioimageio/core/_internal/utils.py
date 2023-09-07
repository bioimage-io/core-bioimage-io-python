from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Union
from urllib.parse import urlsplit, urlunsplit
from zipfile import ZipFile

from bioimageio.spec._internal.types import FileName
from pydantic import AnyUrl, FilePath, HttpUrl
from ruamel.yaml import YAML

yaml = YAML(typ="safe")
if sys.version_info < (3, 9):

    def files(package_name: str):
        assert package_name == "bioimageio.core"
        return Path(__file__).parent.parent

else:
    from importlib.resources import files as files


def get_parent_url(url: HttpUrl) -> HttpUrl:
    parsed = urlsplit(str(url))
    return AnyUrl(
        urlunsplit((parsed.scheme, parsed.netloc, "/".join(parsed.path.split("/")[:-1]), parsed.query, parsed.fragment))
    )


def write_zip(
    path: os.PathLike[str],
    content: Mapping[FileName, Union[str, FilePath, Dict[Any, Any]]],
    *,
    compression: int,
    compression_level: int,
) -> None:
    """Write a zip archive.

    Args:
        path: output path to write to.
        content: dict mapping archive names to local file paths, strings (for text files), or dict (for yaml files).
        compression: The numeric constant of compression method.
        compression_level: Compression level to use when writing files to the archive.
                           See https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile

    """
    with ZipFile(path, "w", compression=compression, compresslevel=compression_level) as myzip:
        for arc_name, file in content.items():
            if isinstance(file, dict):
                buf = io.StringIO()
                YAML.dump(file, buf)
                file = buf.getvalue()

            if isinstance(file, str):
                myzip.writestr(arc_name, file.encode("utf-8"))
            else:
                myzip.write(file, arcname=arc_name)
