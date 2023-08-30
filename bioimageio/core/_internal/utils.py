from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Union
from urllib.parse import urlsplit, urlunsplit
from zipfile import ZipFile

from bioimageio.spec.types import FileName
from pydantic import AnyUrl, FilePath, HttpUrl

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
    content: Mapping[FileName, Union[str, FilePath]],
    *,
    compression: int,
    compression_level: int,
) -> None:
    """Write a zip archive.

    Args:
        path: output path to write to.
        content: dict with archive names and local file paths or strings for text files.
        compression: The numeric constant of compression method.
        compression_level: Compression level to use when writing files to the archive.
                           See https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile

    """
    with ZipFile(path, "w", compression=compression, compresslevel=compression_level) as myzip:
        for arc_name, file_or_str_content in content.items():
            if isinstance(file_or_str_content, str):
                myzip.writestr(arc_name, file_or_str_content)
            else:
                myzip.write(file_or_str_content, arcname=arc_name)
