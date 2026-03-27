from difflib import HtmlDiff, unified_diff
from typing import Sequence

from loguru import logger
from ruyaml import Optional
from typing_extensions import Literal, assert_never

from bioimageio.spec._internal.version_type import Version


def compare(
    a: Sequence[str],
    b: Sequence[str],
    name_a: str = "source",
    name_b: str = "updated",
    *,
    diff_format: Literal["unified", "html"],
):
    if diff_format == "html":
        diff = HtmlDiff().make_file(a, b, name_a, name_b, charset="utf-8")
    elif diff_format == "unified":
        diff = "\n".join(
            unified_diff(
                a,
                b,
                name_a,
                name_b,
                lineterm="",
            )
        )
    else:
        assert_never(diff_format)

    return diff


def warn_about_version(
    name: str, specified_version: Optional[Version], actual_version: Optional[Version]
) -> None:
    if actual_version is None or specified_version is None:
        logger.warning("Could not compare actual and installed {} versions.", name)
    elif actual_version < specified_version:
        logger.warning(
            "Installed {} version {} is lower than specified version {}.",
            name,
            actual_version,
            specified_version,
        )
    elif (specified_version.major, specified_version.minor) != (
        actual_version.major,
        actual_version.minor,
    ):
        logger.warning(
            "Installed {} version {} does not match specified {}.",
            name,
            actual_version,
            specified_version,
        )
