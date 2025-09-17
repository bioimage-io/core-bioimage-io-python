from difflib import HtmlDiff, unified_diff
from typing import Sequence

from typing_extensions import Literal, assert_never


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
