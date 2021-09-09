from pathlib import Path
from typing import List, Optional

import typer

from bioimageio.core import commands
from bioimageio.spec import __version__
from bioimageio.spec.__main__ import app


@app.command()
def package(
    rdf_source: str = typer.Argument(..., help="RDF source as relative file path or URI"),
    path: Path = typer.Argument(Path() / "{src_name}-package.zip", help="Save package as"),
    update_format: bool = typer.Option(
        False,
        help="Update format version to the latest version (might fail even if source adheres to an old format version). "
        "To inform the format update the source may specify fields of future versions in "
        "config:future:<future version>.",  # todo: add future documentation
    ),
    weights_priority_order: Optional[List[str]] = typer.Option(
        None,
        "-wpo",
        help="For model packages only. "
        "If given only the first weights matching the given weight formats are included. "
        "Defaults to include all weights present in source.",
        show_default=False,
    ),
    verbose: bool = typer.Option(False, help="show traceback of exceptions"),
) -> int:
    return commands.package(rdf_source, path, update_format, weights_priority_order, verbose)


package.__doc__ = commands.package.__doc__


if __name__ == "__main__":
    print(f"bioimageio.spec package version {__version__}")
    app()
