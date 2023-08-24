import shutil
import traceback
from pathlib import Path
from typing import List, Optional, Union

from bioimageio.spec import validate
from bioimageio.spec.shared.raw_nodes import URI

from bioimageio.core import export_resource_package
from bioimageio.core._internal.validation_visitors import resolve_source


def package(
    rdf_source: Union[Path, str, URI, dict],
    path: Path = Path() / "{src_name}-package.zip",
    weights_priority_order: Optional[List[str]] = None,
    verbose: bool = False,
) -> int:
    """Package a bioimage.io resource described by a bioimage.io Resource Description File (RDF)."""
    rd, summary = load_description(rdf_source, update_format=True, update_format_inner=True)
    source_name = rdf_source.get("name") if isinstance(rdf_source, dict) else rdf_source
    if code["status"] != "passed":
        print(f"Cannot package invalid bioimage.io RDF {source_name}")
        return 1

    try:
        tmp_package_path = export_resource_package(rdf_source, weights_priority_order=weights_priority_order)
    except Exception as e:
        print(f"Failed to package {source_name} due to: {e}")
        if verbose:
            traceback.print_exc()
        return 1

    try:
        rdf_local_source = resolve_source(rdf_source)
    except Exception as e:
        print(f"Failed to resolve RDF source {rdf_source}: {e}")
        if verbose:
            traceback.print_exc()
        return 1

    try:
        path = path.with_name(path.name.format(src_name=rdf_local_source.stem))
        shutil.move(tmp_package_path, path)
    except Exception as e:
        print(f"Failed to move package from {tmp_package_path} to {path} due to: {e}")
        if verbose:
            traceback.print_exc()
        return 1

    print(f"exported bioimageio package from {source_name} to {path}")
    return 0
