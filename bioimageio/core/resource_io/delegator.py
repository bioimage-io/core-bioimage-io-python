from __future__ import annotations

import os
import pathlib
import warnings
from copy import deepcopy
from io import StringIO
from typing import Dict, Optional, Sequence, TYPE_CHECKING, Tuple, Union
from zipfile import ZIP_DEFLATED

from bioimageio import spec
from bioimageio.core.resource_io.io_ import extract_resource_package, resolve_rdf_source_and_type
from bioimageio.spec.shared import base_nodes, raw_nodes
from bioimageio.spec.shared.nodes import ResourceDescription
from bioimageio.spec.shared.raw_nodes import ResourceDescription as RawResourceDescription
from bioimageio.spec.shared.utils import PathToRemoteUriTransformer
from .common import yaml
from .utils import resolve_uri

if TYPE_CHECKING:
    import bioimageio.spec.model


def load_raw_resource_description(
    source: Union[os.PathLike, str, dict, base_nodes.URI], update_to_current_format: bool = False
) -> RawResourceDescription:
    """load a raw python representation from a BioImage.IO resource description file (RDF).
    Use `load_resource_description` for a more convenient representation.

    Args:
        source: resource description file (RDF)
        update_to_current_format: auto convert content to adhere to the latest appropriate RDF format version

    Returns:
        raw BioImage.IO resource
    """
    data, type_ = resolve_rdf_source_and_type(source)
    raw_rd = spec.load_raw_resource_description(data, update_to_current_format)
    if isinstance(source, base_nodes.URI) or isinstance(source, str) and source.startswith("http"):
        # for a remote source relative paths are invalid; replace all relative file paths in source with URLs
        if isinstance(source, str):
            source = raw_nodes.URI(source)

        warnings.warn(
            f"changing file paths in RDF to URIs due to a remote {source.scheme} source "
            "(may result in an invalid node)"
        )
        raw_rd = PathToRemoteUriTransformer(remote_source=source).transform(raw_rd)

    return raw_rd


def save_raw_resource_description(raw_rd: RawResourceDescription, path: pathlib.Path):
    warnings.warn("only saving serialized rdf, no associated resources.")
    if path.suffix != ".yaml":
        warnings.warn("saving with '.yaml' suffix is strongly encouraged.")

    serialized = spec.serialize_raw_resource_description_to_dict(raw_rd)
    yaml.dump(serialized, path)


def serialize_raw_resource_description(raw_rd: RawResourceDescription) -> str:
    serialized = spec.serialize_raw_resource_description_to_dict(raw_rd)

    with StringIO() as stream:
        yaml.dump(serialized, stream)
        return stream.getvalue()


def ensure_raw_resource_description(
    source: Union[str, dict, os.PathLike, base_nodes.URI, RawResourceDescription],
    root_path: os.PathLike = pathlib.Path(),
    update_to_current_format: bool = True,
) -> Tuple[RawResourceDescription, pathlib.Path]:
    if isinstance(source, raw_nodes.RawNode):
        assert isinstance(source, RawResourceDescription)
        return source, pathlib.Path(root_path)
    elif isinstance(source, dict):
        data = source
    elif isinstance(source, (str, os.PathLike, base_nodes.URI)):
        local_raw_rd = resolve_uri(source, root_path)
        if local_raw_rd.suffix == ".zip":
            local_raw_rd = extract_resource_package(local_raw_rd)
            raw_rd = local_raw_rd  # zip package contains everything. ok to 'forget' that source was remote

        root_path = local_raw_rd.parent
    else:
        raise TypeError(raw_rd)

    assert not isinstance(raw_rd, RawResourceDescription)
    return cls.load_raw_resource_description(raw_rd), root_path


    if isinstance(raw_rd, raw_nodes.ResourceDescription) and not isinstance(raw_rd, base_nodes.URI):
        return raw_rd, pathlib.Path(root_path)

    data, type_ = resolve_rdf_source_and_type(raw_rd)
    format_version = "latest" if update_to_current_format else data.get("format_version", "latest")
    io_cls = _get_matching_io_class(type_, format_version)

    return io_cls.ensure_raw_rd(raw_rd, root_path)


def load_resource_description(
    source: Union[RawResourceDescription, os.PathLike, str, dict, base_nodes.URI],
    root_path: os.PathLike = pathlib.Path(),
    *,
    update_to_current_format: bool = True,
    weights_priority_order: Optional[Sequence[str]] = None,  # model only
) -> ResourceDescription:
    """load a BioImage.IO resource description file (RDF).
    This includes some transformations for convenience, e.g. importing `source`.
    Use `load_raw_resource_description` to obtain a raw representation instead.

    Args:
        source: resource description file (RDF) or raw BioImage.IO resource
        root_path: to resolve relative paths in the RDF (ignored if source is path/URI)
        update_to_current_format: auto convert content to adhere to the latest appropriate RDF format version
        weights_priority_order: If given only the first weights format present in the model resource is included
    Returns:
        BioImage.IO resource
    """
    source = deepcopy(source)
    raw_rd, root_path = ensure_raw_resource_description(source, root_path, update_to_current_format)

    if weights_priority_order is not None:
        for wf in weights_priority_order:
            if wf in raw_rd.weights:
                raw_rd.weights = {wf: raw_rd.weights[wf]}
                break
        else:
            raise ValueError(f"Not found any of the specified weights formats ({weights_priority_order})")

    rd: ResourceDescription = resolve_raw_resource_description(
        raw_rd=raw_rd, root_path=pathlib.Path(root_path), nodes_module=cls.nodes
    )
    assert isinstance(rd, getattr(cls.nodes, get_class_name_from_type(raw_rd.type)))

    return rd


def export_resource_package(
    source: Union[RawResourceDescription, os.PathLike, str, dict, base_nodes.URI],
    root_path: os.PathLike = pathlib.Path(),
    *,
    output_path: Optional[os.PathLike] = None,
    update_to_current_format: bool = False,
    weights_priority_order: Optional[Sequence[Union[bioimageio.spec.model.v0_3.base_nodes.WeightsFormat]]] = None,
    compression: int = ZIP_DEFLATED,
    compression_level: int = 1,
) -> pathlib.Path:
    """Package a BioImage.IO resource as a zip file.

    Args:
        source: raw resource description, path, URI or raw data as dict
        root_path: for relative paths (only used if source is RawResourceDescription or dict)
        output_path: file path to write package to
        update_to_current_format: Convert not only the patch version, but also the major and minor version.
        weights_priority_order: If given only the first weights format present in the model is included.
                                If none of the prioritized weights formats is found all are included.
        compression: The numeric constant of compression method.
        compression_level: Compression level to use when writing files to the archive.
                           See https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile

    Returns:
        path to zipped BioImage.IO package in BIOIMAGEIO_CACHE_PATH or 'output_path'
    """
    raw_rd, _ = ensure_raw_resource_description(source, root_path, update_to_current_format)
    io_cls = _get_matching_io_class(raw_rd.type, raw_rd.format_version)
    return io_cls.export_resource_package(
        source,
        root_path,
        output_path=output_path,
        weights_priority_order=weights_priority_order,
        compression=compression,
        compression_level=compression_level,
    )


def get_resource_package_content(
    source: Union[RawResourceDescription, os.PathLike, str, dict],
    root_path: pathlib.Path,
    update_to_current_format: bool = False,
    *,
    weights_priority_order: Optional[Sequence[str]] = None,
) -> Dict[str, Union[str, pathlib.Path]]:
    """
    Args:
        source: raw resource description, path, URI or raw data as dict
        root_path:  for relative paths (only used if source is RawResourceDescription or dict)
        update_to_current_format: Convert not only the patch version, but also the major and minor version.
        weights_priority_order: If given only the first weights format present in the model is included.
                                If none of the prioritized weights formats is found all are included.

    Returns:
        Package content of local file paths or text content keyed by file names.
    """
    raw_rd, _ = ensure_raw_resource_description(source, root_path, update_to_current_format)
    io_cls = _get_matching_io_class(raw_rd.type, raw_rd.format_version)
    return io_cls.get_resource_package_content(source, root_path, weights_priority_order=weights_priority_order)
