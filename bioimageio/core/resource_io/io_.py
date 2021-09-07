import os
import pathlib
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from zipfile import ZIP_DEFLATED, ZipFile

from marshmallow import ValidationError, missing

from bioimageio import spec
from bioimageio.core.resource_io.nodes import ResourceDescription
from bioimageio.spec.shared import raw_nodes
from bioimageio.spec.shared.common import get_class_name_from_type
from bioimageio.spec.shared.raw_nodes import ResourceDescription as RawResourceDescription
from bioimageio.spec.shared.utils import PathToRemoteUriTransformer
from . import nodes
from .common import BIOIMAGEIO_CACHE_PATH, yaml
from .utils import _download_uri_to_local_path, resolve_local_uri, resolve_raw_resource_description, resolve_uri


serialize_raw_resource_description = spec.io_.serialize_raw_resource_description
save_raw_resource_description = spec.io_.save_raw_resource_description


def _replace_relative_paths_for_remote_source(
    raw_rd: RawResourceDescription, source: Union[Any, str, raw_nodes.URI]
) -> RawResourceDescription:
    if isinstance(source, raw_nodes.URI) or isinstance(source, str) and source.startswith("http"):
        # for a remote source relative paths are invalid; replace all relative file paths in source with URLs
        if isinstance(source, str):
            source = raw_nodes.URI(source)

        warnings.warn(
            f"changing file paths in RDF to URIs due to a remote {source.scheme} source "
            "(may result in an invalid node)"
        )
        raw_rd = PathToRemoteUriTransformer(remote_source=source).transform(raw_rd)

    return raw_rd


def load_raw_resource_description(source: Union[os.PathLike, str, dict, raw_nodes.URI]) -> RawResourceDescription:
    """load a raw python representation from a BioImage.IO resource description file (RDF).
    Use `load_resource_description` for a more convenient representation.

    Args:
        source: resource description file (RDF)

    Returns:
        raw BioImage.IO resource
    """
    data, type_ = resolve_rdf_source_and_type(source)
    raw_rd = spec.load_raw_resource_description_from_dict(data, update_to_current_format=True)
    raw_rd = _replace_relative_paths_for_remote_source(raw_rd, source)
    return raw_rd


def ensure_raw_resource_description(
    source: Union[str, dict, os.PathLike, raw_nodes.URI, RawResourceDescription]
) -> Tuple[RawResourceDescription, pathlib.Path]:
    if isinstance(source, raw_nodes.RawNode) and not isinstance(source, raw_nodes.URI):
        assert isinstance(source, RawResourceDescription)
        return source, pathlib.Path()
    else:
        data, root_path = get_dict_and_root_path_from_yaml_source(source)

    assert isinstance(data, dict)
    raw_rd = load_raw_resource_description(data)
    raw_rd = _replace_relative_paths_for_remote_source(raw_rd, source)
    return raw_rd, root_path


def load_resource_description(
    source: Union[RawResourceDescription, os.PathLike, str, dict, raw_nodes.URI],
    root_path: os.PathLike = pathlib.Path(),
    *,
    weights_priority_order: Optional[Sequence[str]] = None,  # model only
) -> ResourceDescription:
    """load a BioImage.IO resource description file (RDF).
    This includes some transformations for convenience, e.g. importing `source`.
    Use `load_raw_resource_description` to obtain a raw representation instead.

    Args:
        source: resource description file (RDF) or raw BioImage.IO resource
        root_path: to resolve relative paths in the RDF (ignored if source is path/URI)
        weights_priority_order: If given only the first weights format present in the model resource is included
    Returns:
        BioImage.IO resource
    """
    source = deepcopy(source)
    raw_rd, rp = ensure_raw_resource_description(source)
    root_path = root_path / rp

    if weights_priority_order is not None:
        for wf in weights_priority_order:
            if wf in raw_rd.weights:
                raw_rd.weights = {wf: raw_rd.weights[wf]}
                break
        else:
            raise ValueError(f"Not found any of the specified weights formats ({weights_priority_order})")

    rd: ResourceDescription = resolve_raw_resource_description(
        raw_rd=raw_rd, root_path=pathlib.Path(root_path), nodes_module=nodes
    )
    assert isinstance(rd, getattr(nodes, get_class_name_from_type(raw_rd.type)))

    return rd


def get_local_resource_package_content(
    source: RawResourceDescription, root_path: os.PathLike, weights_priority_order: Optional[Sequence[Union[str]]]
) -> Dict[str, Union[pathlib.Path, str]]:
    """

    Args:
        source: raw resource description
        root_path: root for relative paths
        weights_priority_order: If given only the first weights format present in the model is included.
                                If none of the prioritized weights formats is found all are included.

    Returns:
        Package content of local file paths or text content keyed by file names.

    """
    raw_rd, rp = ensure_raw_resource_description(source)
    root_path = root_path / rp
    package_content = spec.get_resource_package_content(raw_rd, weights_priority_order=weights_priority_order)

    local_package_content = {}
    for k, v in package_content.items():
        if isinstance(v, raw_nodes.URI):
            v = resolve_uri(v, root_path)
        elif isinstance(v, pathlib.Path):
            v = root_path / v

        local_package_content[k] = v

    return local_package_content


def export_resource_package(
    source: Union[RawResourceDescription, os.PathLike, str, dict, raw_nodes.URI],
    root_path: os.PathLike = pathlib.Path(),
    *,
    output_path: Optional[os.PathLike] = None,
    weights_priority_order: Optional[Sequence[Union[str]]] = None,
    compression: int = ZIP_DEFLATED,
    compression_level: int = 1,
) -> pathlib.Path:
    """Package a BioImage.IO resource as a zip file.

    Args:
        source: raw resource description, path, URI or raw data as dict
        root_path: for relative paths (only used if source is RawResourceDescription or dict)
        output_path: file path to write package to
        weights_priority_order: If given only the first weights format present in the model is included.
                                If none of the prioritized weights formats is found all are included.
        compression: The numeric constant of compression method.
        compression_level: Compression level to use when writing files to the archive.
                           See https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile

    Returns:
        path to zipped BioImage.IO package in BIOIMAGEIO_CACHE_PATH or 'output_path'
    """
    raw_rd, rp = ensure_raw_resource_description(source)
    root_path = root_path / rp
    package_content = get_local_resource_package_content(raw_rd, root_path, weights_priority_order)
    if output_path is None:
        package_path = _get_tmp_package_path(raw_rd, weights_priority_order)
    else:
        package_path = output_path

    make_zip(package_path, package_content, compression=compression, compression_level=compression_level)
    return package_path


def _get_package_base_name(raw_rd: RawResourceDescription, weights_priority_order: Optional[Sequence[str]]) -> str:
    package_file_name = raw_rd.name
    if raw_rd.version is not missing:
        package_file_name += f"_{raw_rd.version}"

    package_file_name = package_file_name.replace(" ", "_").replace(".", "_")

    return package_file_name


def _get_tmp_package_path(raw_rd: RawResourceDescription, weights_priority_order: Optional[Sequence[str]]):
    package_file_name = _get_package_base_name(raw_rd, weights_priority_order)

    BIOIMAGEIO_CACHE_PATH.mkdir(exist_ok=True, parents=True)
    package_path = (BIOIMAGEIO_CACHE_PATH / package_file_name).with_suffix(".zip")
    max_cached_packages_with_same_name = 100
    for p in range(max_cached_packages_with_same_name):
        if package_path.exists():
            package_path = (BIOIMAGEIO_CACHE_PATH / f"{package_file_name}p{p}").with_suffix(".zip")
        else:
            break
    else:
        raise FileExistsError(
            f"Already caching {max_cached_packages_with_same_name} versions of {BIOIMAGEIO_CACHE_PATH / package_file_name}!"
        )

    return package_path


def extract_resource_package(source: Union[os.PathLike, str, raw_nodes.URI]) -> pathlib.Path:
    """extract a zip source to BIOIMAGEIO_CACHE_PATH"""
    local_source = resolve_uri(source)
    assert isinstance(local_source, pathlib.Path)
    BIOIMAGEIO_CACHE_PATH.mkdir(exist_ok=True, parents=True)
    package_path = BIOIMAGEIO_CACHE_PATH / f"{local_source.stem}_unzipped"
    with ZipFile(local_source) as zf:
        zf.extractall(package_path)

    for rdf_name in ["rdf.yaml", "model.yaml", "rdf.yml", "model.yml"]:
        rdf_path = package_path / rdf_name
        if rdf_path.exists():
            break
    else:
        raise FileNotFoundError(local_source / "rdf.yaml")

    return rdf_path


def make_zip(
    path: os.PathLike, content: Dict[str, Union[str, pathlib.Path]], *, compression: int, compression_level: int
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


def resolve_rdf_source_and_type(source: Union[os.PathLike, str, dict, raw_nodes.URI]) -> Tuple[dict, str]:
    if isinstance(source, dict):
        data = source
    else:
        source = resolve_local_uri(source, pathlib.Path())
        data, root_path = get_dict_and_root_path_from_yaml_source(source)

    type_ = data.get("type", "model")  # todo: remove default 'model' type

    return data, type_


def get_dict_and_root_path_from_yaml_source(
    source: Union[os.PathLike, str, raw_nodes.URI, dict]
) -> Tuple[dict, pathlib.Path]:
    if isinstance(source, dict):
        return source, pathlib.Path()
    elif isinstance(source, (str, os.PathLike, raw_nodes.URI)):
        source = resolve_local_uri(source, pathlib.Path())
    else:
        raise TypeError(source)

    if isinstance(source, raw_nodes.URI):  # remote uri
        local_source = _download_uri_to_local_path(source)
        root_path = pathlib.Path()
    else:
        local_source = source
        root_path = source.parent

    assert isinstance(local_source, pathlib.Path)
    if local_source.suffix == ".zip":
        local_source = extract_resource_package(local_source)
        root_path = local_source.parent

    if local_source.suffix == ".yml":
        warnings.warn(
            "suffix '.yml' is not recommended and will raise a ValidationError in the future. Use '.yaml' instead "
            "(https://yaml.org/faq.html)"
        )
    elif local_source.suffix != ".yaml":
        raise ValidationError(f"invalid suffix {local_source.suffix} for source {source}")

    data = yaml.load(local_source)
    assert isinstance(data, dict)
    return data, root_path
