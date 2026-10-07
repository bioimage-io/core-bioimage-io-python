from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import zarr
from loguru import logger
from torch import TYPE_CHECKING

if TYPE_CHECKING:
    import clearscale


def open_zarr_multiscale_array(
    uri: str,
) -> tuple[zarr.Array[Any], clearscale.Multiscale]:
    """Open a zarr multiscale array from a URI"""
    ms, _uri_to_parent, _sub_uri = _recurse_url_to_find_group(uri)

    zarr_array_or_group = zarr.open(uri)
    if isinstance(zarr_array_or_group, zarr.Group):
        keys = list(ms.keys())
        if len(keys) == 1:
            return open_zarr_multiscale_array(f"{uri}/{keys[0]}")
        else:
            raise ValueError(
                f"{uri} does not point to a zarr array, but a group with keys: {keys}."
                + " Append '/<key>' to the URI to open a specific array."
            )

    return zarr_array_or_group, ms


def _recurse_url_to_find_group(uri: str) -> tuple[clearscale.Multiscale, str, str]:
    """URI may point to an OME-Zarr "multiscales" root or to a specific scale.
    Try to find OME-Zarr spec first at URI, then search parent directories.
    Returns
     - the valid Multiscale
     - root URI of the multiscales group
     - scale sub-URI"""
    uri = uri.rstrip("/")
    i = len(uri)
    parent_group = None
    uri_to_parent = uri
    sub_uri = ""
    for _ in range(uri.count("/")):
        try:
            parent_group = zarr.open(uri_to_parent)
            if isinstance(parent_group, zarr.Group):
                break
        except Exception:
            # stop looking once we checked a ".zarr" directory
            if uri_to_parent.split("/")[-1].endswith(".zarr"):
                break

        i = uri.rfind("/", 0, i)
        uri_to_parent = uri[:i]
        sub_uri = uri[i + 1 :]

    if isinstance(parent_group, zarr.Group):
        return (
            _ms_from_group(parent_group, sub_uri),
            uri_to_parent,
            sub_uri,
        )

    raise ValueError("No multiscale meta found anywhere along the URI")


def _ms_from_group(
    zarr_group: zarr.Group, must_have_dataset: str | None = None
) -> clearscale.Multiscale:
    import clearscale

    # 1. Discover the "multiscales" list
    try:
        ome = zarr_group.attrs["ome"]
        assert isinstance(ome, Mapping)
        ome_multiscales = ome["multiscales"]
    except Exception:
        try:
            ome_multiscales = zarr_group.attrs["multiscales"]
        except Exception:
            raise ValueError("No multiscale metadata found in zarr group")

    if not isinstance(ome_multiscales, Sequence) or isinstance(ome_multiscales, str):
        raise TypeError(
            "Multiscale metadata in this zarr group is not a valid sequence"
        )

    # 2. Convert all entries of the "multiscales" list
    # (There will practically never be more than one)
    valid_multiscales: list[clearscale.Multiscale] = []
    for ome_ms in ome_multiscales:
        try:
            assert isinstance(ome_ms, Mapping)
            valid_multiscales.append(
                clearscale.Multiscale.from_ome_zarr(
                    ome_ms, shape_source=_get_shape_callable(zarr_group)
                )
            )
        except Exception as e:
            logger.debug("Encountered invalid multiscale metadata: {}", e)
            continue  # Invalid multiscale - maybe warn, or just skip

    if not valid_multiscales:
        raise ValueError("Multiscale metadata in this zarr group was all invalid")

    if must_have_dataset:
        selected_ms = next(
            (ms for ms in valid_multiscales if must_have_dataset in ms), None
        )
        if selected_ms is None:
            raise ValueError(
                f"No Multiscale containing a dataset named {must_have_dataset!r} found"
            )
        return selected_ms

    # 3. Handle the off-chance that there is more than one for some reason.
    if len(valid_multiscales) > 1:
        # ilastik bounces back group URLs in this case and forces selection by passing a direct URL to a dataset.
        # Many tools just default to valid_multiscales[0] and ignore further multiscale definitions.
        raise ValueError(
            f"{len(valid_multiscales)} Multiscale definitions found. Selection required."
        )
    else:
        return valid_multiscales[0]


def _get_shape_callable(zarr_group: zarr.Group):
    def get_shape(path: str):
        zarr_array = zarr_group[path]
        assert isinstance(zarr_array, zarr.Array)
        return zarr_array.shape

    return get_shape
