from __future__ import annotations

from pydantic import BaseModel

from ._xml_utils import xml_to_dict


class ImageJMetadata(BaseModel, extra="allow"):
    """(Partial) Metadata for ImageJ TIFF files."""

    unit: str | None = None
    """The unit of measurement for the image (e.g., "micron", "pixel")."""

    yunit: str | None = None
    """The unit of measurement for the Y dimension (e.g., "micron", "pixel")."""

    zunit: str | None = None
    """The unit of measurement for the Z dimension (e.g., "micron", "pixel")."""

    spacing: float | None = None
    """The physical spacing between slices in the Z dimension."""

    finterval: float | None = None
    """Time interval between frames in seconds."""

    fps: float | None = None
    """Frames per second for time-lapse images."""


class OmeChannelMetadata(BaseModel, extra="allow"):
    """Partial Metadata for OME-TIFF channel information."""

    Name: list[str]
    """The name of the channel."""


class OmeMetadata(BaseModel, extra="allow"):
    """Partial Metadata for ImageJ TIFF files."""

    Channel: OmeChannelMetadata | None = None
    DimensionOrder: str | None = None
    PhysicalSizeX: float | None = None
    PhysicalSizeXUnit: str | None = None
    PhysicalSizeY: float | None = None
    PhysicalSizeYUnit: str | None = None
    PhysicalSizeZ: float | None = None
    PhysicalSizeZUnit: str | None = None
    TimeIncrement: float | None = None
    TimeIncrementUnit: str | None = None


def create_ome_metadata_from_xml_string(xml_string: str) -> OmeMetadata:
    raw_ome_metadata = xml_to_dict(xml_string)
    return OmeMetadata.model_validate(raw_ome_metadata)
