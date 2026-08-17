import numpy as np
import pytest

pytest.importorskip("skimage")

from bioimageio.core import microsam_watershed


def _two_cell_maps(shape=(64, 64)):
    """Synthetic AIS decoder maps for two round cells."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    centers = [(16, 16), (16, 48)]
    radius = 16.0

    foreground = np.zeros(shape, dtype="float32")
    center_distances = np.ones(shape, dtype="float32")
    for cy, cx in centers:
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        foreground = np.maximum(foreground, (d < radius).astype("float32"))
        center_distances = np.minimum(center_distances, np.clip(d / radius, 0, 1))
    # inverted boundary distance: low deep inside the cells, high toward the
    # rim, with an explicit ridge at the touching border
    boundary_distances = center_distances.copy()
    boundary_distances[:, 31:33] = 1.0
    return np.stack([foreground, center_distances, boundary_distances])


def test_two_instances_from_stacked_maps():
    maps = _two_cell_maps()
    seg = microsam_watershed(maps)
    ids = set(np.unique(seg)) - {0}
    assert len(ids) == 2
    # instances are separated left/right
    assert seg[16, 16] != seg[16, 48]
    assert seg[16, 16] != 0 and seg[16, 48] != 0
    # background stays background
    assert seg[60, 32] == 0


def test_individual_maps_match_stacked():
    maps = _two_cell_maps()
    seg_stacked = microsam_watershed(maps)
    seg_individual = microsam_watershed(
        foreground=maps[0], center_distances=maps[1], boundary_distances=maps[2]
    )
    assert np.array_equal(seg_stacked, seg_individual)


def test_min_size_filters_small_instances():
    maps = _two_cell_maps()
    # add a tiny speck
    maps[0, 58:62, 58:62] = 1.0
    maps[1, 58:62, 58:62] = 0.0
    maps[2, 58:62, 58:62] = 0.0
    seg_all = microsam_watershed(maps, foreground_smoothing=0, distance_smoothing=0)
    seg_filtered = microsam_watershed(
        maps, foreground_smoothing=0, distance_smoothing=0, min_size=50
    )
    assert len(set(np.unique(seg_all)) - {0}) == 3
    assert len(set(np.unique(seg_filtered)) - {0}) == 2
    assert (seg_filtered[58:62, 58:62] == 0).all()


def test_input_validation():
    maps = _two_cell_maps()
    with pytest.raises(ValueError, match="not both"):
        microsam_watershed(maps, foreground=maps[0])
    with pytest.raises(ValueError, match="channel axis of size 3"):
        microsam_watershed(maps[:2])
    with pytest.raises(ValueError, match="all three"):
        microsam_watershed(foreground=maps[0], center_distances=maps[1])
