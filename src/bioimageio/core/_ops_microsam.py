"""micro-sam (μSAM) instance segmentation postprocessing.

μSAM models with an additional instance segmentation (AIS) decoder predict
three dense maps: foreground probability, center distance and boundary
distance. Instance labels are obtained from these maps with a seeded
watershed, as implemented in
`micro_sam.instance_segmentation.InstanceSegmentationWithDecoder` /
`torch_em.util.segmentation.watershed_from_center_and_boundary_distances`:

- Anna Archit et al. [*Segment Anything for Microscopy*](https://www.nature.com/articles/s41592-024-02580-4).
  Nature Methods, 2025.

This module ports that postprocessing so that the packaged, prompt-free
μSAM AIS models (which output the three maps) can be turned into instance
segmentations by any tool building on bioimageio.core.

Note: the watershed itself requires the `scikit-image` package
(`pip install bioimageio.core[microsam]`).
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as connected_components


def microsam_watershed(
    maps: Optional[NDArray[np.floating]] = None,
    *,
    foreground: Optional[NDArray[np.floating]] = None,
    center_distances: Optional[NDArray[np.floating]] = None,
    boundary_distances: Optional[NDArray[np.floating]] = None,
    center_distance_threshold: float = 0.5,
    boundary_distance_threshold: float = 0.5,
    foreground_threshold: float = 0.5,
    foreground_smoothing: float = 1.0,
    distance_smoothing: float = 1.6,
    min_size: int = 0,
) -> NDArray[np.int64]:
    """Compute an instance segmentation from μSAM AIS decoder predictions.

    Seeds are connected components where both (smoothed) distance predictions
    are below their thresholds, intersected with the foreground mask. The
    seeded watershed then runs on the smoothed boundary distances, restricted
    to the foreground mask.

    Args:
        maps: The stacked decoder output with a leading channel axis of size 3
            in the order (foreground, center_distances, boundary_distances),
            i.e. shape (3, *spatial). Mutually exclusive with passing the
            three maps individually.
        foreground: Foreground probability prediction.
        center_distances: Distance prediction to the object centers.
        boundary_distances: Inverted distance prediction to object boundaries.
        center_distance_threshold: Center distance predictions below this
            value are used to find seeds.
        boundary_distance_threshold: Boundary distance predictions below this
            value are used to find seeds.
        foreground_threshold: Foreground predictions above this value make up
            the foreground mask.
        foreground_smoothing: Sigma for gaussian smoothing of the foreground
            prediction (avoids checkerboard artifacts). Set to 0 to disable.
        distance_smoothing: Sigma for gaussian smoothing of both distance
            predictions. Set to 0 to disable.
        min_size: Minimal object size; smaller instances are removed.

    Returns:
        The instance segmentation as an integer label image.
    """
    try:
        from skimage.segmentation import watershed
    except ImportError as e:
        raise ImportError(
            "microsam_watershed requires the scikit-image package."
            + " Install it e.g. via `pip install bioimageio.core[microsam]`."
        ) from e

    if maps is not None:
        if foreground is not None or center_distances is not None or boundary_distances is not None:
            raise ValueError(
                "Pass either the stacked `maps` or the three individual maps, not both."
            )
        if maps.shape[0] != 3:
            raise ValueError(
                f"Expected a leading channel axis of size 3, got shape {maps.shape}."
            )
        foreground, center_distances, boundary_distances = maps
    elif foreground is None or center_distances is None or boundary_distances is None:
        raise ValueError(
            "Pass either the stacked `maps` or all three of `foreground`,"
            + " `center_distances` and `boundary_distances`."
        )

    if foreground_smoothing > 0:
        foreground = gaussian_filter(foreground, foreground_smoothing)
    if distance_smoothing > 0:
        center_distances = gaussian_filter(center_distances, distance_smoothing)
        boundary_distances = gaussian_filter(boundary_distances, distance_smoothing)

    fg_mask = foreground > foreground_threshold
    marker_map = np.logical_and(
        center_distances < center_distance_threshold,
        boundary_distances < boundary_distance_threshold,
    )
    marker_map[~fg_mask] = False
    markers, _ = connected_components(marker_map)

    seg = watershed(boundary_distances, markers=markers, mask=fg_mask)

    if min_size > 0:
        ids, sizes = np.unique(seg, return_counts=True)
        too_small = ids[(sizes < min_size) & (ids != 0)]
        if too_small.size:
            seg[np.isin(seg, too_small)] = 0

    return seg.astype(np.int64)
