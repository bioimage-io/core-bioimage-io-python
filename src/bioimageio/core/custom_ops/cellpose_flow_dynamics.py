"""
Built-in custom op: cellpose_flow_dynamics
==========================================

Decodes Cellpose / Cellpose-SAM model outputs into instance label images
using flow-dynamics integration and connected-component labelling.

Usage in rdf.yaml
-----------------
::

    postprocessing:
      - id: custom
        callable: cellpose_flow_dynamics
        kwargs:                      # all optional — defaults shown
          cellprob_threshold: 0.0
          flow_threshold: 0.4
          interp: true
          do_3D: false

Expected model outputs (in rdf.yaml declaration order):
  0 - flow_y   : vertical flow field   (H x W), float32
  1 - flow_x   : horizontal flow field (H x W), float32
  2 - cellprob : cell probability map  (H x W), float32, sigmoid-activated

Returns:
  labels : instance label image (H x W), int32
           0 = background, 1..N = individual object instances

References
----------
Stringer et al. (2021) "Cellpose: a generalist algorithm for cellular
segmentation." Nature Methods 18, 100-106.
https://doi.org/10.1038/s41592-020-01018-x

Pachitariu & Stringer (2022) "Cellpose 2.0: how to train your own model."
Nature Methods 19, 1634-1641.
https://doi.org/10.1038/s41592-022-01663-4

---
Two implementation styles are shown below — both are equivalent.
The class style is used as the actual export; the function style is
shown in comments as an alternative for contributors to follow.
---
"""

import numpy as np

# ---------------------------------------------------------------------------
# Style 1 — callable class  (kwargs → __init__, tensors → __call__)
# ---------------------------------------------------------------------------

class cellpose_flow_dynamics:
    """Cellpose flow-dynamics postprocessing as a callable class.

    Instantiated once with configuration kwargs; called once per image.

    Example::

        op = cellpose_flow_dynamics(cellprob_threshold=0.0, flow_threshold=0.4)
        labels = op(flow_y, flow_x, cellprob)
    """

    def __init__(
        self,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        interp: bool = True,
        do_3D: bool = False,
    ) -> None:
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.interp = interp
        self.do_3D = do_3D

    def __call__(self, *arrays: np.ndarray) -> np.ndarray:
        """Decode flow fields into instance labels.

        Args:
            *arrays: Model output tensors in rdf.yaml declaration order:
                arrays[0] = flow_y   (vertical flow field)
                arrays[1] = flow_x   (horizontal flow field)
                arrays[2] = cellprob (cell probability, sigmoid-activated)

        Returns:
            Integer label image (H x W), int32. 0 = background.
        """
        if len(arrays) < 3:
            raise ValueError(
                f"cellpose_flow_dynamics expects 3 output tensors "
                f"(flow_y, flow_x, cellprob), got {len(arrays)}."
            )

        flow_y, flow_x, cellprob = arrays[0], arrays[1], arrays[2]

        try:
            from cellpose import dynamics
        except ImportError as e:
            raise ImportError(
                "cellpose is required for cellpose_flow_dynamics. "
                "Install with: pip install cellpose"
            ) from e

        flows = np.stack([flow_y, flow_x], axis=0)  # (2, H, W)
        masks, *_ = dynamics.compute_masks(
            flows,
            cellprob,
            cellprob_threshold=self.cellprob_threshold,
            flow_threshold=self.flow_threshold,
            interp=self.interp,
            do_3D=self.do_3D,
        )
        return masks.astype(np.int32)


# ---------------------------------------------------------------------------
# Style 2 — factory function  (alternative, identical behaviour)
# ---------------------------------------------------------------------------
#
# def cellpose_flow_dynamics(
#     cellprob_threshold: float = 0.0,
#     flow_threshold: float = 0.4,
#     interp: bool = True,
#     do_3D: bool = False,
# ):
#     """Factory: called once with kwargs, returns per-image function."""
#     def run(*arrays: np.ndarray) -> np.ndarray:
#         flow_y, flow_x, cellprob = arrays[0], arrays[1], arrays[2]
#         from cellpose import dynamics
#         flows = np.stack([flow_y, flow_x], axis=0)
#         masks, *_ = dynamics.compute_masks(
#             flows, cellprob,
#             cellprob_threshold=cellprob_threshold,
#             flow_threshold=flow_threshold,
#             interp=interp,
#             do_3D=do_3D,
#         )
#         return masks.astype(np.int32)
#     return run
