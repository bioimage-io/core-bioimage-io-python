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
        source: <path/to/cellpose_flow_dynamics.py>
        sha256: <sha256 of the file>
        kwargs:                      # all optional — defaults shown
          cellprob_threshold: 0.0
          flow_threshold: 0.4
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
"""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


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
        do_3D: bool = False,
    ) -> None:
        super().__init__()
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.do_3D = do_3D

    def __call__(self, *arrays: "NDArray[Any]") -> "NDArray[np.int32]":
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
            n = len(arrays)
            raise ValueError(
                f"cellpose_flow_dynamics expects 3 output tensors (flow_y, flow_x, cellprob), got {n}."
            )

        flow_y, flow_x, cellprob = arrays[0], arrays[1], arrays[2]

        try:
            from cellpose import dynamics  # pyright: ignore[reportMissingTypeStubs]
        except ImportError as e:
            raise ImportError(
                "cellpose is required for cellpose_flow_dynamics. Install with: pip install cellpose"
            ) from e

        flows: NDArray[Any] = np.stack([flow_y, flow_x], axis=0)  # (2, H, W)
        result: Any = dynamics.compute_masks(  # pyright: ignore[reportUnknownVariableType]
            flows,
            cellprob,
            cellprob_threshold=self.cellprob_threshold,
            flow_threshold=self.flow_threshold,
            do_3D=self.do_3D,
        )
        masks: NDArray[Any] = cast(NDArray[Any], result[0])
        return masks.astype(np.int32)
