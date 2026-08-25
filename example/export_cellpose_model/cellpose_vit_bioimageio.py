"""BioImage.IO-compatible Cellpose-SAM architecture (bundled with the model).

This module ships a fixed ``CPnetBioImageIO`` so the exported bioimage.io model runs
against a **stock, unmodified** cellpose install. It overrides only the three methods
of cellpose's own ``cellpose.vit.CPnetBioImageIO`` that are out of sync with current
cellpose / bioimageio.core:

* ``forward``          -> unpack the 2 tensors ``CPSAM.forward`` returns (it tried 3).
* ``load_state_dict``  -> accept the ``strict`` kwarg bioimageio.core passes in.
* ``load_state_dict``  -> guard the legacy ``output.2.weight`` key (cpsam uses ``out.*``).

The heavy base network (``CPSAM``) is imported unchanged from the installed cellpose
package pinned in ``environment.yaml``; only this thin subclass lives here, so no
modification of the cellpose package (nor an upstream PR) is required.
"""

import torch
from cellpose.vit import CPSAM


class CPnetBioImageIO(CPSAM):
    """CP-SAM subclass compatible with the BioImage.IO Spec (loads BMZ ``cpsam`` weights)."""

    def forward(self, x):
        output_tensor, style_tensor = super().forward(x)
        return output_tensor, style_tensor

    def load_model(self, filename, device=None):
        if (device is not None) and (device.type != "cpu"):
            state_dict = torch.load(filename, map_location=device, weights_only=True)
        else:
            self.__init__(self.nout)
            state_dict = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=True
            )
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict, strict=False, **kwargs):
        if (
            "output.2.weight" in state_dict
            and state_dict["output.2.weight"].shape[0] != self.nout
        ):
            for name in self.state_dict():
                if "output" not in name:
                    self.state_dict()[name].copy_(state_dict[name])
        else:
            super().load_state_dict(
                {name: param for name, param in state_dict.items()}, strict=False
            )
