"""BioImage.IO-compatible SAM 3 image encoder (bundled with the model).

This module ships a thin ``Sam3EncoderBioImageIO`` wrapper so the exported
bioimage.io model runs against a **stock, unmodified** ``sam3`` install (Meta's
"Segment Anything with Concepts", Nov 2025). It exposes just the vision encoder
of SAM 3 as a clean ``forward(x) -> tensor`` module that:

* builds the full SAM 3 image model with ``sam3.model_builder`` *without* any
  weights (bioimage.io loads the weights itself), then keeps only the shared
  vision-language ``backbone``;
* in ``forward`` calls ``backbone.forward_image`` and returns the top-level
  ``vision_features`` map, shape ``(B, 256, H/14, W/14)``;
* remaps the raw HuggingFace ``sam3.pt`` checkpoint (whose keys are prefixed
  ``detector.`` / ``tracker.``) onto this wrapper in ``load_state_dict``, the
  same remapping SAM 3's own ``_load_checkpoint`` does.

Important — precision: SAM 3's fused ``addmm_act`` kernel casts activations to
BFloat16 internally, so the backbone must run in BFloat16 (and, in practice, on
CUDA). ``forward`` therefore casts the incoming (float32) tensor to BFloat16,
runs the encoder, and casts ``vision_features`` back to float32 so the rest of
the bioimage.io pipeline stays in float32.

The heavy network is imported unchanged from the installed ``sam3`` package
pinned in ``environment.yaml``; only this thin wrapper lives here, so no
modification of the ``sam3`` package (nor an upstream PR) is required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sam3.model_builder import build_sam3_image_model


class Sam3EncoderBioImageIO(nn.Module):
    """SAM 3 vision encoder exposed as a BioImage.IO-compatible module.

    Parameters
    ----------
    bpe_path:
        Path to SAM 3's BPE vocabulary (``bpe_simple_vocab_16e6.txt.gz``).
        ``None`` (default) lets ``sam3`` resolve it from its package assets.
    """

    # ImageNet normalisation constants (SAM 3's ViT backbone operates on [0, 1]).
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, bpe_path: str | None = None) -> None:
        super().__init__()
        # Normalise inside forward (see forward) rather than via bioimage.io
        # preprocessing. Non-persistent buffers so they move with .to(device)
        # but stay out of the state dict.
        self.register_buffer(
            "mean", torch.tensor(self._MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(self._STD).view(1, 3, 1, 1), persistent=False
        )
        # Build the full image model with NO checkpoint (bioimage.io loads the
        # weights via load_state_dict). Disable the heavy detection/segmentation
        # and interactivity heads — only the vision backbone is needed here.
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cpu",
            eval_mode=True,
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=False,
            enable_inst_interactivity=False,
        )
        # Keep only the shared vision-language backbone. forward_image() uses
        # the vision half; the (unused) text half loads harmlessly from the
        # checkpoint's backbone.language_backbone.* keys.
        #
        # Cast to BFloat16: SAM 3's fused CUDA kernels run in BFloat16, so the
        # weights must be BFloat16 to match the BFloat16 activations (otherwise
        # the first conv hits "Input type CUDABFloat16 and weight type
        # cuda.FloatTensor should be the same"). bioimage.io then loads the
        # float32 checkpoint into these BFloat16 params, casting on copy — the
        # same rounding used to generate the reference test tensors.
        self.backbone = model.backbone.bfloat16()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SAM 3 vision encoder.

        Parameters
        ----------
        x : (B, 3, H, W) float tensor of raw RGB values in [0, 255].

        Returns
        -------
        (B, 256, H/14, W/14) float32 feature map (``vision_features``).
        """
        # Normalise here (scale to [0, 1] + ImageNet mean/std) instead of via
        # bioimage.io preprocessing. SAM 3's BFloat16 stack is chaotically
        # sensitive: a ~1e-5 difference in the input (e.g. float32 rounding
        # between numpy and bioimage.io's normalisation) explodes into ~1.0
        # output differences. Doing it here means the exact same torch code
        # produces the model input when generating the reference tensors and
        # when bioimage.io validates them, so there is no divergence to amplify.
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        # SAM 3's fused kernel requires BFloat16; cast in, cast the result back.
        # It is also sensitive to input memory layout — a non-contiguous input
        # shifts the BFloat16 output by up to ~1.0 — so force a contiguous tensor
        # to keep the result reproducible regardless of how the caller laid it out.
        out = self.backbone.forward_image(x.bfloat16().contiguous())
        return out["vision_features"].float()

    def load_state_dict(self, state_dict, strict: bool = False, **kwargs):
        """Load a raw SAM 3 checkpoint into this wrapper.

        The HuggingFace ``sam3.pt`` stores the whole model with ``detector.``
        (image model) and ``tracker.`` (video) prefixes. We keep the image
        model's ``backbone.*`` tensors — mirroring ``sam3``'s own
        ``_load_checkpoint`` — and drop everything else (transformer decoder,
        segmentation head, tracker) that this encoder-only wrapper does not have.
        """
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]

        remapped = {
            k.replace("detector.", ""): v
            for k, v in state_dict.items()
            if k.startswith("detector.")
        }
        remapped = {k: v for k, v in remapped.items() if k.startswith("backbone.")}
        return super().load_state_dict(remapped, strict=False)
