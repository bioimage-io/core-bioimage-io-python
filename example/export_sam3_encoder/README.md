# SAM 3 image encoder → bioimage.io

Scripts to export the **SAM 3 image encoder** — the vision backbone of Meta's
["Segment Anything with Concepts"](https://github.com/facebookresearch/sam3)
(SAM 3, Nov 2025) — as a [bioimage.io](https://bioimage.io) model. The exported
model maps a `1008×1008` RGB image to the backbone `vision_features` map, shape
`(1, 256, 72, 72)`.

Modeled after [`../export_cellpose_model`](../export_cellpose_model).

## Setup

SAM 3 is **not** bundled with this example — clone and install it yourself
([facebookresearch/sam3](https://github.com/facebookresearch/sam3)):

```bash
git clone https://github.com/facebookresearch/sam3.git
pip install ./sam3
```

The checkpoint (`sam3.pt`, ~3.45 GB) is the gated HuggingFace `facebook/sam3`
weight; run `hf auth login` and download it first (see `sam3_original.py` for the
default cache path).

## Environment

`sam3` and `bioimageio.core` must be importable **together**, on Python ≥ 3.11 with a CUDA GPU. `bioimageio_export.py` regenerates `output/environment.yaml` with the full pinned list; the non-obvious pieces, all discovered the hard way:

* **`numpy<2`** — SAM 3 requires it; bioimage.io is happy with `numpy>=1.26`, so
  the two coexist.
* **`setuptools<81`** — SAM 3 still imports `pkg_resources`, which setuptools
  removed in v81.
* **SAM 3's load-time extras** — `sam3.model_builder` imports several `[train]` 
  `[notebooks]` extras at import time (`einops`, `submitit`, `tensorboard`,
  `fvcore`, `fairscale`, `torchmetrics`, `zstandard`, `hydra-core`, `psutil`,
  `pycocotools`, …). A plain `pip install git+…sam3` pulls none of them.

The visualization / video / eval / fast-kernel deps (`cv2`, `decord`,
`matplotlib`, `xformers`, `flash_attn`, `detectron2`, …) are **not** needed — the
image encoder never imports them on its path.

## How it works / key design decisions

* **Encoder-only wrapper.** `build_sam3_image_model(..., enable_segmentation=
  False, enable_inst_interactivity=False)` builds the full image model without
  weights; the wrapper keeps only `.backbone` and returns its top-level
  `vision_features`. `load_state_dict` strips the checkpoint's `detector.` prefix
  and keeps the `backbone.*` tensors (mirroring SAM 3's own `_load_checkpoint`).
* **BFloat16 on GPU.** SAM 3's fused `addmm_act` kernel forces BFloat16 (and, in
  practice, CUDA). The wrapper holds the backbone in BFloat16 so weights match
  activations; bioimage.io loads the float32 checkpoint into those BFloat16
  params, casting on copy — the same rounding used to make the reference tensors.
* **Normalisation lives in the wrapper, not in bioimage.io preprocessing.** SAM
  3's BFloat16 stack is chaotically input-sensitive: a ~1e-5 gap (e.g. float32
  rounding between numpy and bioimage.io's `scale_linear`+`normalize`) blows up
  into ~1.0 output differences over ~1% of elements — more than the bioimage.io
  reproducibility-tolerance caps can absorb. So the model declares **no**
  preprocessing; `forward` takes a raw `[0, 255]` RGB tensor and does the
  scale-to-[0,1] + ImageNet mean/std itself. The identical torch code then
  produces the model input both when generating the reference tensors and when
  bioimage.io validates them, so there is nothing to amplify.
* **Contiguous input.** The BFloat16 kernels are also layout sensitive (a
  non-contiguous input shifts the output by ~1.0), so `forward` forces a
  contiguous tensor. With these two fixes the encoder reproduces **bit-exactly**
  across processes on the same env + GPU.

## Input / output

| tensor | axes | notes |
| --- | --- | --- |
| `input` | `b, c(3: red/green/blue), y=1008, x=1008` | raw RGB in `[0, 255]`; scale-to-[0,1] + ImageNet mean/std happen inside the wrapper |
| `vision_features` | `b, c(256), y=72, x=72` | SAM 3 backbone top-level FPN feature map |

## Reproducing this export

With `sam3` installed and the checkpoint available (see [Setup](#setup)):

```console
python sam3_original.py     # run the original encoder, save reference tensors
python bioimageio_export.py # build, export and test the bioimage.io package
```

- `sam3_encoder_bioimageio.py` — the bundled `Sam3EncoderBioImageIO` wrapper the
  exported model runs against (stock, unmodified `sam3`).
- `bioimageio_export.py` writes the packaged model and `output/environment.yaml`.
