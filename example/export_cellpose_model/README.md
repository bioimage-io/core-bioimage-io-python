# Cellpose-SAM

A generalist deep-learning model for 2D cellular instance segmentation. Cellpose-SAM adapts the pretrained Vision Transformer backbone of the Segment Anything Model (SAM) into the Cellpose framework, producing a foundation model that segments cells and nuclei across a very broad range of imaging modalities, cell types and organisms without retraining.

This repository packages the original Cellpose-SAM weights (`cpsam`) as a [BioImage Model Zoo](https://bioimage.io) model so it can be run directly from BioImage.IO-compatible software.

# Table of Contents

- [Model Details](#model-details)
- [Uses](#uses)
- [Task Details](#task-details)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Technical Specifications](#technical-specifications)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)
- [Reproducing this BioImage.IO export](#reproducing-this-bioimageio-export)

# Model Details

## Model Description

Cellpose-SAM replaces the convolutional backbone used in earlier Cellpose releases with the ViT-L transformer encoder from Meta's Segment Anything Model, fine-tuned inside the Cellpose spatial-flow segmentation framework. The network predicts spatial flow fields and a cell-probability map; these are converted into a labelled instance mask by the Cellpose flow-dynamics post-processing. The authors report that the model substantially exceeds inter-human annotation agreement and approaches the human-consensus bound, and that it is explicitly trained to be robust to channel shuffling, cell size, shot noise, downsampling, and isotropic/anisotropic blur.

- **Developed by:** Marius Pachitariu, Michael Rariden and Carsen Stringer (Stringer & Pachitariu labs, HHMI Janelia Research Campus)
- **Shared by:** Packaged for the BioImage Model Zoo by Daniel Franco-Barranco
- **Model type:** Vision Transformer (SAM ViT-L backbone) within the Cellpose segmentation framework
- **Modality:** 2D light microscopy — fluorescence, brightfield, phase contrast, and other cellular imaging
- **License:** [BSD-3-Clause](https://github.com/MouseLand/cellpose/blob/main/LICENSE) (Cellpose); SAM backbone originally released under Apache-2.0
- **Finetuned from model:** Segment Anything Model (SAM), ViT-L image encoder

## Model Sources

- **Repository:** https://github.com/MouseLand/cellpose
- **Paper:** Pachitariu, Rariden & Stringer, *Cellpose-SAM: superhuman generalization for cellular segmentation*, bioRxiv 2025. DOI: [10.1101/2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001)
- **Weights:** https://huggingface.co/mouseland/cellpose-sam
- **Documentation:** https://cellpose.readthedocs.io

# Uses

## Direct Use

The model is a **generalist segmenter** intended to be used out of the box, without fine-tuning, on 2D microscopy images of cells or nuclei. It accepts up to three input channels (e.g. cytoplasm + nucleus, RGB histology, or a single grayscale channel replicated across channels) and returns an instance-label image where `0` is background and each connected object receives a unique integer ID.

Because it was trained with heavy channel and appearance augmentation, it generalizes across many microscope types, stains and organisms without the user needing to specify a "cytoplasm" vs "nucleus" channel order.

## Downstream Use

Cellpose-SAM fits into the wider Cellpose ecosystem and can be:

- fine-tuned or trained human-in-the-loop on domain-specific data (see the Cellpose GUI and `cellpose` training API),
- combined with Cellpose image-restoration models,
- used as the 2D engine for 3D segmentation via slice-wise stitching.

## Out-of-Scope Use

- **Not for clinical or diagnostic use.** The model is a research tool and has not been validated for medical decision-making.
- Not validated for imaging modalities far outside the training distribution (e.g. raw electron microscopy volumes) — validate before relying on it.
- Native inference here is **2D**; true 3D volumetric segmentation requires the Cellpose 3D pipeline rather than this packaged tile-based model.
- Always verify results on your own data before drawing quantitative conclusions.

# Task Details

- **Task type:** Instance segmentation
- **Input modality:** 2D light microscopy (fluorescence, brightfield, phase contrast, histology)
- **Target structures:** Cells and nuclei
- **Imaging technique:** Broadly modality-agnostic by design
- **Spatial resolution:** No fixed pixel size; the model is trained to be robust to cell size and to down-sampling. Very small or very large objects relative to the tile may benefit from rescaling.

# Bias, Risks, and Limitations

## Known Biases

- Training data, while diverse, is dominated by 2D light-microscopy of common model systems; rare or highly unusual morphologies may be under-represented.
- Performance still depends on image quality; extreme noise, artefacts or out-of-focus regions can degrade results.

## Limitations

- Objects that are heavily touching, overlapping or transparent remain challenging.
- The packaged model runs on fixed-size tiles (256×256) with a 16-pixel halo; very large images are processed tile-by-tile and objects spanning tile borders rely on the halo for continuity.
- Results depend on the flow-dynamics thresholds (`flow_threshold`, `cellprob_threshold`, `min_size`), which are fixed in this package but can be tuned in the native Cellpose API.

## Recommendations

- Validate on a representative subset of your own images.
- Where accuracy is critical, use manual verification or human-in-the-loop correction.
- For non-standard data, consider fine-tuning within Cellpose rather than relying on the generalist weights alone.

# Training Details

Full training methodology, datasets and hyperparameters are described in the original paper and repository. In brief:

- **Backbone:** SAM ViT-L image encoder, adapted into the Cellpose flow-field framework.
- **Augmentations (key contribution):** channel shuffling, cell-size variation, shot noise, down-sampling, and isotropic/anisotropic blur, to maximize generalization.
- **Objective:** Cellpose spatial-flow + cell-probability loss.
- **Framework:** PyTorch.

For exact dataset composition, splits, epochs and optimizer settings, refer to the [paper](https://doi.org/10.1101/2025.04.28.651001) and the [Cellpose repository](https://github.com/MouseLand/cellpose).

# Evaluation

The original authors evaluate Cellpose-SAM against inter-human annotation agreement and report that it **exceeds inter-human agreement and approaches the human-consensus segmentation bound** across a diverse benchmark of cellular images, outperforming previous Cellpose generations. Quantitative benchmarks (Average Precision at IoU thresholds and comparisons to prior models) are reported in the paper.

This BioImage.IO package is additionally checked for numerical reproducibility: the exported model's label output is compared against the reference Cellpose inference on the bundled sample image, within the tolerance declared in the model description (see `bioimageio_export.py`).

# Technical Specifications

## Model Architecture and Objective

- **Architecture:** SAM ViT-L transformer encoder within the Cellpose segmentation head (`CPSAM`); loaded here through the BioImage.IO-compatible subclass `CPnetBioImageIO`.
- **Objective:** Predicts spatial flow fields + cell-probability map, decoded into instances by Cellpose flow dynamics.

### Input specification

| Property | Value |
|----------|-------|
| Axes | `batch`, `channel`, `y`, `x` |
| Channels | 3 (`channel0`, `channel1`, `channel2`) |
| Tile size | 256 × 256 |
| Data type | float32 |
| Padding | constant, value `0` |
| Preprocessing | percentile scale-range normalization (1st–99th percentile over `batch, y, x`, `eps=1e-8`) |

### Output specification

| Property | Value |
|----------|-------|
| Tensor | `labels` — instance label image, `0` = background |
| Axes | `batch`, `channel`, `y`, `x` (16-pixel halo on `y` and `x`) |
| Postprocessing | Cellpose flow dynamics (`cellprob_threshold=0.0`, `flow_threshold=0.4`, `do_3D=False`, `min_size=15`) |

## Compute Infrastructure

### Hardware Requirements

- **Inference:** Runs on CPU; a CUDA GPU is strongly recommended for reasonable speed on large images.
- **Storage:** The `cpsam` weights are several hundred MB.

### Software Dependencies

- **Framework:** PyTorch (tested with 2.10.0)
- **Libraries:** `cellpose==4.2.1.1` (pinned in the packaged `environment.yaml`), `bioimageio.core`
- **BioImage.IO compatibility:** loadable via `bioimageio.core`; the weights are provided as a PyTorch state dict with a bundled architecture file, so no upstream Cellpose patch is required.

# How to Get Started with the Model

Using the BioImage.IO Python API:

```python
from bioimageio.core import load_description, predict
import numpy as np

model = load_description("cpsam_bioimageio.zip")  # or the bioimage.io model id

# image: (batch, channel=3, y, x) float array; single-channel data can be
# broadcast across the three channels.
image = np.load("test_input.npy")
labels = predict(model=model, inputs={"input": image})
```

To run the model natively (full Cellpose feature set, tuning, 3D, training), install Cellpose and follow the [official documentation](https://cellpose.readthedocs.io):

```bash
pip install "cellpose==4.2.1.1"
```

# Reproducing this BioImage.IO export

The scripts in this folder build and validate the BioImage.IO package from the original Cellpose-SAM weights:

1. `cellpose_original.py` — runs the original Cellpose model and saves matching input/output tensors used as BioImage.IO test tensors.
2. `bioimageio_export.py` — builds the `ModelDescr`, then exports and tests the packaged model.
3. `analyze_export.py` — compares Cellpose results against the BioImage.IO results in depth.
4. `cellpose_vit_bioimageio.py` — the bundled `CPnetBioImageIO` architecture that lets the exported model run against a stock, unmodified Cellpose install.

```console
python cellpose_original.py
python bioimageio_export.py
python analyze_export.py
```

# Citation

If you use this model, please cite the original work:

> Pachitariu, M., Rariden, M., & Stringer, C. (2025). *Cellpose-SAM: superhuman generalization for cellular segmentation.* bioRxiv. https://doi.org/10.1101/2025.04.28.651001

# License

This model is distributed under the [BSD-3-Clause license](https://github.com/MouseLand/cellpose/blob/main/LICENSE) of the Cellpose project. The SAM backbone from which it is fine-tuned was originally released under Apache-2.0.
