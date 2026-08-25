import glob
import hashlib
import os
import zipfile
from pathlib import Path

from loguru import logger

from bioimageio.core import test_model
from bioimageio.spec import save_bioimageio_package


def _enable_zip64_writes() -> None:
    """Allow zip entries larger than 2 GB when packaging.

    ``save_bioimageio_package`` embeds the weights into the zip, but the ~3.45 GB
    SAM 3 checkpoint exceeds Python's 2 GB-per-entry ZIP limit. bioimageio's
    writer calls ``ZipFile.open(name, "w")`` without ``force_zip64=True``, which
    raises "File size too large, try using force_zip64". Force ZIP64 for every
    write-mode entry (harmless for small files) so the checkpoint fits.
    """
    _orig_open = zipfile.ZipFile.open

    def _open(self, name, mode="r", pwd=None, *, force_zip64=False):
        if mode == "w":
            force_zip64 = True
        return _orig_open(self, name, mode, pwd, force_zip64=force_zip64)

    zipfile.ZipFile.open = _open
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromFileDescr,
    AxisId,
    BatchAxis,
    BioimageioConfig,
    ChannelAxis,
    Config,
    FileDescr,
    Identifier,
    InputTensorDescr,
    ModelDescr,
    OutputTensorDescr,
    PytorchStateDictWeightsDescr,
    ReproducibilityTolerance,
    Sha256,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
    Version,
    WeightsDescr,
)

# `sam3` version installed in this environment (`pip show sam3`). Pinned so the
# exported model resolves `sam3_encoder_bioimageio.Sam3EncoderBioImageIO`
# against the same code. SAM 3 is installed from source (git+github).
SAM3_GIT = "git+https://github.com/facebookresearch/sam3.git"

# SAM 3 vision encoder geometry: 1008x1008 RGB in, 256-channel 72x72 map out.
IMG_SIZE = 1008
FEATURE_CHANNELS = 256
FEATURE_SIZE = IMG_SIZE // 14  # patch size 14 -> 72

# sha256 of the HuggingFace `facebook/sam3` checkpoint (sam3.pt, ~3.45 GB).
SAM3_CHECKPOINT_SHA256 = (
    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
)


def resolve_checkpoint() -> Path:
    """Locate the SAM 3 checkpoint to bundle.

    The HF weights are gated, so referencing the URL directly makes bioimage.io
    download-verify fail with 401 unless authenticated. We therefore point at a
    local copy: ``$SAM3_CHECKPOINT`` if set, otherwise the HuggingFace cache
    (populated by a prior ``hf auth login`` + download of ``facebook/sam3``).

    To publish a shareable package instead, swap the local path for
    ``HttpUrl("https://huggingface.co/facebook/sam3/resolve/main/sam3.pt")``.
    """
    env = os.environ.get("SAM3_CHECKPOINT")
    if env:
        return Path(env)
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt"
        )
    )
    if hits:
        return Path(hits[0])
    raise FileNotFoundError(
        "SAM 3 checkpoint not found. Set $SAM3_CHECKPOINT to a local sam3.pt, or "
        "run `hf auth login` (with access to facebook/sam3) and download it first."
    )


def create_environment_file_for_model(building_dir: Path) -> Path:
    """Write a conda environment.yaml pinning the deps needed to run the model.

    SAM 3 needs numpy<2 and is installed from source; the BFloat16 CUDA kernel it
    uses means the model is intended to run on a GPU. A plain ``pip install`` of
    the git repo only pulls SAM 3's *core* deps, but ``sam3.model_builder``
    imports several ``[train]``/``[notebooks]`` extras at module-load time
    (submitit, tensorboard, fvcore, fairscale, einops, ...), so those are listed
    explicitly. ``setuptools<81`` is required because SAM 3 still uses
    ``pkg_resources``, which setuptools removed in v81.
    """
    env_yaml = (
        "name: sam3-encoder\n"
        "channels:\n"
        "  - conda-forge\n"
        "  - nodefaults\n"
        "dependencies:\n"
        "  - python>=3.11\n"
        "  - pip\n"
        "  - pip:\n"
        "      - numpy>=1.24,<2\n"
        "      - setuptools<81\n"  # provides pkg_resources (removed in setuptools 81)
        # SAM 3 core deps
        "      - ftfy==6.1.1\n"
        "      - regex\n"
        "      - iopath>=0.1.10\n"
        "      - typing_extensions\n"
        "      - huggingface_hub\n"
        "      - tqdm\n"
        "      - timm>=1.0.17\n"
        "      - psutil\n"  # imported by sam3 but not declared as a dependency
        # imported by sam3.model_builder at load time (train/notebooks/dev extras)
        "      - einops\n"
        "      - hydra-core\n"
        "      - submitit\n"
        "      - tensorboard\n"
        "      - zstandard\n"
        "      - torchmetrics\n"
        "      - fvcore\n"
        "      - fairscale\n"
        "      - scipy\n"
        "      - scikit-image\n"
        "      - scikit-learn\n"
        "      - pycocotools\n"
        f"      - {SAM3_GIT}\n"
    )
    building_dir.mkdir(parents=True, exist_ok=True)
    env_file = building_dir / "environment.yaml"
    with open(env_file, "w", encoding="utf8") as outfile:
        outfile.write(env_yaml)
    return env_file


def sha256sum(path: Path) -> str:
    """Return the hex sha256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model_descr() -> ModelDescr:
    """Build the bioimage.io description of the SAM 3 vision encoder."""
    # Conda environment file pinning sam3 so the model works out of the box.
    env_file = create_environment_file_for_model(Path("output"))
    env_descriptor = FileDescr(source=env_file, sha256=Sha256(sha256sum(env_file)))

    return ModelDescr(
        name="SAM3-Image-Encoder",
        description=(
            "Vision encoder of Meta's SAM 3 (Segment Anything with Concepts). "
            "Maps a 1008x1008 RGB image to a 256-channel 72x72 feature map "
            "(the backbone 'vision_features'). Runs in BFloat16 on GPU."
        ),
        inputs=[
            InputTensorDescr(
                id=TensorId("input"),
                axes=[
                    BatchAxis(),
                    ChannelAxis(
                        channel_names=[
                            Identifier("red"),
                            Identifier("green"),
                            Identifier("blue"),
                        ]
                    ),
                    SpaceInputAxis(id=AxisId("y"), size=IMG_SIZE),
                    SpaceInputAxis(id=AxisId("x"), size=IMG_SIZE),
                ],
                test_tensor=FileDescr(source=Path("output/test_input.npy")),
                sample_tensor=FileDescr(source=Path("sample_input.png")),
                # No bioimage.io preprocessing: the wrapper normalises internally
                # (scale to [0, 1] + ImageNet mean/std). SAM 3's BFloat16 stack is
                # so input-sensitive that even the ~1e-5 float32 rounding gap
                # between numpy and bioimage.io normalisation blows up into ~1.0
                # output differences, so we keep a single normalisation code path.
            )
        ],
        outputs=[
            OutputTensorDescr(
                id=TensorId("vision_features"),
                axes=[
                    BatchAxis(),
                    ChannelAxis(
                        channel_names=[
                            Identifier(f"feature{i:03d}")
                            for i in range(FEATURE_CHANNELS)
                        ],
                        description="SAM 3 backbone vision_features channels",
                    ),
                    SpaceOutputAxis(id=AxisId("y"), size=FEATURE_SIZE),
                    SpaceOutputAxis(id=AxisId("x"), size=FEATURE_SIZE),
                ],
                test_tensor=FileDescr(source=Path("output/test_output.npy")),
            ),
        ],
        weights=WeightsDescr(
            pytorch_state_dict=PytorchStateDictWeightsDescr(
                # Local cached sam3.pt (see resolve_checkpoint). The gated HF URL
                # https://huggingface.co/facebook/sam3/resolve/main/sam3.pt can be
                # used instead once you host/authenticate access to it.
                source=resolve_checkpoint(),
                sha256=Sha256(SAM3_CHECKPOINT_SHA256),
                architecture=ArchitectureFromFileDescr(
                    # Bundle the thin Sam3EncoderBioImageIO wrapper so the model
                    # runs against a stock `sam3` install (no package patch).
                    source=Path("sam3_encoder_bioimageio.py"),
                    sha256=Sha256(sha256sum(Path("sam3_encoder_bioimageio.py"))),
                    callable=Identifier("Sam3EncoderBioImageIO"),
                ),
                pytorch_version=Version("2.11.0"),
                strict=False,
                dependencies=env_descriptor,
            ),
        ),
        config=Config(
            bioimageio=BioimageioConfig(
                reproducibility_tolerance=[
                    # SAM 3 runs in BFloat16 on GPU. With the contiguous-input fix
                    # in Sam3EncoderBioImageIO the output is reproducible across
                    # processes on the same GPU; this tolerance leaves headroom for
                    # BFloat16 rounding and float32 preprocessing differences.
                    ReproducibilityTolerance(
                        absolute_tolerance=0.05,
                        relative_tolerance=0.01,
                        mismatched_elements_per_million=1000,
                    )
                ]
            )
        ),
    )


if __name__ == "__main__":
    logger.enable("bioimageio")

    os.chdir(Path(__file__).parent)

    descr = build_model_descr()

    # Test the description directly (no packaging needed). Tolerances match the
    # ReproducibilityTolerance in the model config.
    summary = test_model(
        descr,
        working_dir=Path("output/export_test"),
        absolute_tolerance=0.05,
        relative_tolerance=0.01,
    )
    summary.display()

    # Package to a zip. This embeds the ~3.45 GB checkpoint, so the resulting
    # archive is large; the shim above lets the >2 GB entry through.
    _enable_zip64_writes()
    out = save_bioimageio_package(
        descr, output_path=Path("output/sam3_encoder_bioimageio.zip")
    )
    print(f"Packaged: {out}")
