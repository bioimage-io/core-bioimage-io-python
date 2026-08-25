import hashlib
import os
from pathlib import Path

from loguru import logger

from bioimageio.core import test_model
from bioimageio.spec import save_bioimageio_package
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromFileDescr,
    AxisId,
    BatchAxis,
    BioimageioConfig,
    CellposeFlowDynamicsDescr,
    CellposeFlowDynamicsKwargs,
    ChannelAxis,
    Config,
    ConstantPadding,
    FileDescr,
    HttpUrl,
    Identifier,
    InputTensorDescr,
    ModelDescr,
    OutputTensorDescr,
    PytorchStateDictWeightsDescr,
    ReproducibilityTolerance,
    ScaleRangeDescr,
    ScaleRangeKwargs,
    Sha256,
    SizeReference,
    SpaceInputAxis,
    SpaceOutputAxisWithHalo,
    TensorId,
    Version,
    WeightsDescr,
)

# Cellpose version installed in this environment (`pip show cellpose`). Pinned so
# the exported model resolves `cellpose.vit.CPnetBioImageIO` against the same code.
CELLPOSE_VERSION = "4.2.1.1"


def create_environment_file_for_model(building_dir: Path) -> Path:
    """Write a conda environment.yaml pinning the deps needed to run the model.

    Mirrors BiaPy's `create_environment_file_for_model` so the bioimage.io
    `PytorchStateDictWeightsDescr.dependencies` field points at a reproducible env.
    """
    env_yaml = (
        "name: cellpose-sam\n"
        "channels:\n"
        "  - conda-forge\n"
        "  - nodefaults\n"
        "dependencies:\n"
        "  - python>=3.11\n"
        "  - pip\n"
        "  - pip:\n"
        f"      - cellpose=={CELLPOSE_VERSION}\n"
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

if __name__ == "__main__":
    logger.enable("bioimageio")

    os.chdir(Path(__file__).parent)

    # Conda environment file pinning cellpose (and bioimageio.core) so the model
    # works out of the box. Attached to the pytorch weights via `dependencies`.
    env_file = create_environment_file_for_model(Path("output"))
    env_descriptor = FileDescr(source=env_file, sha256=Sha256(sha256sum(env_file)))

    descr = ModelDescr(
        name="Cellpose-SAM",
        inputs=[
            InputTensorDescr(
                id=TensorId("input"),
                axes=[
                    BatchAxis(),
                    ChannelAxis(
                        channel_names=[
                            Identifier("channel0"),
                            Identifier("channel1"),
                            Identifier("channel2"),
                        ]
                    ),
                    SpaceInputAxis(id=AxisId("y"), size=256),
                    SpaceInputAxis(id=AxisId("x"), size=256),
                ],
                test_tensor=FileDescr(source=Path("output/test_input.npy")),
                sample_tensor=FileDescr(source=Path("sample_input.png")),
                pad=ConstantPadding(value=0),
                preprocessing=[
                    ScaleRangeDescr(
                        kwargs=ScaleRangeKwargs(
                            min_percentile=1.0,
                            max_percentile=99.0,
                            axes=[AxisId("batch"), AxisId("y"), AxisId("x")],
                            eps=1e-8,
                        ),
                    )
                ],
            )
        ],
        outputs=[
            OutputTensorDescr(
                id=TensorId("labels"),
                axes=[
                    BatchAxis(),
                    ChannelAxis(
                        channel_names=[Identifier("labels")],
                        description="Output labels, 0=background",
                    ),
                    SpaceOutputAxisWithHalo(
                        id=AxisId("y"),
                        size=SizeReference(
                            tensor_id=TensorId("input"),
                            axis_id=AxisId("y"),  # , offset=-16
                        ),
                        halo=16,
                    ),
                    SpaceOutputAxisWithHalo(
                        id=AxisId("x"),
                        size=SizeReference(
                            tensor_id=TensorId("input"),
                            axis_id=AxisId("x"),  # , offset=-16
                        ),
                        halo=16,
                    ),
                ],
                postprocessing=[
                    CellposeFlowDynamicsDescr(
                        kwargs=CellposeFlowDynamicsKwargs(
                            cellprob_threshold=0.0,
                            flow_threshold=0.4,
                            do_3D=False,
                            min_size=15,
                        )
                    )
                ],
                test_tensor=FileDescr(source=Path("output/test_output.npy")),
                sample_tensor=FileDescr(source=Path("sample_output.png")),
            ),
        ],
        weights=WeightsDescr(
            pytorch_state_dict=PytorchStateDictWeightsDescr(
                source=HttpUrl(
                    "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam"
                ),
                sha256=Sha256(
                    "e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2"
                ),
                architecture=ArchitectureFromFileDescr(
                    # Bundle the fixed CPnetBioImageIO so the model runs against a
                    # stock cellpose install (no package patch / upstream PR needed).
                    source=Path("cellpose_vit_bioimageio.py"),
                    sha256=Sha256(sha256sum(Path("cellpose_vit_bioimageio.py"))),
                    callable=Identifier("CPnetBioImageIO"),
                ),
                pytorch_version=Version("2.10.0"),
                strict=False,
                dependencies=env_descriptor,
            ),
        ),
        config=Config(
            bioimageio=BioimageioConfig(
                reproducibility_tolerance=[
                    # adjust reproducibility tolerance to label image output
                    ReproducibilityTolerance(
                        relative_tolerance=0.0,
                        absolute_tolerance=0.0,
                        mismatched_elements_per_million=2000,
                    )
                ]
            )
        ),
    )

    out = save_bioimageio_package(
        descr, output_path=Path("output/cpsam_bioimageio.zip")
    )

    summary = test_model(descr, working_dir=Path("output/export_test"), absolute_tolerance=1e-3, relative_tolerance=1e-3)
    summary.display()
