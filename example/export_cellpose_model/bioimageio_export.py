from pathlib import Path

from loguru import logger

from bioimageio.core import test_model
from bioimageio.spec import save_bioimageio_package
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromLibraryDescr,
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

if __name__ == "__main__":
    logger.enable("bioimageio")

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
                test_tensor=FileDescr(source=Path("test_input.npy")),
                sample_tensor=FileDescr(source=Path("sample_input.png")),
                pad=ConstantPadding(value=0),
                preprocessing=[
                    ScaleRangeDescr(
                        kwargs=ScaleRangeKwargs(
                            min_percentile=1.0,
                            max_percentile=99.0,
                            axes=[AxisId("batch"), AxisId("y"), AxisId("x")],
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
                test_tensor=FileDescr(source=Path("test_output.npy")),
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
                architecture=ArchitectureFromLibraryDescr(
                    callable=Identifier("Transformer"),
                    import_from="cellpose.vit_sam",
                ),
                pytorch_version=Version("2.10.0"),
                strict=False,
            ),
        ),
        config=Config(
            bioimageio=BioimageioConfig(
                reproducibility_tolerance=[
                    # adjust reproducibility tolerance to label image output
                    ReproducibilityTolerance(
                        relative_tolerance=0.0,
                        absolute_tolerance=0.0,
                        mismatched_elements_per_million=200,
                    )
                ]
            )
        ),
    )

    out = save_bioimageio_package(descr, output_path=Path("cpsam_bioimageio.zip"))

    summary = test_model(descr, working_dir=Path("export_test"))
    summary.display()
