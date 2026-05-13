from pathlib import Path

from bioimageio.core.utils._type_guards import is_list, is_ndarray


def test_cellpose_export(tmp_path: Path):
    """test case analog to the example in example/export_cellpose_model"""
    import cellpose.models
    import imageio
    import numpy as np

    from bioimageio.core import test_model
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

    sample_input_path = (
        Path(__file__).parent / "../example/export_cellpose_model/sample_input.png"
    )

    cellpose_original = cellpose.models.CellposeModel(gpu=False)

    input_array = imageio.imread(sample_input_path).transpose(2, 0, 1)
    assert is_ndarray(input_array)
    print("input:", input_array.shape)
    input_array = input_array[None, :, 256:512, 256:512]
    print("input roi:", input_array.shape)

    # 16 pixel halo to roughly match the 0.1*256 (tile size) tile overlap used in cellpose
    h = 16
    input_array = input_array[:, :, h:-h, h:-h]
    print("input cropped:", input_array.shape)

    # pad by 8 (additionally) before calling cellpose model (that pads another 8 internally)
    # so that cellpose input has 16 pixels padded (just like the bioimageio package we will export).
    p = 8
    input_array = np.pad(input_array, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    print("input padded:", input_array.shape)

    # add the cellpose internal padding of 8 pixels and save as test input for the bioimageio description
    input_array_padded = np.pad(
        input_array, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant"
    )
    print("test input padded:", input_array_padded.shape)
    np.save(tmp_path / "test_input.npy", input_array_padded)
    imageio.imwrite(
        tmp_path / "test_input.tiff", input_array_padded[0].transpose(1, 2, 0)
    )

    # run original cellpose model
    mask, flows, _ = cellpose_original.eval(input_array.transpose(0, 2, 3, 1))  # pyright: ignore[reportUnknownVariableType]
    assert is_ndarray(mask)
    assert is_list(flows)
    print("mask shape:", mask.shape)

    # pad output mask by 8 pixels to that were cropped internally by cellpose
    mask = np.pad(mask, ((p, p), (p, p)), mode="constant")

    # save output with batch and channel axes
    print("padded mask shape:", mask.shape)
    np.save(tmp_path / "test_output.npy", mask[None, None])
    imageio.imwrite(tmp_path / "test_output.tiff", mask.astype(np.uint16))

    # write out flows for debugging
    flows = flows[0]
    assert is_ndarray(flows)
    print("flows", flows.shape)
    np.save(tmp_path / "flows.npy", flows)
    imageio.imwrite(tmp_path / "flows.tiff", flows)

    ### bioimageio export
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
                test_tensor=FileDescr(source=tmp_path / "test_input.npy"),
                sample_tensor=FileDescr(source=sample_input_path),
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
                test_tensor=FileDescr(source=tmp_path / "test_output.npy"),
                sample_tensor=FileDescr(
                    source=sample_input_path.parent / "sample_output.png"
                ),
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
                        mismatched_elements_per_million=2000,
                    )
                ]
            )
        ),
    )

    summary = test_model(descr, working_dir=Path("output/export_test"))
    assert summary.status == "passed", summary.display()
