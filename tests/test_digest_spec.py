from bioimageio.spec import load_model_description


def test_get_block_transform(unet2d_nuclei_broad_model: str):
    from bioimageio.core.axis import AxisId
    from bioimageio.core.block import LinearAxisTransform
    from bioimageio.core.common import MemberId
    from bioimageio.core.digest_spec import get_block_transform

    model = load_model_description(
        unet2d_nuclei_broad_model, perform_io_checks=False, format_version="latest"
    )
    block_transform = get_block_transform(model)

    assert block_transform == {
        MemberId("probability"): {
            AxisId("batch"): LinearAxisTransform(
                member=MemberId("raw"), axis=AxisId("batch"), scale=1, offset=0
            ),
            AxisId("channel"): 1,
            AxisId("x"): LinearAxisTransform(
                member=MemberId("raw"), axis=AxisId("x"), scale=1, offset=0
            ),
            AxisId("y"): LinearAxisTransform(
                member=MemberId("raw"), axis=AxisId("y"), scale=1, offset=0
            ),
        }
    }
