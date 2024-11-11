from bioimageio.spec import load_description
from bioimageio.spec.model import v0_5


# TODO: don't just test with unet2d_nuclei_broad_model
def test_get_block_transform(unet2d_nuclei_broad_model: str):
    from bioimageio.core.axis import AxisId
    from bioimageio.core.common import MemberId
    from bioimageio.core.digest_spec import (
        get_block_transform,
        get_io_sample_block_metas,
    )

    model = load_description(unet2d_nuclei_broad_model)
    assert isinstance(model, v0_5.ModelDescr)
    block_transform = get_block_transform(model)

    ns = {
        (ipt.id, a.id): 1
        for ipt in model.inputs
        for a in ipt.axes
        if isinstance(a.size, v0_5.ParameterizedSize)
    }

    input_sample_shape = {
        MemberId("raw"): {
            AxisId("batch"): 3,
            AxisId("channel"): 1,
            AxisId("x"): 4000,
            AxisId("y"): 3000,
        }
    }

    _, blocks = get_io_sample_block_metas(
        model,
        input_sample_shape=input_sample_shape,
        ns=ns,
    )

    for ipt_block, out_block in blocks:
        trf_block = ipt_block.get_transformed(block_transform)
        assert out_block == trf_block
