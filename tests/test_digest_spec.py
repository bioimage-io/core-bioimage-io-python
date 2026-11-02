import numpy as np
import pytest

from bioimageio.core import AxisId, MemberId, Tensor
from bioimageio.core.digest_spec import (
    create_sample_for_model,
    transpose_inputs_for_model,
)
from bioimageio.core.stat_measures import SampleMean, Stat
from bioimageio.spec import load_model_description
from bioimageio.spec.model import v0_5


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


@pytest.fixture
def two_input_model():
    return v0_5.ModelDescr.model_construct(
        inputs=[
            v0_5.InputTensorDescr(
                id=MemberId(name),
                optional=optional,
                axes=[
                    v0_5.BatchAxis(),
                    v0_5.SpaceInputAxis(id=AxisId("x"), size=4),
                ],
            )
            for name, optional in [("required", False), ("optional_input", True)]
        ],
        outputs=[],
    )


def test_create_sample_allows_omitting_optional_input(two_input_model: v0_5.ModelDescr):
    array = np.arange(4).reshape(1, 4)
    stat: Stat = {SampleMean(member_id=MemberId("required")): 1.5}
    sample = create_sample_for_model(
        two_input_model, {MemberId("required"): array}, stat=stat, sample_id="sample"
    )
    assert sample.id == "sample"
    assert sample.stat is stat
    assert set(sample.members) == {MemberId("required")}
    assert sample.members[MemberId("required")].dims == ("batch", "x")
    np.testing.assert_array_equal(
        sample.members[MemberId("required")].to_numpy(), array
    )


@pytest.mark.parametrize(
    "members, message",
    [
        ([], "Missing non-optional model inputs"),
        (["optional_input"], "Missing non-optional model inputs"),
        (["required", "unknown"], "Got unexpected inputs"),
    ],
)
def test_create_sample_rejects_invalid_members(
    two_input_model: v0_5.ModelDescr, members: list[str], message: str
):
    with pytest.raises(ValueError, match=message):
        _ = create_sample_for_model(
            two_input_model, {MemberId(m): np.zeros((1, 4)) for m in members}
        )


def test_multiple_inputs_require_mapping(two_input_model: v0_5.ModelDescr):
    with pytest.raises(TypeError, match="Expected `inputs` to be a mapping"):
        _ = create_sample_for_model(two_input_model, np.zeros((1, 4)))


def test_create_sample_rejects_missing_axis(two_input_model: v0_5.ModelDescr):
    with pytest.raises(ValueError, match="Missing axis 'x'"):
        _ = create_sample_for_model(
            two_input_model,
            {MemberId("required"): Tensor(np.zeros((1, 4)), dims=["batch", "y"])},
        )


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="transpose_inputs_for_model rejects omitted optional model inputs",
)
def test_transpose_allows_omitting_optional_input(two_input_model: v0_5.ModelDescr):
    tensor = Tensor(np.arange(4).reshape(4, 1), dims=["x", "batch"])
    transposed = transpose_inputs_for_model(
        {MemberId("required"): tensor}, two_input_model
    )
    assert set(transposed) == {MemberId("required")}
    assert transposed[MemberId("required")].dims == ("batch", "x")
    np.testing.assert_array_equal(
        transposed[MemberId("required")].to_numpy(), tensor.to_numpy().T
    )


@pytest.mark.parametrize("missing_required", [False, True])
def test_transpose_rejects_incompatible_inputs(
    two_input_model: v0_5.ModelDescr, missing_required: bool
):
    inputs = {MemberId("optional_input"): Tensor(np.zeros((2, 4)), dims=["batch", "x"])}
    if not missing_required:
        inputs[MemberId("required")] = Tensor(np.zeros((1, 4)), dims=["batch", "x"])
    with pytest.raises(
        ValueError,
        match="missing non-optional member"
        if missing_required
        else "incompatible batch lengths",
    ):
        _ = transpose_inputs_for_model(inputs, two_input_model)
