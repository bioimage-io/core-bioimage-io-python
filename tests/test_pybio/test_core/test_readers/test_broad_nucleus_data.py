from pybio.core.readers.base import PyBioReader
from pybio.core.readers.broad_nucleus_data import BroadNucleusDataBinarized
from pybio.spec.nodes import Axes, MagicShapeValue, OutputArray


def test_broad_nucleus():
    reader = BroadNucleusDataBinarized(
        outputs=(
            OutputArray(
                name="raw",
                axes=Axes("bxy"),
                shape=MagicShapeValue.dynamic,
                data_type="float32",
                data_range=(0, float("inf")),
                halo=(0, 0),
            ),
            OutputArray(
                name="label",
                axes=Axes("bxy"),
                shape=MagicShapeValue.dynamic,
                data_type="bool",
                data_range=(0, 1),
                halo=(0, 0),
            ),
        )
    )
    assert isinstance(reader, PyBioReader)
    assert len(reader.shape) == 2
    one_slice_tuple = (slice(1, 4), slice(2, 4), slice(3, 4))
    fov = reader[one_slice_tuple, one_slice_tuple]
    assert len(fov) == 2
    assert fov[0].shape == (3, 2, 1)
    assert fov[1].shape == (3, 2, 1)
