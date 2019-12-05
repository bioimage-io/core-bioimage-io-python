from pybio.readers.base import PyBioReader
from pybio.readers.broad_nucleus_data import BroadNucleusData


def test_broad_nucleus():
    reader = BroadNucleusData()
    assert isinstance(reader, PyBioReader)
    assert len(reader.shape) == 2
    one_slice_tuple = (slice(1, 4), slice(2, 4), slice(3, 4))
    fov = reader[one_slice_tuple, one_slice_tuple]
    assert len(fov) == 2
    assert fov[0].shape == (3, 2, 1)
    assert fov[1].shape == (3, 2, 1)
