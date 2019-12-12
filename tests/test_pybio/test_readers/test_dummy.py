from pybio.readers.base import PyBioReader
from pybio.readers.dummy import DummyReader


def test_dummy_reader():
    reader = DummyReader()
    assert isinstance(reader, PyBioReader)
    assert len(reader.shape) == 2
    fov = reader[(slice(2, 5), slice(1, 3)), (slice(2, 5), )]
    assert fov[0].shape == (3, 2)
    assert fov[1].shape == (3, )
