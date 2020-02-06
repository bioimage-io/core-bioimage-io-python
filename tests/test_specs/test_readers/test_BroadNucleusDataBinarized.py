from pathlib import Path

import numpy

from pybio.core.readers.broad_nucleus_data import BroadNucleusDataBinarized
from pybio.spec import load_spec_and_kwargs, utils


def test_BroadNucleusDataBinarized(cache_path):
    spec_path = Path(__file__).parent / "../../../specs/readers/BroadNucleusDataBinarized.reader.yaml"
    pybio_reader = load_spec_and_kwargs(str(spec_path), kwargs={}, cache_path=cache_path)
    reader = utils.get_instance(pybio_reader)
    assert isinstance(reader, BroadNucleusDataBinarized)

    roi = ((slice(2), slice(4), slice(6)), (slice(3), slice(5), slice(7)))
    x, y = reader[roi]

    assert numpy.equal(
        x,
        [
            [
                [138.0, 130.0, 140.0, 141.0, 138.0, 139.0],
                [134.0, 134.0, 135.0, 137.0, 139.0, 136.0],
                [127.0, 132.0, 140.0, 137.0, 130.0, 140.0],
                [131.0, 134.0, 136.0, 135.0, 141.0, 141.0],
            ],
            [
                [133.0, 136.0, 134.0, 134.0, 130.0, 135.0],
                [125.0, 132.0, 136.0, 133.0, 132.0, 127.0],
                [135.0, 132.0, 137.0, 127.0, 138.0, 131.0],
                [135.0, 129.0, 133.0, 138.0, 139.0, 133.0],
            ],
        ],
    ).all()
    assert numpy.equal(y, numpy.zeros((3, 5, 7), dtype=bool)).all()
