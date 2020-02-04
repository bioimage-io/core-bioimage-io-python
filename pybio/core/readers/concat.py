from typing import Any, Dict, Sequence, Tuple, Union

import numpy

from pybio.core.readers.base import PyBioReader
from pybio.spec import node, utils
from pybio.spec.node import OutputArray


class SimpleConcatenatedReader(PyBioReader):
    def __init__(
        self, readers: Sequence[Union[PyBioReader, node.Reader, Dict[str, Any]]], dims: Union[int, Sequence[int]] = 0
    ):
        """Concatenate the tensors of all readers along the dimension dims.

        The readers need to provide lists of tensors with the same axes and equal shape
        (except of the respective concatenation dimension)
        """
        assert len(readers) > 0
        reader_instances = []
        for r in readers:
            if isinstance(r, dict):
                reader_instances.append(
                    utils.get_instance(utils.load_spec_and_kwargs(uri=r["uri"], kwargs=r.get("kwargs", {})))
                )
            elif isinstance(r, node.Reader):
                reader_instances.append(utils.get_instance(r))
            elif isinstance(r, PyBioReader):
                reader_instances.append(r)
            else:
                raise TypeError(type(r))

        readers = reader_instances
        assert all(isinstance(r, PyBioReader) for r in readers), [type(r) for r in readers]
        axes = readers[0].axes
        assert all(r.axes == axes for r in readers)
        n_tensors = len(axes)
        if isinstance(dims, int):
            dims = [dims] * n_tensors
        else:
            assert n_tensors == len(dims), (n_tensors, dims)

        first_shape = readers[0].shape
        assert len(first_shape) == n_tensors
        concat_shape = [list(s) for s in first_shape]
        self.cumsums = []
        for i, (s, d) in enumerate(zip(first_shape, dims)):
            self.cumsums.append(numpy.cumsum([r.shape[i][d] for r in readers]))
            concat_shape[i][d] += sum(r.shape[i][d] for r in readers[1:])
            assert all(s[:d] == r.shape[i][:d] for r in readers[1:])
            assert all(s[d + 1 :] == r.shape[i][d + 1 :] for r in readers[1:])

        shape = [tuple(cs) for cs in concat_shape]
        self.readers = readers
        self.dims = dims

        outputs = [r.output for r in readers]
        name = ["/".join(set(out[i].name for out in outputs)) for i in range(n_tensors)]
        data_types = [set(out[i].data_type for out in outputs) for i in range(n_tensors)]
        assert all(len(dt) == 1 for dt in data_types)
        data_type = [dt.pop() for dt in data_types]

        data_ranges = [set(out[i].data_range for out in outputs) for i in range(n_tensors)]
        assert all(len(dr) == 1 for dr in data_ranges)
        data_range = [dr.pop() for dr in data_ranges]
        halo = [[0] * len(a) for a in axes]
        assert len(name) == len(axes) == len(data_type) == len(data_range) == len(shape) == len(halo), (
            name,
            axes,
            data_type,
            data_range,
            shape,
            halo,
        )
        super().__init__(
            output=tuple(
                OutputArray(name=n, axes=a, data_type=dt, data_range=dr, shape=s, halo=h)
                for n, a, dt, dr, s, h in zip(name, axes, data_type, data_range, shape, halo)
            )
        )

    def __getitem__(self, rois: Tuple[Tuple[slice, ...], ...]) -> Sequence[numpy.ndarray]:
        reader_start = 0
        this_rois = [list(roi) for roi in rois]
        parts = []
        for r, reader in enumerate(self.readers):
            for t, d in enumerate(self.dims):
                reader_stop = self.cumsums[t][r]
                slice_start = rois[t][d].start or 0
                slice_stop = rois[t][d].stop or reader_stop + reader_start
                this_roi_min = max(0, slice_start - reader_start)
                this_roi_max = min(reader_stop - reader_start, slice_stop - reader_start)
                this_rois[t][d] = slice(this_roi_min, this_roi_max)

            parts.append(reader[tuple(tuple(this_roi) for this_roi in this_rois)])

        return [numpy.concatenate([p[i] for p in parts], axis=d) for i, d in enumerate(self.dims)]
