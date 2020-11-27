import numpy
import sklearn.ensemble

try:
    from typing import OrderedDict
except ImportError:
    from typing import MutableMapping as OrderedDict


class RandomForestClassifier:
    def __init__(self, **kwargs):
        self.clf = sklearn.ensemble.RandomForestClassifier(**kwargs)

    # def to_bc(self, *arrays: numpy.ndarray):
    #     return numpy.concatenate(
    #         [
    #             array.flatten()[:, None]
    #             if self.c_indices[i] is None
    #             else numpy.moveaxis(array, self.c_indices[i], -1).reshape((-1, array.shape[self.c_indices[i]]))
    #             for i, array in enumerate(arrays)
    #         ],
    #         axis=0,
    #     )
    #
    # def from_bc(self, bc_array: numpy.ndarray, original_arrays: Sequence[numpy.ndarray]):
    #     bc_arrays = numpy.split(bc_array, len(original_arrays))
    #     o_shapes_c_last = [list(oa.shape) for oa in original_arrays]
    #     [
    #         o_shape.append(o_shape.pop(self.c_indices[i]))
    #         for i, o_shape in enumerate(o_shapes_c_last)
    #         if self.c_indices[i] is not None
    #     ]
    #
    #     reshaped_c_last = [a.reshape(o_shape) for a, o_shape in zip(bc_arrays, o_shapes_c_last)]
    #     reshaped = [
    #         a if self.c_indices[i] is None else numpy.moveaxis(a, -1, self.c_indices[i])
    #         for i, a in enumerate(reshaped_c_last)
    #     ]
    #     return reshaped

    def __call__(self, raw: numpy.ndarray):
        assert isinstance(raw, numpy.ndarray)
        assert len(raw.shape) == 2
        assert raw.shape[1] == 1
        return self.clf.predict(raw)[:, None]

    def fit(self, raw: numpy.ndarray, target: numpy.ndarray):
        assert isinstance(raw, numpy.ndarray)
        assert len(raw.shape) == 2, raw.shape
        assert isinstance(target, numpy.ndarray)
        assert len(target.shape) == 2, target.shape
        self.clf.fit(raw, target)
