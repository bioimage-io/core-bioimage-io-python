from typing import Optional, Sequence

import numpy
import sklearn.ensemble


class RandomForestClassifier:
    def __init__(self, channel_indices: Sequence[Optional[int]], **kwargs):
        self.c_indices = list(channel_indices)
        self.clf = sklearn.ensemble.RandomForestClassifier(**kwargs)

    def to_bc(self, arrays: Sequence[numpy.ndarray]):
        return numpy.concatenate(
            [
                array.flatten()[:, None]
                if self.c_indices[i] is None
                else numpy.moveaxis(array, self.c_indices[i], -1).reshape((-1, array.shape[self.c_indices[i]]))
                for i, array in enumerate(arrays)
            ],
            axis=0,
        )

    def from_bc(self, bc_array: numpy.ndarray, original_arrays: Sequence[numpy.ndarray]):
        bc_arrays = numpy.split(bc_array, len(original_arrays))
        o_shapes_c_last = [list(oa.shape) for oa in original_arrays]
        [
            o_shape.append(o_shape.pop(self.c_indices[i]))
            for i, o_shape in enumerate(o_shapes_c_last)
            if self.c_indices[i] is not None
        ]

        reshaped_c_last = [a.reshape(o_shape) for a, o_shape in zip(bc_arrays, o_shapes_c_last)]
        reshaped = [
            a if self.c_indices[i] is None else numpy.moveaxis(a, -1, self.c_indices[i])
            for i, a in enumerate(reshaped_c_last)
        ]
        return reshaped

    def __call__(self, arrays: Sequence[numpy.ndarray]):
        bc_arrays = self.to_bc(arrays)
        bc_predictions = self.clf.predict(bc_arrays)
        return self.from_bc(bc_predictions, arrays)

    def fit(self, inputs: Sequence[numpy.ndarray], targets: Sequence[numpy.ndarray]):
        X_bc = self.to_bc(inputs)
        y_bc = self.to_bc(targets)
        self.clf.fit(X_bc, y_bc)
