import numpy
import sklearn.ensemble

try:
    from typing import OrderedDict
except ImportError:
    from typing import MutableMapping as OrderedDict


class RandomForestClassifier:
    def __init__(self, **kwargs):
        self.clf = sklearn.ensemble.RandomForestClassifier(**kwargs)

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
