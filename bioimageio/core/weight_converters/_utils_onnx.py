from collections import defaultdict
from itertools import chain
from typing import DefaultDict, Dict

from bioimageio.spec.model.v0_5 import ModelDescr


def get_dynamic_axes(model_descr: ModelDescr):
    dynamic_axes: DefaultDict[str, Dict[int, str]] = defaultdict(dict)
    for d in chain(model_descr.inputs, model_descr.outputs):
        for i, ax in enumerate(d.axes):
            if not isinstance(ax.size, int):
                dynamic_axes[str(d.id)][i] = str(ax.id)

    return dynamic_axes
