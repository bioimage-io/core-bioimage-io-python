import numpy as np
from bioimageio.core.transformations import SynchronizedPyBioTransformation

# TODO nd ?
# TODO random seed
class RandomAxisRotation(SynchronizedPyBioTransformation):
    def __init__(self, apply_to):
        super().__init__(apply_to)
        self.k = 0

    def set_next_state(self):
        self.k = np.random.randint(0, 4)

    # FIXME only works for 2d inputs
    def apply_to_chosen(self, array):
        return np.rot90(array, k=self.k)
