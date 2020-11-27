try:
    from typing import OrderedDict
except ImportError:
    from typing import MutableMapping as OrderedDict

from pybio.core.protocols import Tensor


class BatchTransformation:
    def apply(self, tensor: OrderedDict[str, Tensor]) -> None:
        raise NotImplementedError


class TensorTransformation:
    def apply(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError
