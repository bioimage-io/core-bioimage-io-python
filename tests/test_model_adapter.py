from unittest.mock import Mock

import pytest
from dask.delayed import (
    Delayed,
    delayed,  # pyright: ignore[reportUnknownVariableType]
)

from bioimageio.core._model_adapter import LocalModelAdapter, ModelAdapter
from bioimageio.core._utils.type_hints import cast_and_assert


def test_lazy_forward_after_unload():
    adapter = Mock(spec=LocalModelAdapter)
    adapter._loaded = True
    adapter._model_queue = Mock()
    adapter._model_queue.get.return_value = ("cpu", object())
    adapter._forward_impl.return_value = []
    output = cast_and_assert(
        Delayed,
        delayed(
            LocalModelAdapter._forward_arrays  # pyright: ignore[reportPrivateUsage]
        )(adapter, []),
    )

    assert output.compute() == []
    adapter._model_queue.put.assert_called_once_with(
        adapter._model_queue.get.return_value
    )

    ModelAdapter.unload(adapter)
    adapter._model_queue.get.reset_mock()
    # Fail immediately rather than hang if the unloaded adapter accesses the queue.
    adapter._model_queue.get.side_effect = AssertionError("Accessed unloaded queue")
    with pytest.raises(RuntimeError, match="Compute lazy outputs inside"):
        output.compute()

    adapter._model_queue.get.assert_not_called()
