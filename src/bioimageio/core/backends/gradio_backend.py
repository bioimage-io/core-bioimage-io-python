import tempfile
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from gradio_client import Client, handle_file
from numpy.typing import NDArray

from bioimageio.core.io import save_tensor
from bioimageio.spec import ModelDescr

from ..io import JsonValue, load_stat, serialize_stat
from ._model_adapter import ModelAdapter, Sample, SampleBlock

BlockRoiIO = dict[str, dict[str, tuple[int, int]]]


def normalize_suffix(suffix: str) -> str:
    normalized = suffix.strip().lower()
    if not normalized:
        return ".npy"
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


def serialize_sample(
    sample: Union[Sample, SampleBlock],
    out_suffix: str,
) -> tuple[dict[str, Path], BlockRoiIO, Sequence[JsonValue]]:
    tensor_paths: dict[str, Path] = {}
    rois: BlockRoiIO = {}
    for m in sample.members:
        with tempfile.NamedTemporaryFile(
            suffix=normalize_suffix(out_suffix), delete=False
        ) as tmp:
            if isinstance(sample, SampleBlock):
                t = sample.blocks[m].inner_data
                rois[str(m)] = {
                    str(k): (v.start, v.stop)
                    for k, v in sample.blocks[m].inner_slice.items()
                }
            else:
                t = sample.members[m]

            save_tensor(tmp.name, t)
            tensor_paths[str(m)] = Path(tmp.name)

    if isinstance(sample, SampleBlock):
        {k: v.inner_slice for k, v in sample.blocks.items()}

    output_stats = serialize_stat(sample.stat)
    return tensor_paths, rois, output_stats


class GradioModelAdapter(ModelAdapter):
    """Model adapter to use the bioimage-io-gradio-runner as a backend for model inference."""

    def __init__(
        self,
        model_description: ModelDescr,
        bioimageio_gradio_runner_url: str = "http://bioimage-io-bioimage-io-gradio-runner.hf.space",
    ):
        """Initialize the GradioModelAdapter.

        Note:
            - This adapter requires an environment with the same gradio version as the one used on the bioimage-io-gradio-runner server.

        Args:
            model_description: The model to run inference with.
            bioimageio_gradio_runner_url: The URL of a running bioimage-io-gradio-runner instance (default runner might not be availability/compatible).

        """
        super().__init__(model_description)
        self._client = Client(bioimageio_gradio_runner_url)

    def forward(self, input_sample: Union[Sample, SampleBlock]) -> Sample:

        tensor_paths, _, stat = serialize_sample(input_sample, out_suffix=".npy")
        input_tensors = [handle_file(tensor_paths[k]) for k in self._input_ids]

        job = self._client.submit(
            api_name="/predict",
            model="bioimage-io/affable-shark",
            input_tensors=input_tensors,
            input_dims=[",".join(map(str, a)) for a in self._input_axes],
            input_stats=stat,
            out_suffix=".npy",
            blocksize_parameter=10,
            # skip_preprocessing = True,
            # skip_postprocessing = True
        )
        n_blocks, outputs, output_rois, serialized_stats = job.result()
        print(n_blocks)
        for n_blocks, outputs, output_rois, serialized_stats in job:
            stats = load_stat(serialized_stats)

        return Sample()

    def _forward_impl(
        self, input_arrays: Sequence[Optional[NDArray[Any]]]
    ) -> Union[List[Optional[NDArray[Any]]], Tuple[Optional[NDArray[Any]], ...]]:
        raise NotImplementedError(
            "GradioModelAdapter does not support direct forwarding of arrays."
        )

    def unload(self):
        return super().unload()
