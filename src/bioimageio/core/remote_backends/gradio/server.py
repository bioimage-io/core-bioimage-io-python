from itertools import chain
from typing import (
    Any,
    Dict,
    Iterable,
    Literal,
    Optional,
    Union,
)

import gradio as gr
from loguru import logger

import bioimageio.core
from bioimageio.core import AxisId, Stat
from bioimageio.core.axis import PerAxis
from bioimageio.core.backends import create_model_adapter
from bioimageio.core.common import PerMember
from bioimageio.core.remote_backends.gradio.serializer import (
    DescriptionSerializer,
    GradioSampleSerializer,
    SerializedSampleBlock,
)
from bioimageio.spec import load_model_description
from bioimageio.spec.common import Sha256
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5

try:
    import spaces  # pyright: ignore
except ImportError:
    logger.warning("Failed to import 'spaces' package")

    class spaces:
        @staticmethod
        def GPU(func: Any):
            return func


logger.enable("bioimageio")

app = gr.Server()


@app.api(name="predict")  # pyright: ignore[reportUntypedFunctionDecorator]
@spaces.GPU
def predict(
    model: str,
    sha256: str,
    input_sample: Iterable[SerializedSampleBlock],
    blocksize: Optional[
        Union[int, Literal["blockwise_as_serialized"], PerMember[PerAxis[int]]]
    ] = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    skip_input_padding: bool = False,
    skip_output_cropping: bool = False,
    batch_size: Optional[int] = None,
) -> Iterable[SerializedSampleBlock]:
    """Run prediction on a sample

    Args:
        input_sample: Input sample as a sequence of serialized sample blocks.
             Use bioimageio.core.backends.gradio_backend.GradioModelAdapter.serialize_sample to create this from a Sample object.
        model: A model source: URL, nickname or base64 encoded model package (if len(model) > 2083).
        sha256: Sha256 hash of the model's bioimageio.yaml file at the model source or of the encoded model package.
        blocksize:
            - None (default): run non-blockwise, full-sample prediction.
            - integer: run blockwise prediction with a block size derived from the model and this blocksize parameter.
            - "blockwise_as_serialized": run blockwise prediction with the same blocking as the serialized input sample.
              (Non-blockwise pre- and postprocessing steps will be ignored.)
            - PerMember[PerAxis[int]]: run blockwise prediction with a fixed block shape given for each sample member.
        skip_preprocessing: If True, skip preprocessing steps defined in the model.
        skip_postprocessing: If True, skip postprocessing steps defined in the model.
        skip_input_padding: If True, skip input padding for non-blockwise prediction.
            Set this flag when predicting an (overlapping) sample block rather than a full sample.
        skip_output_cropping: If True, skip output cropping for non-blockwise prediction.
            Set this flag when predicting an (overlapping) sample block rather than a full sample.
        batch_size: Optional batch size only applicable to predicting input samples with batch dimension.
    """

    def setup(stat: Stat):
        model_adapter = _get_model_adapter(model, sha256=sha256)
        return bioimageio.core.create_prediction_pipeline(
            model_adapter.model_descr, fixed_dataset_statistics=stat
        )

    if blocksize == "blockwise_as_serialized":
        sample_block_iterator = iter(input_sample)
        deserialized_input_block = GradioSampleSerializer.deserialize_sample_block(
            next(sample_block_iterator)
        )
        pp = setup(deserialized_input_block.stat)
        for block in chain(
            [deserialized_input_block],
            (
                GradioSampleSerializer.deserialize_sample_block(b)
                for b in sample_block_iterator
            ),
        ):
            output_block = pp.predict_sample_block(
                block,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
            )
            yield GradioSampleSerializer.serialize_sample_block(output_block)
    else:
        deserialized_input_sample = GradioSampleSerializer.deserialize_sample(
            input_sample
        )
        pp = setup(deserialized_input_sample.stat)

        output_sample = None
        if isinstance(blocksize, int):
            try:
                if pp.has_non_blockwise_postprocessing and not skip_postprocessing:
                    output_sample = pp.predict_sample_with_blocking(
                        deserialized_input_sample,
                        skip_preprocessing=skip_preprocessing,
                        skip_postprocessing=skip_postprocessing,
                        ns=blocksize,
                        batch_size=batch_size,
                    )
                else:
                    for output in pp.predict_sample_with_blocking_yield_intermediates(
                        deserialized_input_sample,
                        skip_preprocessing=skip_preprocessing,
                        skip_postprocessing=skip_postprocessing,
                        ns=blocksize,
                        batch_size=batch_size,
                    )[1]:
                        # with purely blockwise postprocesssing or with postprocessing skipped,
                        # predicted blocks are part of the final result, so we yield them immediately.
                        yield GradioSampleSerializer.serialize_sample_block(
                            output.last_block
                        )

                    return

            except Exception as e:
                logger.warning(
                    "Falling back to full-sample prediction for model {}: {}",
                    pp.model_descr.id or pp.model_descr.name,
                    e,
                )
        if output_sample is None:
            output_sample = pp.predict_sample_without_blocking(
                deserialized_input_sample,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
                skip_input_padding=skip_input_padding,
                skip_output_cropping=skip_output_cropping,
            )

        if all(
            axes.get(AxisId("batch"), 1) > 1 for axes in output_sample.shape.values()
        ):
            # yield batches
            yield from GradioSampleSerializer.serialize_sample_with_fixed_blocking(
                output_sample,
                block_shapes={
                    m: {AxisId("batch"): batch_size or 1} for m in output_sample.shape
                },
                halo={},
            )
        else:
            yield from GradioSampleSerializer.serialize_sample(output_sample)


@app.api(name="load_model")  # pyright: ignore[reportUntypedFunctionDecorator]
def load_model(
    model: str,
    sha256: str,
) -> dict[Literal["message"], str]:
    """Load a model into the server's model cache. This can be used to pre-load a model before running predictions to avoid the overhead of loading the model during the first prediction request."""
    _ = _get_model_adapter(model, sha256=sha256)
    return {"message": "Model loaded successfully"}


@app.api(name="test_model")  # pyright: ignore[reportUntypedFunctionDecorator]
def test_model(
    model: str,
    sha256: str,
) -> str:
    """Run the bioimageio model test and return the validation summary. Returns None if testing failed."""
    model_adapter = _get_model_adapter(model, sha256=sha256)
    summary = bioimageio.core.test_model(model_adapter.model_descr)
    return summary.model_dump_json()


def _cache_key(kwargs: Dict[str, Any]) -> str:
    return kwargs["sha256"]


@gr.cache(  # pyright: ignore[reportUntypedFunctionDecorator]
    key=_cache_key,
    max_size=bioimageio.core.settings.gradio_server_model_cache_max_size,
    max_memory=bioimageio.core.settings.gradio_server_model_cache_max_memory,
    per_session=False,
)
def _get_model_adapter(
    model: str,
    *,
    sha256: str,
):
    """Get a model adapter for the given model

    Args:
        model: A model source: URL (len(model) <= 2083)) or model base64 encoded package bytes (len(model) > 2083).
        sha256: Sha256 hash of the model source at model URL or of the encoded model package bytes.
    """
    if not model:
        raise ValueError("Model source cannot be empty")

    model_descr = _get_model(model, sha256=sha256)
    return create_model_adapter(model_description=model_descr)


def _get_model(
    model: str,
    *,
    sha256: str,
) -> AnyModelDescr:
    if len(model) > 2083:
        ret = DescriptionSerializer.deserialize_from_string(model)
        if not isinstance(ret, (v0_4.ModelDescr, v0_5.ModelDescr)):
            raise ValueError(
                f"Deserialized model description is not a valid model description: got {ret.type} {ret.format_version}"
            )
        return ret
    else:
        return load_model_description(model, sha256=Sha256(sha256) if sha256 else None)


@app.get("/")
def root():
    return {
        "message": f"Running bioimageio.core {bioimageio.core.__version__} gradio server."
    }


def main(port: Optional[int] = None) -> str:
    _app, local_url, _share_url = app.launch(
        mcp_server=True, show_error=True, server_port=port
    )
    return local_url


if __name__ == "__main__":
    _ = main()
