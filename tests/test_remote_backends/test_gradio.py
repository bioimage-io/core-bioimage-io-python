from __future__ import annotations

import multiprocessing
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from typing import Literal

import pytest
from loguru import logger

from bioimageio.core import Sample, Tensor, load_model
from bioimageio.core._description_serializer import DescriptionSerializer
from bioimageio.core.axis import AxisId, AxisInfo, AxisSize, PerAxis
from bioimageio.core.common import PerMember
from bioimageio.core.digest_spec import get_test_input_sample, get_test_output_sample
from bioimageio.core.remote_backends.gradio.client import GradioPredictionPipeline
from bioimageio.core.remote_backends.gradio.serializer import (
    GradioSampleSerializer,
)
from bioimageio.core.remote_backends.gradio.server import (
    main as gradio_server_main,
)
from bioimageio.core.remote_backends.gradio.server import predict as server_predict
from bioimageio.spec import AnyModelDescr


def test_gradio_sample_serializer(unet2d_diff_output_shape: str):
    model = load_model(
        unet2d_diff_output_shape, format_version="latest", perform_io_checks=False
    )
    sample = get_test_input_sample(model)
    serialized = GradioSampleSerializer.serialize_sample(sample)
    deserialized = GradioSampleSerializer.deserialize_sample(serialized)
    assert deserialized.members == sample.members
    assert deserialized == sample

    serialized_blockwise = GradioSampleSerializer.serialize_sample_blockwise(
        sample, model=model, blocksize_parameter=10
    )
    deserialized_blockwise = GradioSampleSerializer.deserialize_sample(
        serialized_blockwise
    )
    assert deserialized_blockwise.members == sample.members
    assert deserialized_blockwise == sample


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python 3.10 or higher")
def test_gradio_adapter():
    from bioimageio.core import load_model
    from bioimageio.core.digest_spec import get_test_input_sample
    from bioimageio.core.remote_backends.gradio.client import GradioModelAdapter
    from bioimageio.core.remote_backends.gradio.server import main as gradio_server_main

    port = 7860
    if _is_port_in_use(port):
        sock = socket.socket()
        sock.bind(("", 0))
        _host, port = sock.getsockname()

    server_process = Process(target=gradio_server_main, kwargs={"port": port})
    server_process.start()

    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    break
            except OSError:
                pass
            time.sleep(0.2)
        else:
            raise TimeoutError(f"gradio server did not become ready on port {port}")

        server_url = f"http://localhost:{port}/"
        prepared: list[tuple[str, GradioModelAdapter, PerMember[Tensor]]] = []
        for model_id in ("affable-shark", "ambitious-sloth"):
            model = load_model(
                model_id, format_version="latest", perform_io_checks=False
            )
            sample = get_test_input_sample(model)

            logger.debug("connecting adapter to {} for {}", server_url, model_id)
            adapter = GradioModelAdapter(model, server=server_url)
            prepared.append((model_id, adapter, sample.members))

        # Exercise concurrent requests pooled across both loaded models.
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_model_id = {
                executor.submit(adapter.forward, sample_members): model_id
                for model_id, adapter, sample_members in prepared
                for _ in range(2)
            }

        for future, model_id in future_to_model_id.items():
            assert future.result() is not None, model_id

        for model_id, adapter, _sample_members in prepared:
            summary = adapter.test()
            assert summary is not None
            assert summary.status == "passed", f"{model_id}: {summary.display()}"
    finally:
        server_process.terminate()
        server_process.join()


# @pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python 3.10 or higher")
# def test_gradio_pipeline():
#     from bioimageio.core import Sample, load_model
#     from bioimageio.core.axis import AxisId, AxisInfo, AxisSize
#     from bioimageio.core.digest_spec import (
#         get_test_input_sample,
#         get_test_output_sample,
#     )
#     from bioimageio.core.remote_backends.gradio.client import GradioPredictionPipeline
#     from bioimageio.core.remote_backends.gradio.server import main as gradio_server_main

#     port = 7860
#     if _is_port_in_use(port):
#         sock = socket.socket()
#         sock.bind(("", 0))
#         _host, port = sock.getsockname()

#     server_process = Process(target=gradio_server_main, kwargs={"port": port})
#     server_process.start()

#     try:
#         deadline = time.monotonic() + 30
#         while time.monotonic() < deadline:
#             try:
#                 with socket.create_connection(("localhost", port), timeout=1):
#                     break
#             except OSError:
#                 pass
#             time.sleep(0.2)
#         else:
#             raise TimeoutError(f"gradio server did not become ready on port {port}")

#         server_url = f"http://localhost:{port}/"
#         prepared: list[tuple[str, GradioPredictionPipeline, Sample, Sample]] = []
#         for model_id in ("affable-shark", "ambitious-sloth"):
#             model = load_model(
#                 model_id, format_version="latest", perform_io_checks=False
#             )
#             sample = get_test_input_sample(
#                 model,
#                 repetitions=[
#                     (
#                         2,
#                         AxisInfo(
#                             id=AxisId("extra_dim"),
#                             type="index",
#                             size=AxisSize(min=2, max=2),
#                         ),
#                         [1.0, 0.8],
#                     )
#                 ],
#             )

#             logger.debug(
#                 "connecting prediction pipeline to {} for {}", server_url, model_id
#             )
#             pipeline = GradioPredictionPipeline(model, server=server_url)
#             prepared.append((model_id, pipeline, sample, get_test_output_sample(model)))

#         # Exercise concurrent requests pooled across both loaded models.
#         with ThreadPoolExecutor(max_workers=4) as executor:
#             futures = [
#                 (
#                     model_id,
#                     executor.submit(pp.predict_sample_with_blocking, sample_in),
#                     expected_out,
#                 )
#                 for model_id, pp, sample_in, expected_out in prepared
#             ]
#             for model_id, future, expected_out in futures:
#                 actual_out = future.result(timeout=20)[{AxisId("extra_dim"): 0}]
#                 assert actual_out == expected_out  # refine sample comparison

#         pipeline.close()
#     finally:
#         server_process.terminate()
#         server_process.join(timeout=20)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python 3.10 or higher")
@pytest.mark.parametrize("with_blocking", [False, True])
def test_gradio_pipeline_direct(with_blocking: bool):
    def predict_direct(
        model_descr: AnyModelDescr,
        sample: Sample,
        *,
        blocksize: int
        | Literal["blockwise_as_serialized"]
        | PerMember[PerAxis[int]]
        | None = None,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        skip_input_padding: bool = False,
        skip_output_cropping: bool = False,
        batch_size: int | None = None,
    ) -> Sample:
        """helper to use server predict function directly without going through the gradio server process"""
        serialized_model, sha256 = DescriptionSerializer.serialize_to_string_and_hash(
            model_descr
        )
        serialized_input_sample = GradioSampleSerializer.serialize_sample(sample)
        return GradioSampleSerializer.deserialize_sample(
            server_predict(
                model=serialized_model,
                sha256=str(sha256),
                input_sample=serialized_input_sample,
                blocksize=blocksize,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
                skip_input_padding=skip_input_padding,
                skip_output_cropping=skip_output_cropping,
                batch_size=batch_size,
            )
        )

    for model_id in ("affable-shark", "ambitious-sloth"):
        model = load_model(model_id, format_version="latest", perform_io_checks=False)
        sample = get_test_input_sample(
            model,
            repetitions=[
                (
                    2,
                    AxisInfo(
                        id=AxisId("extra_dim"),
                        type="index",
                        size=AxisSize(min=2, max=2),
                    ),
                    [1.0, 0.8],
                )
            ],
        )
        expected_out = get_test_output_sample(model)

        direct_out = predict_direct(
            model,
            sample,
            blocksize=10 if with_blocking else None,
            batch_size=1 if with_blocking else None,
        )
        direct_out = direct_out.unstack_batch_multi_index(errors="ignore")[
            {AxisId("extra_dim"): 0}
        ]
        logger.info("direct prediction check for model {}", model_id)
        assert direct_out.members == expected_out.members, (
            direct_out.members,
            expected_out.members,
        )


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python 3.10 or higher")
@pytest.mark.parametrize("with_blocking", [False, True])
@pytest.mark.parametrize("with_server", [False, True])
def test_gradio_pipeline(
    with_blocking: bool,
    with_server: bool,
    shape_change_model: str,
    unet2d_expand_output_shape: str,
    unet2d_multi_tensor: str,
):
    port = 7860

    if with_server:
        if _is_port_in_use(port):
            with socket.socket() as s:
                s.bind(("", 0))
                _host, port = s.getsockname()

        server_process = multiprocessing.Process(
            target=gradio_server_main, kwargs={"port": port}
        )
        server_process.start()
    else:
        server_process = None

    prepared: list[tuple[str, GradioPredictionPipeline, Sample, Sample]] = []
    try:
        if with_server:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("localhost", port), timeout=1):
                        break
                except OSError:
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError(f"gradio server did not become ready on port {port}")

            server_url = f"http://localhost:{port}/"
        else:
            server_url = "run_in_client_process"

        for model_source in (
            shape_change_model,
            unet2d_expand_output_shape,
            unet2d_multi_tensor,
        ):
            model = load_model(
                model_source, format_version="latest", perform_io_checks=False
            )
            input_sample = get_test_input_sample(
                model,
                repetitions=[
                    (
                        2,
                        AxisInfo(
                            id=AxisId("extra_dim"),
                            type="index",
                            size=AxisSize(min=2, max=2),
                        ),
                        [1.0, 0.8],
                    )
                ],
            )
            expected_out = get_test_output_sample(model)
            logger.debug(
                "connecting prediction pipeline to {} for {}", server_url, model_source
            )
            pipeline = GradioPredictionPipeline(model, server=server_url)
            server_status = pipeline.get_status()
            logger.info("Server status for model {}: {}", model_source, server_status)
            prepared.append((model_source, pipeline, input_sample, expected_out))

        executor = ThreadPoolExecutor(max_workers=4)
        try:
            futures = [
                (
                    model_source,
                    (
                        executor.submit(pp.predict_with_blocking, sample_in)
                        if with_blocking
                        else executor.submit(pp.predict_without_blocking, sample_in)
                    ),
                    expected_out_loop,
                )
                for model_source, pp, sample_in, expected_out_loop in prepared
            ]
            for _model_source, future, expected_out_loop in futures:
                actual_out = future.result(timeout=20).unstack_batch_multi_index(
                    errors="ignore"
                )[{AxisId("extra_dim"): 0}]
                assert actual_out.members == expected_out_loop.members

        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    finally:
        logger.info("Closing prediction pipelines before shutting down gradio server")
        for model_source, pp, _, _ in prepared:
            try:
                pp.unload()
            except Exception:
                logger.exception(
                    "Failed to unload prediction pipeline for {}", model_source
                )

        logger.info("Terminating gradio server process on port {}", port)
        if server_process is not None and server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=10)

        if server_process is not None and server_process.is_alive():
            logger.warning(
                "Gradio server on port {} did not exit cleanly; force killing.", port
            )
            server_process.kill()
            server_process.join(timeout=10)

        assert server_process is None or not server_process.is_alive(), (
            "gradio server still alive after shutdown"
        )
        if server_process is not None:
            logger.info("gradio server exited with code {}", server_process.exitcode)
