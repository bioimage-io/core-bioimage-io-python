import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from typing import List, Tuple

import pytest
from loguru import logger

from bioimageio.core import Tensor
from bioimageio.core.common import PerMember


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python 3.10 or higher")
def test_gradio_backend():
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
        prepared: List[Tuple[str, GradioModelAdapter, PerMember[Tensor]]] = []
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
