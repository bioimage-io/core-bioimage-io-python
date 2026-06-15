import socket
import sys
from multiprocessing import Process
from pathlib import Path

import pytest
from loguru import logger


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python 3.10 or higher")
def test_gradio_backend():
    from bioimageio.core import load_model
    from bioimageio.core.digest_spec import get_test_input_sample
    from bioimageio.core.remote_backends.gradio.client import GradioModelAdapter
    from bioimageio.core.remote_backends.gradio.server import main as gradio_server_main

    sock = socket.socket()
    sock.bind(("", 0))
    _host, port = sock.getsockname()

    server_process = Process(target=gradio_server_main, kwargs={"port": port})
    server_process.start()

    server_url = f"http://localhost:{port}/"
    model_source = Path(__file__).parent / "affable-shark-local.zip"
    model = load_model(model_source, format_version="latest")
    sample = get_test_input_sample(model)

    logger.debug("connecting adapter to {}", server_url)
    adapter = GradioModelAdapter(model, server=server_url)

    adapter.load_model()
    _ = adapter.forward(sample.members)
    summary = adapter.test_model()
    assert summary is not None
    assert summary.status == "passed", summary.display()

    server_process.terminate()
    server_process.join()
