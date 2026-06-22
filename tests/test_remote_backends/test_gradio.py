import socket
import sys
from multiprocessing import Process

import pytest
from loguru import logger


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python 3.10 or higher")
def test_gradio_backend():
    from bioimageio.core import load_model
    from bioimageio.core.digest_spec import get_test_input_sample
    from bioimageio.core.remote_backends.gradio.client import GradioModelAdapter
    from bioimageio.core.remote_backends.gradio.server import main as gradio_server_main

    port = 7860
    try:
        server_process = Process(target=gradio_server_main, kwargs={"port": port})
    except OSError:
        sock = socket.socket()
        sock.bind(("", 0))
        _host, port = sock.getsockname()
        server_process = Process(target=gradio_server_main, kwargs={"port": port})

    server_process.start()

    try:
        server_url = f"http://localhost:{port}/"
        model = load_model(
            "affable-shark", format_version="latest", perform_io_checks=False
        )
        sample = get_test_input_sample(model)

        logger.debug("connecting adapter to {}", server_url)
        adapter = GradioModelAdapter(model, server=server_url)

        adapter.load()
        _ = adapter.forward(sample.members)
        summary = adapter.test()
        assert summary is not None
        assert summary.status == "passed", summary.display()
    finally:
        server_process.terminate()
        server_process.join()
