# todo: maybe only keep bioimageio.core.contrib.__main__ to avoid redundant code and multiple entry points?
import asyncio
from pathlib import Path

from bioimageio.core.contrib.utils import start_contrib_service

loop = asyncio.get_event_loop()
loop.create_task(start_contrib_service(Path(__file__).parent.stem))
loop.run_forever()
