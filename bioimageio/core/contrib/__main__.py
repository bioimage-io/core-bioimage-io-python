import argparse
import asyncio
from pathlib import Path

from bioimageio.core.contrib.utils import start_contrib_service

parser = argparse.ArgumentParser()
parser.add_argument("contrib_name", nargs="+")

args = parser.parse_args()

loop = asyncio.get_event_loop()
for contrib_name in args.contrib_name:
    loop.create_task(start_contrib_service(Path(__file__).parent.stem))

loop.run_forever()
