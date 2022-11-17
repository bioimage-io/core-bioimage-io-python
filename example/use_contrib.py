import asyncio

from bioimageio.core.contrib.contrib_a import hello as auto_hello
from bioimageio.core.contrib.contrib_a.local import hello as local_hello
from bioimageio.core.contrib.contrib_a.remote import hello as remote_hello


async def main():
    print(await auto_hello("auto hello"))
    print(await local_hello("local hello"))
    print(await remote_hello("remote hello"))

    print("auto func type:", type(auto_hello))
    print("local func type:", type(local_hello))
    print("remote func type:", type(remote_hello))


if __name__ == "__main__":
    # start up contrib_a service before with these two processes:
    # $python -m hypha.server --host=0.0.0.0 --port=9000
    # $python -m bioimageio.core.contrib.contrib_a
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
