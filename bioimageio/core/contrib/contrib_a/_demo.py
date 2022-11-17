import asyncio


async def hello(msg="Hello!"):
    print(msg)
    return msg


# async def main():
#     task = asyncio.create_task(meh())
#
#     out1 = await task
#     out2 = await task
#
#     print(out1, out2)
#
#
# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(main())
