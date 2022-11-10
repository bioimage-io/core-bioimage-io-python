import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple


def my_op(a: int) -> str:
    return f"{a:~^10}"


async def heavy_compute(p):
    Path(p).write_text("done")


def parallel_op(*, max_thread_workers: int) -> Tuple[str, str]:
    srcs = ("src1.txt", "src2.txt")
    dests = ("dest1.txt", "dest2.txt")
    with ThreadPoolExecutor(max_workers=max_thread_workers) as e:
        for src, dest in zip(srcs, dests):
            e.submit(shutil.copy, src, dest)

    return dests


if __name__ == "__main__":
    print(my_op(5))
