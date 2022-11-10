from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple

from ops import heavy_compute, my_op, parallel_op


def my_wf(a: int) -> Dict[str, str]:
    m = my_op(a)
    return {"centered": m}


def wf_with_p_ops(*, max_thread_workers: int, max_process_workers: int) -> Tuple[str, str]:
    with ProcessPoolExecutor(max_workers=max_process_workers) as e:
        e.submit(heavy_compute, "src1.txt")
        e.submit(heavy_compute, "src2.txt")

    return parallel_op(max_thread_workers=max_thread_workers)


if __name__ == "__main__":
    print(my_wf(5))
    print(wf_with_p_ops(max_thread_workers=2, max_process_workers=2))
