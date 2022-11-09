from typing import Dict

from ops import my_op


def my_wf(a: int) -> Dict[str, str]:
    m = my_op(a)
    return {"centered": m}


if __name__ == "__main__":
    print(my_wf(5))
