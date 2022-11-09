from typing import Dict
from compiled_ops import my_op


async def my_wf(a: int) -> Dict[str, str]:
    m = await my_op(a)
    return {"centered": m}


if __name__ == "__main__":
    print(my_wf(5))
