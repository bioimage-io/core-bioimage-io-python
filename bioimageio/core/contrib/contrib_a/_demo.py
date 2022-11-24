import asyncio
from typing import Optional
import xarray as xr


async def hello(
    msg: str = "Hello!", tensor_a: Optional[xr.DataArray] = None, tensor_b: Optional[xr.DataArray] = None
) -> str:
    """dummy workflow printing msg

    This dummy workflow is intended as a demonstration and for testing.

    .. code-block:: yaml
    cite: [{text: BioImage.IO, url: "https://doi.org/10.1101/2022.06.07.495102"}]

    Args:
        msg: Message
        tensor_a: tensor_a whose shape is added to message
            axes: arbitrary
        tensor_b: tensor_b whose shape is added to message
            axes:
            - type: batch
            - type: space
              name: x
              description: x dimension
              unit: millimeter
              step: 1.5
            - type: index
              name: demo index
              description: a special index axis

    Returns:
        msg. A possibly manipulated message.
    """
    if tensor_a is not None:
        msg += f" tensor_a shape: {tensor_a.shape}"

    if tensor_a is not None:
        msg += f" tensor_a shape: {tensor_a.shape}"

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
