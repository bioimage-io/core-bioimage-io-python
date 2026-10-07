from __future__ import annotations

from typing import cast


def hex_to_rgb(value: str):
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(v * 2 for v in value)

    if len(value) not in (6, 8):
        raise ValueError(f"Invalid hex color: {value}")

    ret = tuple(int(value[i : i + 2], 16) for i in range(0, len(value), 2))
    assert len(ret) in (3, 4)
    return cast(tuple[int, int, int] | tuple[int, int, int, int], ret)


def get_default_channel_colors(n_channels: int) -> list[str]:
    """Get default channel colors for visualization purposes.

    - For < 8 channels: colorblind-friendly palette from https://www.nature.com/articles/nmeth.1618 (without black)
    - For < 21 channels: discrete matplotlib colormap 'tab20b' (redistributed for more even color distribution < 20 channels)
    - For >= 21 channels: sample colors from continuous matplotlib colormap 'cividis'

    Returns:
        List of hex color strings.
    """
    if n_channels < 8:
        # use colorblind-friendly palette from https://www.nature.com/articles/nmeth.1618
        # (without black)
        channel_colors = [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
        ][:n_channels]
    elif n_channels < 21:
        # use discrete matplotlib colormap 'tab20b'
        # (redistributed for more even color distribution < 20 channels)
        channel_colors = [
            "#393b79",
            "#8ca252",
            "#e7ba52",
            "#e7969c",
            "#7b4173",
            "#5254a3",
            "#b5cf6b",
            "#e7cb94",
            "#843c39",
            "#a55194",
            "#6b6ecf",
            "#cedb9c",
            "#8c6d31",
            "#d6616b",
            "#ce6dbd",
            "#9c9ede",
            "#637939",
            "#bd9e39",
            "#ad494a",
            "#de9ed6",
        ][:n_channels]
    else:
        # sample colors from continuous matplotlib colormap 'cividis'
        import matplotlib.colors
        import matplotlib.pyplot as plt

        cmap = plt.colormaps["cividis"].resampled(n_channels)
        channel_colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(n_channels)]

    return channel_colors
