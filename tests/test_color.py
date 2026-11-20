import pytest

from bioimageio.core._utils.color import get_default_channel_colors, hex_to_rgb


@pytest.mark.parametrize(
    "value,expected",
    [
        ("#f80", (255, 136, 0)),
        ("12ABef", (18, 171, 239)),
        ("#12345678", (18, 52, 86, 120)),
    ],
)
def test_hex_to_rgb(value: str, expected: tuple[int, ...]):
    assert hex_to_rgb(value) == expected


@pytest.mark.parametrize("value", ["", "#1234", "12345", "123456789", "gg0000"])
def test_invalid_hex_color(value: str):
    with pytest.raises(ValueError):
        _ = hex_to_rgb(value)


@pytest.mark.parametrize("count", [0, 1, 7, 8, 20, 21])
def test_channel_palette_boundaries(count: int):
    if count >= 21:
        _ = pytest.importorskip("matplotlib")
    colors = get_default_channel_colors(count)
    assert len(colors) == count
    assert len(set(colors)) == count
    assert all(len(hex_to_rgb(color)) == 3 for color in colors)
    if 0 < count < 8:
        assert colors[0] == "#E69F00"
    elif 8 <= count <= 20:
        assert colors[0] == "#393b79"
    elif count == 21:
        assert colors[0] == "#00224e"
        assert colors[-1] == "#fee838"
