"""Rendering of undegraded marker scenes.

A scene is one printed sheet pasted onto a textured background.

The sheet itself is drawn per sequence: its ink, its paper and its four margins.
"""

from __future__ import annotations

import cv2
import numpy as np

from fiducial import config


def aruco_dictionary() -> cv2.aruco.Dictionary:
    """Return the project's ArUco dictionary.

    Kept in one place so detection and generation cannot disagree about which
    family is in use, which would silently produce a 0% recall run.
    """
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT_NAME))


def background(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Generate a textured background.

    Low-frequency blobs plus fine noise, which is a crude stand-in for walls, furniture edges and floor texture.

    Args:
        rng: Seeded generator.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns a BGR image of shape (height, width, 3).
    """
    coarse = rng.integers(60, 190, size=(height // 32 + 1, width // 32 + 1, 3), dtype=np.uint8)
    canvas = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 6)
    grain = rng.normal(0.0, 6.0, size=canvas.shape)
    return np.clip(canvas.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def draw_appearance(rng: np.random.Generator, side: int) -> config.PrintAppearance:
    """Draw one sheet's printed appearance.

    The four margins are drawn separately because there is no reason for a page to be cut evenly around the marker.

    Args:
        rng: Seeded generator.
        side: Marker side in pixels, which the margins are expressed against.

    Returns the appearance to render this sequence at.
    """
    band_low, band_high = config.SHEET_BAND_RANGE
    ink_low = int(rng.integers(config.INK_GRAY_RANGE[0], config.INK_GRAY_RANGE[1] + 1))
    ink_high = min(config.INK_GRAY_RANGE[1], ink_low + int(rng.integers(band_low, band_high + 1)))

    paper_floor = max(config.PAPER_GRAY_RANGE[0], ink_high + config.MIN_PRINT_CONTRAST)
    paper_low = int(rng.integers(paper_floor, config.PAPER_GRAY_RANGE[1] + 1))
    paper_high = min(
        config.PAPER_GRAY_RANGE[1], paper_low + int(rng.integers(band_low, band_high + 1))
    )

    low, high = config.PAPER_MARGIN_RANGE
    margins = [int(round(float(rng.uniform(low, high)) * side)) for _ in range(4)]
    return config.PrintAppearance(
        ink_low=ink_low,
        ink_high=ink_high,
        paper_low=paper_low,
        paper_high=paper_high,
        margin_left=margins[0],
        margin_top=margins[1],
        margin_right=margins[2],
        margin_bottom=margins[3],
    )


def _sheet_field(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """A smooth field in [0, 1] over one sheet, for the drift across it.

    Per-pixel noise would be sensor noise, which the low-light axis already covers, and it washes out under any blur.
    Uneven toner and uneven light both vary over centimetres, not pixels.

    Args:
        rng: Seeded generator.
        width: Sheet width in pixels.
        height: Sheet height in pixels.

    Returns a float array of shape (height, width) with values in [0, 1].
    """
    coarse = rng.random(size=(max(2, height // 24), max(2, width // 24)))
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(field, 0.0, 1.0)


def render(
    marker_id: int, rng: np.random.Generator, width: int | None = None, height: int | None = None
) -> tuple[np.ndarray, np.ndarray, config.PrintAppearance]:
    """Render one clean scene containing a single marker.

    Args:
        marker_id: ArUco id to draw.
        rng: Seeded generator, controls the appearance, background and placement.
        width: Frame width; defaults to `config.IMAGE_WIDTH`.
        height: Frame height; defaults to `config.IMAGE_HEIGHT`.

    Returns a tuple of (BGR scene, ground-truth corners of shape (4, 2) ordered clockwise from top-left, the appearance it was rendered at).

    Raises a ValueError if the sheet cannot fit inside the frame.
    """
    width = width or config.IMAGE_WIDTH
    height = height or config.IMAGE_HEIGHT
    side = config.MARKER_SIDE_PX
    look = draw_appearance(rng, side)

    canvas = background(rng, width, height)
    patch = cv2.aruco.generateImageMarker(aruco_dictionary(), marker_id, side)

    x_low, x_high = look.margin_left, width - side - look.margin_right
    y_low, y_high = look.margin_top, height - side - look.margin_bottom
    if x_high <= x_low or y_high <= y_low:
        raise ValueError(
            f"a {side} px marker with margins "
            f"({look.margin_left}, {look.margin_top}, {look.margin_right}, {look.margin_bottom}) "
            f"does not fit in a {width}x{height} frame"
        )
    x = int(rng.integers(x_low, x_high))
    y = int(rng.integers(y_low, y_high))

    sheet_w = look.margin_left + side + look.margin_right
    sheet_h = look.margin_top + side + look.margin_bottom
    drift = _sheet_field(rng, sheet_w, sheet_h)
    sheet = look.paper_low + drift * (look.paper_high - look.paper_low)

    inner = drift[
        look.margin_top : look.margin_top + side, look.margin_left : look.margin_left + side
    ]
    dark = look.ink_low + inner * (look.ink_high - look.ink_low)
    light = look.paper_low + inner * (look.paper_high - look.paper_low)
    sheet[look.margin_top : look.margin_top + side, look.margin_left : look.margin_left + side] = (
        np.where(patch > 127, light, dark)
    )

    sheet_bgr = cv2.cvtColor(np.clip(sheet, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    canvas[
        y - look.margin_top : y + side + look.margin_bottom,
        x - look.margin_left : x + side + look.margin_right,
    ] = sheet_bgr

    # The marker occupies pixels x .. x+side-1 inclusive, so the far corners sit at x+side-1, not x+side.
    # An off-by-one here is invisible in every visual check and shows up as a constant ~1 px corner error that masks the real
    # localization degradation the sweep is meant to measure.
    corners = np.array(
        [[x, y], [x + side - 1, y], [x + side - 1, y + side - 1], [x, y + side - 1]],
        dtype=np.float32,
    )
    return canvas, corners, look
