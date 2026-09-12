"""Rendering of undegraded marker scenes.

A scene is one marker pasted onto a textured background.

Pasting markers on flat gray would make the detector's job unrealistically easy,
because most false-negative pressure in a real room comes from clutter competing
with the marker's quad contour.
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


def render(
    marker_id: int, rng: np.random.Generator, width: int | None = None, height: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Render one clean scene containing a single marker.

    Args:
        marker_id: ArUco id to draw.
        rng: Seeded generator, controls background and marker placement.
        width: Frame width; defaults to `config.IMAGE_WIDTH`.
        height: Frame height; defaults to `config.IMAGE_HEIGHT`.

    Returns a tuple of (BGR scene, ground-truth corners of shape (4, 2) ordered clockwise from top-left).
    """
    width = width or config.IMAGE_WIDTH
    height = height or config.IMAGE_HEIGHT
    side = config.MARKER_SIDE_PX

    canvas = background(rng, width, height)
    patch = cv2.aruco.generateImageMarker(aruco_dictionary(), marker_id, side)
    patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

    # ArUco needs white space around the pattern, and a marker flush against the border is a different
    # failure mode that this sweep is not trying to measure.
    margin = side // 2
    x = int(rng.integers(margin, width - side - margin))
    y = int(rng.integers(margin, height - side - margin))

    # The white quiet zone is part of the marker specification, not decoration.
    quiet = side // 8
    canvas[y - quiet : y + side + quiet, x - quiet : x + side + quiet] = 255
    canvas[y : y + side, x : x + side] = patch_bgr

    # The marker occupies pixels x .. x+side-1 inclusive, so the far corners sit at x+side-1, not x+side.
    # An off-by-one here is invisible in every visual check and shows up as a constant ~1 px corner error that masks the real
    # localization degradation the sweep is meant to measure.
    corners = np.array(
        [[x, y], [x + side - 1, y], [x + side - 1, y + side - 1], [x, y + side - 1]],
        dtype=np.float32,
    )
    return canvas, corners
