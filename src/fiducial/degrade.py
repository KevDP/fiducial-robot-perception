"""Controlled degradations applied to rendered marker scenes.

Every function takes the image, the ground-truth marker corners and a level in
the range used by the corresponding sweep in `config` file, returning the degraded image.

These are synthetic degradations. They approximate the failure modes of a camera in a dim, crowded indoor space,
but a real occlusion has texture and a real dim frame has sensor-specific noise.

Treat the curves produced from this data as a lower bound on difficulty and validate against recorded footage
before trustingabsolute numbers.
"""

from __future__ import annotations

import cv2
import numpy as np

# A person or a tray in front of a marker reads as a dark, low-texture blob.
OCCLUDER_GRAY = 55


def _bbox(corners: np.ndarray) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box of the marker as (x0, y0, x1, y1)."""
    xs, ys = corners[:, 0], corners[:, 1]
    return int(xs.min()), int(ys.min()), int(np.ceil(xs.max())), int(np.ceil(ys.max()))


def occlude(
    image: np.ndarray, corners: np.ndarray, level: float, rng: np.random.Generator
) -> np.ndarray:
    """Cover `level` of the marker area with an opaque object.

    The occluder enters from one of the four edges, chosen by `rng`, which is how
    a person walking past or a tray held in front actually clips a marker.

    Args:
        image: BGR scene.
        corners: Ground-truth marker corners, shape (4, 2).
        level: Fraction of marker area to cover, in [0, 1).
        rng: Seeded generator, so the edge choice is reproducible.

    Returns a new BGR image with the occluder generated.
    """

    if level <= 0.0:
        return image
    out = image.copy()
    x0, y0, x1, y1 = _bbox(corners)
    w, h = x1 - x0, y1 - y0
    side = int(rng.integers(0, 4))
    if side == 0:  # from the left
        out[y0:y1, x0 : x0 + int(w * level)] = OCCLUDER_GRAY
    elif side == 1:  # from the right
        out[y0:y1, x1 - int(w * level) : x1] = OCCLUDER_GRAY
    elif side == 2:  # from the top
        out[y0 : y0 + int(h * level), x0:x1] = OCCLUDER_GRAY
    else:  # from the bottom
        out[y1 - int(h * level) : y1, x0:x1] = OCCLUDER_GRAY
    return out


def low_light(image: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """

    The noise is the point. Simply scaling brightness down is recoverable by any contrast normalization,
    so a detector would look artificially robust.

    A real camera in a dim room raises gain, and the noise that comes with it is what actually destroys the marker's bit pattern.

    Args:
        image: BGR scene.
        level: Severity in [0, 1). 0.8 is a very dim room.
        rng: Seeded generator for the noise field.

    Returns a new BGR image dimmed and noisy.
    """
    if level <= 0.0:
        return image
    gain = 1.0 - 0.85 * level
    noise_sigma = 18.0 * level
    dim = image.astype(np.float32) * gain
    dim += rng.normal(0.0, noise_sigma, size=dim.shape)
    return np.clip(dim, 0, 255).astype(np.uint8)


def backlight(image: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Blow out one side of the frame, as a window behind the marker would.

    Args:
        image: BGR scene.
        level: Severity in [0, 1]. At 1.0 the bright edge is fully saturated.
        rng: Seeded generator, used to pick which side the light comes from.

    Returns a new BGR image with a saturating horizontal gradient.
    """

    if level <= 0.0:
        return image
    h, w = image.shape[:2]
    ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)
    if rng.integers(0, 2) == 0:
        ramp = ramp[::-1]
    gradient = np.tile(ramp, (h, 1))[:, :, None]
    lit = image.astype(np.float32) + gradient * (255.0 * level)
    return np.clip(lit, 0, 255).astype(np.uint8)


def warm_tint(image: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Shift white balance toward warm incandescent light.

    Args:
        image: BGR scene.
        level: Severity in [0, 1].
        rng: Unused; kept for a uniform signature across degradations.

    Returns a new BGR image with the blue channel suppressed and red lifted.
    """
    if level <= 0.0:
        return image
    out = image.astype(np.float32)
    out[:, :, 0] *= 1.0 - 0.45 * level  # blue down
    out[:, :, 2] *= 1.0 + 0.25 * level  # red up
    return np.clip(out, 0, 255).astype(np.uint8)


def hard_shadow(
    image: np.ndarray, corners: np.ndarray, level: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Unlike `low_light`, which dims everything uniformly, this splits the marker into a lit half and a dark half.
    Adaptive thresholding handles a global dim frame far better than it handles a strong local gradient, so this axis and
    the low-light axis are not redundant.

    Args:
        image: BGR scene.
        corners: Ground-truth marker corners, shape (4, 2).
        level: Fraction of the marker width that falls in shadow, in [0, 1).
        rng: Seeded generator, used to pick the shadow direction.

    Returns:
        A new BGR image with a hard-edged shadow across the marker.
    """
    if level <= 0.0:
        return image
    out = image.astype(np.float32)
    x0, _y0, x1, _y1 = _bbox(corners)
    width = int((x1 - x0) * level)

    # The cut is measured from whichever edge the shadow enters, so the darkened fraction of the marker is `level` in both directions.
    # Using one shared cut point would darken `level` from the left but `1 - level` from the right.

    if rng.integers(0, 2) == 0:
        out[:, : x0 + width] *= 0.35
    else:
        out[:, x1 - width :] *= 0.35
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(image: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Apply directional blur, as camera or robot motion would.

    Args:
        image: BGR scene.
        level: Kernel length in pixels. 0.0 disables.
        rng: Seeded generator, used to pick the blur direction.

    Returns a new BGR image, blurred along one direction.
    """
    if level <= 0.0:
        return image
    k = int(level) | 1  # force odd so the kernel has a center
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = float(rng.uniform(0, 180))
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return cv2.filter2D(image, -1, kernel)


def viewpoint(
    image: np.ndarray, corners: np.ndarray, level: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    This is the one degradation that moves the ground truth, so it returns the transformed corners alongside the image.
    Applying the same homography to both keeps labels and pixels in sync; recomputing corners separately is how label
    drift gets introduced.

    Args:
        image: BGR scene.
        corners: Ground-truth marker corners, shape (4, 2).
        level: Out-of-plane angle in degrees, in [0, 90).
        rng: Seeded generator, used to pick the rotation axis.

    Returns a tuple of (warped image, transformed corners).
    """
    if level <= 0.0:
        return image, corners
    h, w = image.shape[:2]
    shrink = 1.0 - np.cos(np.deg2rad(level))
    inset = shrink * w * 0.5
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if rng.integers(0, 2) == 0:  # rotate about the vertical axis
        dst = np.float32([[inset, 0], [w, 0], [w, h], [inset, h]])
    else:  # rotate about the horizontal axis
        inset_v = shrink * h * 0.5
        dst = np.float32([[0, inset_v], [w, inset_v], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (w, h), borderValue=(120, 120, 120))
    pts = corners.reshape(-1, 1, 2).astype(np.float32)
    moved = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    return warped, moved


def apply(
    image: np.ndarray, corners: np.ndarray, axis: str, level: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to one degradation axis.

    Args:
        image: BGR scene.
        corners: Ground-truth marker corners, shape (4, 2).
        axis: Axis name, one of the keys of `config.SWEEP`, or "clean".
        level: Level along that axis.
        rng: Seeded generator.

    Returns a tuple of (degraded image, ground-truth corners after the degradation).

    Raises a ValueError if `axis` is not a known degradation axis.
    """
    if axis == "clean" or level <= 0.0:
        return image, corners
    if axis == "occlusion":
        return occlude(image, corners, level, rng), corners
    if axis == "low_light":
        return low_light(image, level, rng), corners
    if axis == "backlight":
        return backlight(image, level, rng), corners
    if axis == "warm_tint":
        return warm_tint(image, level, rng), corners
    if axis == "hard_shadow":
        return hard_shadow(image, corners, level, rng), corners
    if axis == "motion_blur":
        return motion_blur(image, level, rng), corners
    if axis == "viewpoint":
        return viewpoint(image, corners, level, rng)
    raise ValueError(f"unknown degradation axis: {axis!r}")
