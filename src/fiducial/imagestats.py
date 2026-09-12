"""Objective image statistics measured on every generated sample.

They are stored in the manifest next to the ground truth, so they are available at evaluation time without re-reading the images.
"""

from __future__ import annotations

import cv2
import numpy as np


def mean_luminance(image: np.ndarray) -> float:
    """Mean perceived brightness over the whole frame, in [0, 255]."""
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())


def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian: the standard cheap sharpness proxy.

    Low values means a high-frequency detail, which is what blur, heavy noise suppression or severe defocus all produce.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def marker_region_contrast(image: np.ndarray, corners: np.ndarray) -> float:
    """Standard deviation of intensity inside the marker's bounding box.

    A marker is a high-contrast binary pattern, so this should be large.

    It collapses when the marker is occluded, washed out by backlight or crushed by shadow.

    Args:
        image: BGR scene.
        corners: Ground-truth marker corners, shape (4, 2).

    Returns a standard deviation inside the box, or 0.0 if the box falls outside the frame.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    xs, ys = corners[:, 0], corners[:, 1]
    x0, y0 = max(0, int(xs.min())), max(0, int(ys.min()))
    x1, y1 = min(w, int(np.ceil(xs.max()))), min(h, int(np.ceil(ys.max())))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(gray[y0:y1, x0:x1].std())


def describe(image: np.ndarray, corners: np.ndarray) -> dict[str, float]:
    """Compute every statistic for one sample.

    Args:
        image: BGR scene.
        corners: Ground-truth marker corners, shape (4, 2).

    Returns a mapping of statistic name to value, ready to serialize into the manifest.
    """
    return {
        "mean_luminance": mean_luminance(image),
        "laplacian_variance": laplacian_variance(image),
        "marker_region_contrast": marker_region_contrast(image, corners),
    }
