"""OpenCV's ArUco detector.

Every number this project reports is a comparison against this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fiducial.scene import aruco_dictionary


@dataclass(frozen=True)
class Detection:
    """
    Attributes:
        marker_id: The decoded ArUco id.
        corners: Detected corners, shape (4, 2), in image coordinates.
    """

    marker_id: int
    corners: np.ndarray


def build_detector(params: cv2.aruco.DetectorParameters | None = None) -> cv2.aruco.ArucoDetector:
    """Construct the detector with the project's dictionary.

    Args:
        params: Optional detector parameters. Defaults to OpenCV's defaults,
            deliberately: tuning them per condition would turn the baseline into
            a moving target and make the degradation curve meaningless.

    Returns a configured ArucoDetector.
    """
    return cv2.aruco.ArucoDetector(aruco_dictionary(), params or cv2.aruco.DetectorParameters())


def detect(image: np.ndarray, detector: cv2.aruco.ArucoDetector) -> list[Detection]:
    """Run the classical detector on one frame.

    Args:
        image: BGR scene.
        detector: Detector built by `build_detector`.

    Returns every marker found, can be empty.
    """
    corners, ids, _rejected = detector.detectMarkers(image)
    if ids is None or len(ids) == 0:
        return []

    flat = np.asarray(ids).reshape(-1)
    return [
        Detection(marker_id=int(marker_id), corners=np.asarray(quad).reshape(4, 2))
        for quad, marker_id in zip(corners, flat, strict=True)
    ]


def find(detections: list[Detection], marker_id: int) -> Detection | None:
    """Return the detection matching `marker_id`, or None if it was missed."""
    for det in detections:
        if det.marker_id == marker_id:
            return det
    return None
