"""End-to-end sanity: what we render must be detectable when undegraded.

This is the test that catches a broken generator. If the clean render is not
detected at 100%, every degradation curve below it is measuring a rendering bug
rather than a detector limit, and the whole phase would be wasted.
"""

from __future__ import annotations

import numpy as np

from fiducial import detect, scene


def test_clean_render_is_detected_with_the_right_id():
    detector = detect.build_detector()
    for marker_id in (0, 7, 23, 49):
        image, _ = scene.render(marker_id, np.random.default_rng(marker_id))
        found = detect.detect(image, detector)
        assert detect.find(found, marker_id) is not None, f"clean marker {marker_id} was missed"


def test_clean_render_localizes_corners_tightly():
    """Corner error on a clean render bounds the best case for every later number."""
    from fiducial import metrics

    detector = detect.build_detector()
    image, corners = scene.render(11, np.random.default_rng(11))
    match = detect.find(detect.detect(image, detector), 11)
    assert match is not None
    # Tight on purpose. A 2 px tolerance is loose enough to hide a systematic
    # off-by-one in the ground-truth corner convention, which is exactly the bug
    # this bound now guards against.
    assert metrics.corner_rmse(match.corners, corners) < 0.5


def test_render_places_the_marker_inside_the_frame():
    image, corners = scene.render(3, np.random.default_rng(3))
    height, width = image.shape[:2]
    assert corners[:, 0].min() >= 0 and corners[:, 0].max() <= width
    assert corners[:, 1].min() >= 0 and corners[:, 1].max() <= height


def test_background_is_textured_not_flat():
    """A flat background would make the detector's quad search unrealistically easy."""
    bg = scene.background(np.random.default_rng(0), 160, 120)
    assert bg.std() > 5.0
