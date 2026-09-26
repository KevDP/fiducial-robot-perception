"""Invariant guards for the degradation functions."""

from __future__ import annotations

import numpy as np
import pytest

from fiducial import config, degrade


@pytest.fixture
def scene_and_corners():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    corners = np.array([[40, 30], [100, 30], [100, 90], [40, 90]], dtype=np.float32)
    return image, corners


def _tone_for(axis: str) -> dict[str, int]:
    """Supply the occluder tone the occlusion axis requires and no other accepts."""
    return {"occluder_gray": 55} if axis == "occlusion" else {}


@pytest.mark.parametrize("axis", list(config.SWEEP))
def test_level_zero_is_a_no_op(axis, scene_and_corners):
    """Level 0.0 must produce the clean sample bit for bit.

    If it did not, the undegraded end of every curve would sit on a slightly
    different image than the clean sample, and the curves would not share an origin.
    """
    image, corners = scene_and_corners
    out_image, out_corners = degrade.apply(
        image, corners, axis, 0.0, np.random.default_rng(0), **_tone_for(axis)
    )
    assert np.array_equal(out_image, image)
    assert np.array_equal(out_corners, corners)


@pytest.mark.parametrize("axis", list(config.SWEEP))
def test_nonzero_level_changes_the_image(axis, scene_and_corners):
    """A degradation that silently does nothing would look like a robust detector."""
    image, corners = scene_and_corners
    level = max(config.SWEEP[axis])
    out_image, _ = degrade.apply(
        image.copy(), corners.copy(), axis, level, np.random.default_rng(0), **_tone_for(axis)
    )
    assert not np.array_equal(out_image, image)


def test_degradation_is_reproducible_given_the_same_seed(scene_and_corners):
    image, corners = scene_and_corners
    first, _ = degrade.apply(
        image.copy(), corners.copy(), "occlusion", 0.2, np.random.default_rng(5), occluder_gray=55
    )
    second, _ = degrade.apply(
        image.copy(), corners.copy(), "occlusion", 0.2, np.random.default_rng(5), occluder_gray=55
    )
    assert np.array_equal(first, second)


def test_occlusion_requires_an_explicit_tone(scene_and_corners):
    """A default tone is how the tone became an invisible constant the first time.

    Binarization sees a tone, not an object, so a caller that does not choose
    one is still choosing one, just without saying so.
    """
    image, corners = scene_and_corners
    with pytest.raises(ValueError, match="requires an explicit occluder_gray"):
        degrade.apply(image, corners, "occlusion", 0.1, np.random.default_rng(0))


def test_a_tone_is_rejected_on_the_axes_that_occlude_nothing(scene_and_corners):
    image, corners = scene_and_corners
    with pytest.raises(ValueError, match="meaningless on"):
        degrade.apply(image, corners, "low_light", 0.4, np.random.default_rng(0), occluder_gray=55)


def test_viewpoint_moves_the_ground_truth_corners(scene_and_corners):
    """The only geometric axis must transform labels with the pixels, or labels drift."""
    image, corners = scene_and_corners
    _, moved = degrade.apply(image, corners, "viewpoint", 45.0, np.random.default_rng(0))
    assert not np.allclose(moved, corners)


def test_photometric_axes_leave_corners_alone(scene_and_corners):
    image, corners = scene_and_corners
    for axis in ("low_light", "backlight", "warm_tint", "hard_shadow", "motion_blur"):
        _, out = degrade.apply(
            image, corners, axis, max(config.SWEEP[axis]), np.random.default_rng(0)
        )
        assert np.array_equal(out, corners), axis


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_occluder_is_not_clipped_to_the_marker(seed):
    """A real object covers the marker, the page around it, and keeps going.

    Clipping the occluder to the bounding box left the sheet's margin intact on
    all four sides at every coverage, so a quad could still be closed around the
    marker however deep the occluder went. Nothing physical covers the edge of a
    marker without covering the page beside it.
    """
    flat = np.full((120, 160, 3), 200, dtype=np.uint8)
    corners = np.array([[40, 30], [100, 30], [100, 90], [40, 90]], dtype=np.float32)
    out = degrade.occlude(flat, corners, 0.2, np.random.default_rng(seed), 55)

    changed = (out != flat).any(axis=2)
    beyond_the_marker = changed.copy()
    beyond_the_marker[30:90, 40:100] = False
    assert beyond_the_marker.any(), "the occluder stopped at the marker's bounding box"

    touches_an_edge = (
        changed[0, :].any() or changed[-1, :].any() or changed[:, 0].any() or changed[:, -1].any()
    )
    assert touches_an_edge, "the occluder floats inside the frame instead of entering it"


@pytest.mark.parametrize("gray", list(config.OCCLUDER_GRAY_LEVELS))
def test_the_occluder_is_painted_at_the_tone_it_was_given(gray):
    flat = np.full((120, 160, 3), 200, dtype=np.uint8)
    corners = np.array([[40, 30], [100, 30], [100, 90], [40, 90]], dtype=np.float32)
    out = degrade.occlude(flat, corners, 0.2, np.random.default_rng(0), gray)
    assert gray in np.unique(out), f"tone {gray} never reached the image"


def test_unknown_axis_is_rejected(scene_and_corners):
    image, corners = scene_and_corners
    with pytest.raises(ValueError, match="unknown degradation axis"):
        degrade.apply(image, corners, "sunlight", 0.5, np.random.default_rng(0))


def _fraction_darkened(before: np.ndarray, after: np.ndarray, corners: np.ndarray) -> float:
    """Fraction of the marker's bounding box whose intensity dropped.

    Measured on a uniform field rather than on a rendered marker: the marker's
    own black modules would sit at zero and stay at zero under a multiplicative
    shadow, undercounting the affected area by roughly half.
    """
    x0, y0, x1, y1 = degrade._bbox(corners)
    return float((after[y0:y1, x0:x1] < before[y0:y1, x0:x1]).mean())


@pytest.mark.parametrize("level", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_hard_shadow_darkens_the_stated_fraction_from_either_direction(level, seed):
    """The level must mean the same thing whichever edge the shadow enters from.

    Sharing one cut point between the two directions darkens `level` of the
    marker from one side and `1 - level` from the other, which produced a
    non-monotonic recall curve (33% at 0.3, 0% at 0.5, 39% at 0.7) that looked
    like a finding and was a generator bug.
    """
    flat = np.full((120, 160, 3), 200, dtype=np.uint8)
    corners = np.array([[40, 30], [100, 30], [100, 90], [40, 90]], dtype=np.float32)
    out = degrade.hard_shadow(flat, corners, level, np.random.default_rng(seed))
    fraction = _fraction_darkened(flat, out, corners)
    assert abs(fraction - level) < 0.05, f"level={level} darkened {fraction:.2f}"


@pytest.mark.parametrize("level", [0.05, 0.10, 0.20])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_occlusion_covers_the_stated_fraction_from_every_edge(level, seed):
    """Same invariant for occlusion, which enters from any of the four edges.

    The occluder now runs past the marker and out of the frame, so what it
    covers of the wider scene is no longer the level. What the level still has
    to mean, and what this pins, is the fraction of the marker it covers.
    """
    flat = np.full((120, 160, 3), 200, dtype=np.uint8)
    corners = np.array([[40, 30], [100, 30], [100, 90], [40, 90]], dtype=np.float32)
    out = degrade.occlude(flat, corners, level, np.random.default_rng(seed), 55)
    fraction = _fraction_darkened(flat, out, corners)
    assert abs(fraction - level) < 0.05, f"level={level} covered {fraction:.2f}"
