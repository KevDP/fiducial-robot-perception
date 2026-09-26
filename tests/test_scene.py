"""End-to-end sanity: what we render must be detectable when undegraded.

This is the test that catches a broken generator. If the clean render is not
detected at 100%, every degradation curve below it is measuring a rendering bug
rather than a detector limit, and the whole phase would be wasted.
"""

from __future__ import annotations

import numpy as np

from fiducial import config, detect, scene


def test_clean_render_is_detected_with_the_right_id():
    detector = detect.build_detector()
    for marker_id in (0, 7, 23, 49):
        image, _, _ = scene.render(marker_id, np.random.default_rng(marker_id))
        found = detect.detect(image, detector)
        assert detect.find(found, marker_id) is not None, f"clean marker {marker_id} was missed"


def test_clean_render_is_detected_at_every_printed_appearance():
    """The appearance draw must never be the thing that breaks a clean render.

    Ink, paper and the four margins are randomized per sequence, so a range that
    reaches too far produces sheets no detector can read. Those failures would
    land in the undegraded end of every axis and read as a fragile detector
    rather than as a generator drawing prints that do not exist.
    """
    detector = detect.build_detector()
    for seed in range(40):
        image, _, look = scene.render(seed % 50, np.random.default_rng(seed))
        found = detect.detect(image, detector)
        assert detect.find(found, seed % 50) is not None, (
            f"seed {seed} missed at ink={look.ink_low}-{look.ink_high} "
            f"paper={look.paper_low}-{look.paper_high} "
            f"margins=({look.margin_left}, {look.margin_top}, "
            f"{look.margin_right}, {look.margin_bottom})"
        )


def test_appearance_stays_inside_its_declared_ranges():
    for seed in range(200):
        look = scene.draw_appearance(np.random.default_rng(seed), config.MARKER_SIDE_PX)
        assert config.INK_GRAY_RANGE[0] <= look.ink_low <= look.ink_high
        assert look.ink_high <= config.INK_GRAY_RANGE[1]
        assert config.PAPER_GRAY_RANGE[0] <= look.paper_low <= look.paper_high
        assert look.paper_high <= config.PAPER_GRAY_RANGE[1]
        assert look.contrast >= config.MIN_PRINT_CONTRAST, (
            f"seed {seed} drew ink {look.ink_low}-{look.ink_high} and "
            f"paper {look.paper_low}-{look.paper_high}, a worst contrast of {look.contrast}"
        )


def test_the_sheet_is_a_range_of_dark_and_light_not_two_values():
    """Two flat values is still a constant, only a randomized one.

    Ink lays down unevenly and light falls unevenly, so both tones have to
    spread across the page. A sweep whose every marker is exactly two grays
    measures a print that does not exist, wherever those two grays sit.
    """
    image, corners, look = scene.render(5, np.random.default_rng(5))
    x0, y0 = int(corners[0][0]), int(corners[0][1])
    patch = image[y0 : y0 + config.MARKER_SIDE_PX, x0 : x0 + config.MARKER_SIDE_PX, 0]

    assert len(np.unique(patch)) > 2, "the marker renders at exactly two gray values"
    assert patch.min() >= look.ink_low and patch.max() <= look.paper_high
    dark = patch[patch <= look.ink_high]
    light = patch[patch >= look.paper_low]
    assert dark.size and light.size
    assert light.min() - dark.max() >= config.MIN_PRINT_CONTRAST


def test_the_page_margins_are_not_the_same_on_the_four_sides():
    """A uniform border is an assumption about how the page was cut and framed.

    Whatever it is, it is not the same on four sides for every sheet ever
    printed, and holding it fixed showed the detector one framing per sample.
    """
    seen = set()
    for seed in range(20):
        look = scene.draw_appearance(np.random.default_rng(seed), config.MARKER_SIDE_PX)
        sides = (look.margin_left, look.margin_top, look.margin_right, look.margin_bottom)
        seen.add(sides)
        assert len(set(sides)) > 1, f"seed {seed} drew a uniform border {sides}"
    assert len(seen) > 1, "every sequence drew the same border"


def test_clean_render_localizes_corners_tightly():
    """Corner error on a clean render bounds the best case for every later number."""
    from fiducial import metrics

    detector = detect.build_detector()
    image, corners, _ = scene.render(11, np.random.default_rng(11))
    match = detect.find(detect.detect(image, detector), 11)
    assert match is not None
    # Tight on purpose. A 2 px tolerance is loose enough to hide a systematic
    # off-by-one in the ground-truth corner convention, which is exactly the bug
    # this bound now guards against.
    assert metrics.corner_rmse(match.corners, corners) < 0.5


def test_render_places_the_marker_inside_the_frame():
    image, corners, _ = scene.render(3, np.random.default_rng(3))
    height, width = image.shape[:2]
    assert corners[:, 0].min() >= 0 and corners[:, 0].max() <= width
    assert corners[:, 1].min() >= 0 and corners[:, 1].max() <= height


def test_background_is_textured_not_flat():
    """A flat background would make the detector's quad search unrealistically easy."""
    bg = scene.background(np.random.default_rng(0), 160, 120)
    assert bg.std() > 5.0
