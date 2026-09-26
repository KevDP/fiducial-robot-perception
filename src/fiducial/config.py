"""Shared configuration for dataset generation, detection and evaluation.

This module is imported by both the offline pipeline and (later) the serving node.
Any preprocessing constant that training and serving must agree on lives here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Image geometry
# ---------------------------------------------------------------------------
# 640x480 is the best practice for a low-cost robot camera streams.
# Generating at a higher resolution would flatter the detector for free and make the whole degradation curve optimistic.

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# ---------------------------------------------------------------------------
# Marker family
# ---------------------------------------------------------------------------
# DICT_4X4_50 has the coarsest bit grid of the standard dictionaries, so it is the most forgiving under blur and low resolution.

ARUCO_DICT_NAME = "DICT_4X4_50"

# Marker side length in pixels at the reference distance, before any viewpoint warp.

MARKER_SIDE_PX = 96

# ---------------------------------------------------------------------------
# Printed appearance
# ---------------------------------------------------------------------------
# Nuisance variables, drawn per sequence. They are not swept, because the
# question is never "how does recall depend on toner density": it is that a
# curve measured at one printed appearance does not transfer to another.
#
# The dictionary renders the pattern at 0 on 255, and the generator kept those
# two values. Nothing prints that way, and nothing prints at any other fixed
# pair either, so what these bounds have to cover is the spread: toner, paper
# stock, exposure and the evenness of the light all move where dark and light
# land. The bounds are wide on purpose and are not fitted to any measurement.
# A measurement would tell us where one sheet sat, and that is the mistake this
# replaces, not a target to aim at.

INK_GRAY_RANGE = (0, 110)
PAPER_GRAY_RANGE = (120, 255)

# Since ink lays down unevenly and light falls unevenly, each sequence draws a band for its dark
# and a band for its light, and the value drifts across the page inside it.
SHEET_BAND_RANGE = (0, 45)

# Without a floor gap, the generator can render a sheet unexpectedly, and this rendering failure
# would land in every axis and read as a fragile detection.
#
# Adaptive thresholding works locally, so it holds a clean render down to a gap of about 20.
# This leaves room above the boundary: a wider floor would be the generator refusing to draw
# prints that are perfectly readable.
MIN_PRINT_CONTRAST = 30
PAPER_MARGIN_RANGE = (0.1, 0.8)

# ---------------------------------------------------------------------------
# Degradation sweep
# ---------------------------------------------------------------------------

# One axis per sample, crossing axes would confound attribution: when recall drops we need to know which single condition caused it.
# Combined conditions belong to a later phase, once the per-axis curves exist.
# A "Level 0.0" always means "no degradation on this axis", so every axis shares a common origin and the curves are comparable.

OCCLUSION_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
"""Fraction of the marker area covered by an opaque object (a person, a carried object).

Spread across the range where the transition sits is what the sweep is for, and a grid placed around guess measures the guess. 
One pixel of the rendered marker is the resolution floor, so a level below about 0.01 cannot be drawn at all.
"""

OCCLUDER_GRAY_LEVELS = (0, 60, 120, 180, 240)
"""Gray value of the occluding object. A second dimension of the occlusion cell.

An occluder tone only means something when something is occluded, so it multiplies the occlusion cells and nothing else.
It earns the dimension because binarization sees a tone, not an object, and one fixed tone reports one object.
Evenly spaced across the scale, since which part of the scale matters is not something the sweep gets to assume.
"""

LOW_LIGHT_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
"""Severity of dim ambient light. Also raises sensor noise, as a real camera would."""

BACKLIGHT_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
"""Severity of a bright window behind the marker, blowing out one side."""

WARM_TINT_LEVELS = (0.0, 0.3, 0.6, 0.9)
"""Severity of warm incandescent lighting shifting the white balance."""

HARD_SHADOW_LEVELS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
"""Position of a hard shadow edge across the marker, as a fraction of its width.

This axis is deliberately NOT a severity axis, and the levels are spread to show why. 

Detection difficulty peaks near 0.5 and falls off toward both ends.
Adaptive thresholding recalibrates against a uniformly dark region, while an edge splitting the pattern down the middle breaks binarization on half the modules. 
Reading this axis as "more shadow is worse" would invert the actual result.
"""

MOTION_BLUR_LEVELS = (0.0, 3.0, 7.0, 11.0, 15.0)
"""Motion blur kernel length in pixels. Secondary axis: a service robot moves slowly."""

VIEWPOINT_LEVELS = (0.0, 15.0, 30.0, 45.0, 60.0)
"""Out-of-plane viewing angle in degrees."""

SWEEP: dict[str, tuple[float, ...]] = {
    "occlusion": OCCLUSION_LEVELS,
    "low_light": LOW_LIGHT_LEVELS,
    "backlight": BACKLIGHT_LEVELS,
    "warm_tint": WARM_TINT_LEVELS,
    "hard_shadow": HARD_SHADOW_LEVELS,
    "motion_blur": MOTION_BLUR_LEVELS,
    "viewpoint": VIEWPOINT_LEVELS,
}

# ---------------------------------------------------------------------------
# Dataset composition
# ---------------------------------------------------------------------------
# A "sequence" is one physical scene: a specific marker on a specific background at a specific placement.
# Every degraded sample derived from it shares its id.

# Splits are made over sequences, never over samples, because two samples from the same sequence are near-duplicates and would leak across the split.
# 60 sequences puts ~45 samples in each training-split cell. At 24 sequences the cells held 18 samples, where a single scene moves recall by 5.6 points and
# small differences are indistinguishable from sampling noise.
N_SEQUENCES = 60

# Fraction of sequences reserved for the sealed holdout.
HOLDOUT_FRACTION = 0.25


@dataclass(frozen=True)
class PrintAppearance:
    """How one printed sheet photographs, drawn per sequence.

    Dark and light are bands rather than values, and the sheet drifts across
    them, so no sample asserts that a print is any particular gray.

    Recorded in the manifest so the draw is auditable. Without it a recall
    number cannot be read against the contrast it was measured at, and a
    randomized nuisance variable becomes an unexamined one.

    Attributes:
        ink_low: Darkest the marker's dark modules go on this sheet.
        ink_high: Lightest they go.
        paper_low: Darkest the light parts of the sheet go.
        paper_high: Lightest they go.
        margin_left: Sheet visible to the left of the marker, in pixels.
        margin_top: Sheet visible above the marker, in pixels.
        margin_right: Sheet visible to the right of the marker, in pixels.
        margin_bottom: Sheet visible below the marker, in pixels.
    """

    ink_low: int
    ink_high: int
    paper_low: int
    paper_high: int
    margin_left: int
    margin_top: int
    margin_right: int
    margin_bottom: int

    @property
    def contrast(self) -> int:
        """Worst separation anywhere on the sheet, which is what has to stay readable."""
        return self.paper_low - self.ink_high


@dataclass(frozen=True)
class Sample:
    """One generated image and his info.

    Attrib:
        sample_id: Unique id, also the image filename stem.
        sequence_id: Id of the scene this sample was derived from. Split unit.
        marker_id: Ground-truth ArUco id present in the image.
        axis: Which degradation axis was applied ("clean" when none).
        level: Level along that axis. 0.0 means undegraded.
        corners: Ground-truth marker corners as four (x, y) pairs, clockwise
            from top-left in image coordinates.
        appearance: The printed appearance this sample's scene was rendered at.
        occluder_gray: Tone of the occluding object, on the occlusion axis only.
            None everywhere else, because nothing is occluded there.
    """

    sample_id: str
    sequence_id: str
    marker_id: int
    axis: str
    level: float
    corners: list[list[float]]
    appearance: PrintAppearance
    occluder_gray: int | None = None
