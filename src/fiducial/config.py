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
#
# Expressed as a framing rule rather than a distance, because a distance is only true
# for one lens. The marker spans 15% of the frame width, so the scene visible at the
# marker's plane is 6.67 times the marker side: a 15 cm marker needs a 1.0 m wide
# field, which sits about 0.87 m away on a 60 degree horizontal FOV camera and
# somewhere else on any other one.
#
# This note previously read "15 cm seen from 2 m with a 60 degree horizontal FOV".
# Those numbers are not compatible: at 2 m that marker spans 42 px, not 96. Shooting
# real footage there would run the detector at half the resolution the sweep renders
# at, and the loss would read as the generator being optimistic.

MARKER_SIDE_PX = 96

# ---------------------------------------------------------------------------
# Degradation sweep
# ---------------------------------------------------------------------------

# One axis per sample, crossing axes would confound attribution: when recall drops we need to know which single condition caused it.
# Combined conditions belong to a later phase, once the per-axis curves exist.
# A "Level 0.0" always means "no degradation on this axis", so every axis shares a common origin and the curves are comparable.

OCCLUSION_LEVELS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
"""Fraction of the marker area covered by an opaque object (a person, a carried object)."""

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
    """

    sample_id: str
    sequence_id: str
    marker_id: int
    axis: str
    level: float
    corners: list[list[float]]
