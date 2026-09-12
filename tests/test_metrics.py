"""Invariant guards for the evaluation metrics."""

from __future__ import annotations

import numpy as np

from fiducial import metrics


def test_corner_rmse_is_zero_for_a_perfect_match():
    corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    assert metrics.corner_rmse(corners, corners) == 0.0


def test_corner_rmse_ignores_corner_ordering():
    """ArUco reports corners from whichever it decoded first; that is not an error."""
    truth = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    rotated = np.roll(truth, shift=1, axis=0)
    assert metrics.corner_rmse(rotated, truth) == 0.0


def test_recall_and_wrong_id_are_tracked_separately():
    """A wrong id is not a miss: a robot acts on it, so it must not hide inside recall."""
    agg = metrics.Aggregator()
    agg.add("occlusion", 0.3, hit=True, wrong_id=False, corner_error=1.0)
    agg.add("occlusion", 0.3, hit=False, wrong_id=True)
    agg.add("occlusion", 0.3, hit=False, wrong_id=False)
    cell = agg.results()[0]
    assert cell.n == 3
    assert cell.recall == 1 / 3
    assert cell.wrong_id_rate == 1 / 3


def test_empty_cell_reports_no_corner_error():
    agg = metrics.Aggregator()
    agg.add("low_light", 0.8, hit=False, wrong_id=False)
    assert agg.results()[0].mean_corner_rmse is None
