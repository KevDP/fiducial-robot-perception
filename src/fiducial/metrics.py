"""Evaluation metrics for marker detection.

The headline number is recall per condition, not overall accuracy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConditionResult:
    """Aggregated outcome for one (axis, level) cell of the sweep.

    Attributes:
        axis: Degradation axis name.
        level: Level along that axis.
        n: Number of samples in this cell.
        hits: Samples where the correct marker id was detected.
        wrong_id: Samples where some other id was reported.
        corner_errors: Per-hit RMSE of the four corner positions, in pixels.
    """

    axis: str
    level: float
    n: int = 0
    hits: int = 0
    wrong_id: int = 0
    corner_errors: list[float] = field(default_factory=list)

    @property
    def recall(self) -> float:
        """Fraction of samples where the correct id was recovered."""
        return self.hits / self.n if self.n else 0.0

    @property
    def wrong_id_rate(self) -> float:
        """Fraction of samples where a wrong id was reported."""
        return self.wrong_id / self.n if self.n else 0.0

    @property
    def mean_corner_rmse(self) -> float | None:
        """Mean corner RMSE over hits, or None when nothing was detected."""
        return float(np.mean(self.corner_errors)) if self.corner_errors else None

    def to_dict(self) -> dict:
        """Serialize for the experiment record."""
        return {
            "axis": self.axis,
            "level": self.level,
            "n": self.n,
            "hits": self.hits,
            "recall": self.recall,
            "wrong_id": self.wrong_id,
            "wrong_id_rate": self.wrong_id_rate,
            "mean_corner_rmse": self.mean_corner_rmse,
        }


def corner_rmse(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square distance between predicted and true corners.

    Corner order is normalized by matching each true corner to its nearest
    predicted one, because ArUco reports corners starting from whichever corner
    it decoded first, and a rotation of the ordering is not a localization error.

    Args:
        predicted: Detected corners, shape (4, 2).
        truth: Ground-truth corners, shape (4, 2).

    Returns:
        RMSE in pixels.
    """
    dists = np.linalg.norm(truth[:, None, :] - predicted[None, :, :], axis=2)
    nearest = dists.min(axis=1)
    return float(np.sqrt((nearest**2).mean()))


class Aggregator:
    """Accumulates per-sample outcomes into per-condition results."""

    def __init__(self) -> None:
        self._cells: dict[tuple[str, float], ConditionResult] = {}

    def add(
        self,
        axis: str,
        level: float,
        hit: bool,
        wrong_id: bool,
        corner_error: float | None = None,
    ) -> None:
        """Record one sample's outcome.

        Args:
            axis: Degradation axis of the sample.
            level: Level along that axis.
            hit: Whether the correct marker id was detected.
            wrong_id: Whether a different id was reported.
            corner_error: Corner RMSE, when there was a hit.
        """
        cell = self._cells.setdefault((axis, level), ConditionResult(axis=axis, level=level))
        cell.n += 1
        if hit:
            cell.hits += 1
            if corner_error is not None:
                cell.corner_errors.append(corner_error)
        if wrong_id:
            cell.wrong_id += 1

    def results(self) -> list[ConditionResult]:
        """Every cell, sorted by axis then level."""
        return sorted(self._cells.values(), key=lambda c: (c.axis, c.level))

    def by_axis(self) -> dict[str, list[ConditionResult]]:
        """Cells grouped by axis, each list sorted by level, for curve plotting."""
        grouped: dict[str, list[ConditionResult]] = defaultdict(list)
        for cell in self.results():
            grouped[cell.axis].append(cell)
        return dict(grouped)
