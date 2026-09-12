"""Invariant guards for the sealed split.

The leakage test is the one that matters. Everything downstream (the degradation
curve, the learned detector, the final comparison) is meaningless if a sequence
appears on both sides, and that failure is silent: the numbers just come out
better than they should.
"""

from __future__ import annotations

import json

import pytest

from fiducial import splits


def _ids(n: int = 24) -> list[str]:
    return [f"seq{i:03d}" for i in range(n)]


def test_no_sequence_appears_in_both_sides():
    """No sequence may leak across the split. This is the guard, not a nicety."""
    split = splits.make_split(_ids(), holdout_fraction=0.25, seed=0, fingerprint_="abc")
    assert set(split.train) & set(split.holdout) == set()


def test_split_covers_every_sequence():
    """Dropping sequences silently would shrink the dataset without warning."""
    ids = _ids()
    split = splits.make_split(ids, holdout_fraction=0.25, seed=0, fingerprint_="abc")
    assert sorted(split.train + split.holdout) == sorted(ids)


def test_split_is_deterministic_across_input_ordering():
    """Filesystem ordering is not stable; the split must not depend on it."""
    forward = splits.make_split(_ids(), 0.25, seed=7, fingerprint_="abc")
    backward = splits.make_split(list(reversed(_ids())), 0.25, seed=7, fingerprint_="abc")
    assert forward.holdout == backward.holdout


def test_rejects_fraction_that_empties_a_side():
    with pytest.raises(ValueError, match="empty side"):
        splits.make_split(_ids(4), holdout_fraction=0.01, seed=0, fingerprint_="abc")


def test_seal_refuses_to_overwrite(tmp_path):
    """Resealing after seeing results turns a holdout into a second training set."""
    split = splits.make_split(_ids(), 0.25, seed=0, fingerprint_="abc")
    splits.seal(split, tmp_path)
    with pytest.raises(FileExistsError):
        splits.seal(split, tmp_path)


def test_load_detects_dataset_drift(tmp_path):
    """A regenerated dataset must invalidate the seal instead of being reused."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": []}))
    split = splits.make_split(_ids(), 0.25, seed=0, fingerprint_=splits.fingerprint(manifest))
    splits.seal(split, tmp_path)

    manifest.write_text(json.dumps({"samples": [{"changed": True}]}))
    with pytest.raises(ValueError, match="dataset has changed"):
        splits.load_sealed(tmp_path, manifest)


def test_holdout_access_is_counted(tmp_path):
    """The count is what tells a reader whether a holdout number is still honest."""
    assert splits.record_holdout_access(tmp_path, "first look") == 1
    assert splits.record_holdout_access(tmp_path, "second look") == 2
    log = (tmp_path / splits.ACCESS_LOG_FILENAME).read_text().splitlines()
    assert len(log) == 2
    assert "first look" in log[0]
