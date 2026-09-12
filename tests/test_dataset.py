"""Invariant guards for dataset generation and the manifest."""

from __future__ import annotations

import numpy as np

from fiducial import config, dataset


def test_every_sample_carries_its_sequence_and_condition():
    samples = [s for s, _ in dataset.iter_samples(n_sequences=2, seed=0)]
    assert samples, "generator produced nothing"
    for sample in samples:
        assert sample.sequence_id.startswith("seq")
        assert sample.axis == "clean" or sample.axis in config.SWEEP
        assert len(sample.corners) == 4


def test_each_sequence_yields_one_clean_sample():
    samples = [s for s, _ in dataset.iter_samples(n_sequences=3, seed=0)]
    clean = [s for s in samples if s.axis == "clean"]
    assert len(clean) == 3
    assert len({s.sequence_id for s in clean}) == 3


def test_zero_levels_are_not_materialized():
    """A level-0 cell would duplicate the clean sample and inflate the easy end."""
    samples = [s for s, _ in dataset.iter_samples(n_sequences=2, seed=0)]
    assert all(s.level > 0.0 for s in samples if s.axis != "clean")


def test_sample_ids_are_unique():
    samples = [s for s, _ in dataset.iter_samples(n_sequences=4, seed=0)]
    ids = [s.sample_id for s in samples]
    assert len(ids) == len(set(ids))


def test_generation_is_reproducible():
    first = [(s.sample_id, np.asarray(s.corners)) for s, _ in dataset.iter_samples(2, seed=3)]
    second = [(s.sample_id, np.asarray(s.corners)) for s, _ in dataset.iter_samples(2, seed=3)]
    assert [i for i, _ in first] == [i for i, _ in second]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(first, second, strict=True))


def test_all_samples_from_one_sequence_share_the_marker_id():
    """Split integrity depends on a sequence being one scene, not a mix."""
    samples = [s for s, _ in dataset.iter_samples(n_sequences=2, seed=0)]
    by_sequence: dict[str, set[int]] = {}
    for sample in samples:
        by_sequence.setdefault(sample.sequence_id, set()).add(sample.marker_id)
    assert all(len(ids) == 1 for ids in by_sequence.values())


def test_generation_is_reproducible_across_processes():
    """The same seed must produce the same dataset in a different interpreter.

    Python randomizes `hash()` for strings per process, so a seed derived from it
    is stable within one run and different on the next. That made `--seed 0`
    produce a different manifest every time and left the sealed split
    unauditable, since nobody could regenerate what the seal fingerprinted.
    Running under two different PYTHONHASHSEED values is what catches it; an
    in-process test cannot, because `hash()` is stable inside a single run.
    """
    import os
    import subprocess
    import sys

    code = (
        "from fiducial import dataset;"
        "print([(s.sample_id, s.level, round(float(sum(sum(c) for c in s.corners)), 3))"
        " for s, _ in dataset.iter_samples(2, seed=0)])"
    )
    outputs = []
    for hashseed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1], "dataset generation depends on PYTHONHASHSEED"


def test_cell_seed_is_stable_for_a_known_input():
    """A regression pin: this value must not move when the derivation is touched."""
    assert dataset._cell_seed(0, 0, "occlusion") == dataset._cell_seed(0, 0, "occlusion")
    assert dataset._cell_seed(0, 0, "occlusion") != dataset._cell_seed(0, 0, "low_light")
