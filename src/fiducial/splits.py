"""Sealed train/holdout split.

Two rules, both of which exist because they are easy to break by accident:

1. Split over sequences, never over samples: Every degraded sample derived
   from one scene shares that scene's background, marker placement and marker id.
   Splitting at the sample level would put near-duplicates on both sides and
   inflate every number that follows.

2. The holdout is sealed and its use is logged: The split is written once,
   fingerprinted against the manifest it was computed from, and every evaluation
   that touches the holdout appends a line to an access log. Running forty
   experiments and reporting the best one is a real way to fool yourself, and it
   leaves no trace unless something records it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SEAL_FILENAME = "split.sealed.json"
ACCESS_LOG_FILENAME = "holdout_access.log"


@dataclass(frozen=True)
class Split:
    """A train/holdout partition over sequence ids.

    Attributes:
        train: Sequence ids available for development and model selection.
        holdout: Sequence ids reserved for the final, one-shot evaluation.
        manifest_fingerprint: SHA-256 of the manifest this split was computed from.
        If the dataset is regenerated, the fingerprint stops matching
        and the split must be re-sealed rather than silently reused.
        seed: Seed used to shuffle sequences.
    """

    train: list[str]
    holdout: list[str]
    manifest_fingerprint: str
    seed: int


def fingerprint(manifest_path: Path) -> str:
    """SHA-256 of a manifest file, used to detect dataset drift."""
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def make_split(
    sequence_ids: list[str], holdout_fraction: float, seed: int, fingerprint_: str
) -> Split:
    """Partition sequences deterministically.

    Sorting before shuffling matters: dictionary or filesystem ordering is not stable across machines,
    and an unstable split makes results irreproducible in a way that is very hard to notice.

    Args:
        sequence_ids: All sequence ids in the dataset.
        holdout_fraction: Fraction of sequences to reserve, in (0, 1).
        seed: Shuffle seed.
        fingerprint_: Manifest fingerprint to record in the split.

    This returns the partition.

    Raises a ValueError if the fraction would leave either side empty.
    """
    import random

    ordered = sorted(sequence_ids)
    n_holdout = int(len(ordered) * holdout_fraction)
    if n_holdout == 0 or n_holdout == len(ordered):
        raise ValueError(
            f"holdout_fraction={holdout_fraction} over {len(ordered)} sequences leaves an empty side"
        )
    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)
    return Split(
        train=sorted(shuffled[n_holdout:]),
        holdout=sorted(shuffled[:n_holdout]),
        manifest_fingerprint=fingerprint_,
        seed=seed,
    )


def seal(split: Split, out_dir: Path) -> Path:
    """Write the split to disk, refusing to overwrite an existing seal.

    Overwriting would defeat the purpose, because a split that can be regenerated after
    seeing results is not a holdout, it is a second training set.

    Args:
        split: The partition to record.
        out_dir: Directory to write into.

    Returns a path to the written seal.

    Raises a FileExistsError if a seal is already present.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SEAL_FILENAME
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Delete it deliberately if you really mean to reseal, "
            "and note in the README that previously reported holdout numbers are void."
        )
    payload = {
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_fingerprint": split.manifest_fingerprint,
        "seed": split.seed,
        "n_train": len(split.train),
        "n_holdout": len(split.holdout),
        "train": split.train,
        "holdout": split.holdout,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_sealed(out_dir: Path, manifest_path: Path) -> Split:
    """Load the sealed split and verify it matches the current dataset.

    Args:
        out_dir: Directory holding the seal.
        manifest_path: Manifest to fingerprint and compare against.

    Returns a he recorded partition.

    Raises a FileNotFoundError if no seal exists yet or a ValueError if the dataset no longer matches the sealed fingerprint.
    """
    path = out_dir / SEAL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"no sealed split at {path}; run `python -m fiducial.generate` first"
        )
    payload = json.loads(path.read_text())
    current = fingerprint(manifest_path)
    if payload["manifest_fingerprint"] != current:
        raise ValueError(
            "the dataset has changed since the split was sealed. The sealed holdout no longer "
            "refers to the same images, so any number computed against it would be meaningless."
        )
    return Split(
        train=payload["train"],
        holdout=payload["holdout"],
        manifest_fingerprint=payload["manifest_fingerprint"],
        seed=payload["seed"],
    )


def record_holdout_access(out_dir: Path, reason: str) -> int:
    """Append one line to the holdout access log and return the running count.

    The returned count belongs in the README next to any holdout number. A result
    from the first access means something different from the twelfth, and the
    reader deserves to know which one they are looking at.

    Args:
        out_dir: Directory holding the log.
        reason: Short description of what was evaluated and why.

    Returns how many times the holdout has now been accessed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ACCESS_LOG_FILENAME
    previous = path.read_text().splitlines() if path.exists() else []
    count = len(previous) + 1
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a") as handle:
        handle.write(f"{count}\t{stamp}\t{reason}\n")
    return count
