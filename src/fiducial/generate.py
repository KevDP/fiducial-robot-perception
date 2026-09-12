"""Build the preliminar dataset and seal the holdout split.

How to use:
    python -m fiducial.generate --out data/phase0 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fiducial import config, dataset, splits


def main() -> None:
    """Generate the sweep, then seal a sequence-level train/holdout split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/phase0"), help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="base seed for the whole dataset")
    parser.add_argument(
        "--sequences", type=int, default=config.N_SEQUENCES, help="number of distinct scenes"
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("experiments"),
        help="where the sealed split and access log live (committed to git)",
    )
    args = parser.parse_args()

    manifest_path = dataset.generate(args.out, args.sequences, args.seed)
    manifest = dataset.load_manifest(manifest_path)
    n_samples = len(manifest["samples"])
    print(f"wrote {n_samples} samples across {args.sequences} sequences -> {manifest_path}")

    ids = dataset.sequence_ids(manifest)
    split = splits.make_split(
        ids, config.HOLDOUT_FRACTION, args.seed, splits.fingerprint(manifest_path)
    )
    try:
        seal_path = splits.seal(split, args.split_dir)
    except FileExistsError as exc:
        print(f"\nsplit NOT sealed: {exc}")
        return
    print(
        f"sealed split -> {seal_path}\n"
        f"  train:   {len(split.train)} sequences\n"
        f"  holdout: {len(split.holdout)} sequences (do not touch until the end)"
    )


if __name__ == "__main__":
    main()
