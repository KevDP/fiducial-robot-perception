"""Runs on the training split by default. Evaluating against the holdout requires `--holdout` and appends to the access log,
because a number from the first look at a holdout means something different from a number from the twelfth.

How to use:
    python -m fiducial.baseline --dataset data/phase0 --out experiments/baseline_aruco.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fiducial import dataset, detect, metrics, splits


def evaluate(data_dir: Path, manifest: dict, keep: set[str]) -> metrics.Aggregator:
    """Run the detector over every sample whose sequence is in `keep`.

        data_dir: Directory holding the `images/` folder.
        manifest: Parsed manifest.
        keep: Sequence ids to evaluate.

    It returns the opulated aggregator.
    """
    detector = detect.build_detector()
    agg = metrics.Aggregator()
    for record in manifest["samples"]:
        if record["sequence_id"] not in keep:
            continue
        image = cv2.imread(str(data_dir / "images" / f"{record['sample_id']}.png"))
        if image is None:
            raise FileNotFoundError(f"missing image for sample {record['sample_id']}")
        found = detect.detect(image, detector)
        match = detect.find(found, record["marker_id"])
        error = None
        if match is not None:
            error = metrics.corner_rmse(
                match.corners, np.asarray(record["corners"], dtype=np.float32)
            )
        agg.add(
            axis=record["axis"],
            level=record["level"],
            hit=match is not None,
            wrong_id=match is None and bool(found),
            corner_error=error,
        )
    return agg


def print_curves(agg: metrics.Aggregator) -> None:
    """Print one recall curve per degradation axis."""
    for axis, cells in agg.by_axis().items():
        print(f"\n{axis}")
        print(f"  {'level':>8}  {'n':>4}  {'recall':>7}  {'wrong id':>8}  {'corner rmse':>11}")
        for cell in cells:
            rmse = cell.mean_corner_rmse
            rmse_text = f"{rmse:.2f} px" if rmse is not None else "n/a"
            print(
                f"  {cell.level:>8.2f}  {cell.n:>4}  {cell.recall:>6.1%}  "
                f"{cell.wrong_id_rate:>7.1%}  {rmse_text:>11}"
            )


def main() -> None:
    """Evaluate the classical baseline and write the experiment record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/phase0"))
    parser.add_argument("--out", type=Path, default=Path("experiments/baseline_aruco.json"))
    parser.add_argument("--split-dir", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="evaluate against the sealed holdout instead of train. Logs the access.",
    )
    parser.add_argument("--reason", default="", help="why the holdout is being opened")
    args = parser.parse_args()

    manifest_path = args.dataset / dataset.MANIFEST_FILENAME
    manifest = dataset.load_manifest(manifest_path)
    split = splits.load_sealed(args.split_dir, manifest_path)

    if args.holdout:
        if not args.reason:
            parser.error("--holdout requires --reason: say what you are evaluating and why")
        access = splits.record_holdout_access(args.split_dir, args.reason)
        keep, partition = set(split.holdout), "holdout"
        print(f"HOLDOUT ACCESS #{access}: {args.reason}")
    else:
        keep, partition = set(split.train), "train"
        access = None

    agg = evaluate(args.dataset, manifest, keep)
    print(f"\nclassical ArUco baseline on the {partition} split")
    print_curves(agg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "method": "cv2.aruco (default parameters)",
                "partition": partition,
                "holdout_access_number": access,
                "manifest_fingerprint": split.manifest_fingerprint,
                "conditions": [cell.to_dict() for cell in agg.results()],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
