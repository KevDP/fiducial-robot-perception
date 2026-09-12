"""Dataset generation: the manifest is the dataset. It carries the condition
labels, the ground-truth corners and the objective image statistics, and it is
what the split is fingerprinted against.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from fiducial import config, degrade, imagestats, scene

MANIFEST_FILENAME = "manifest.json"


def _sequence_seed(base_seed: int, index: int) -> int:
    """Derive a per-sequence seed so sequences are independent but reproducible"""
    return base_seed * 100_003 + index


def _cell_seed(base_seed: int, index: int, axis: str) -> int:
    """Derive a per-cell seed that is stable across processes.

    Uses CRC32 rather than the builtin `hash`, which is randomized per process for strings.

    Args:
        base_seed: Dataset-wide seed.
        index: Sequence index.
        axis: Degradation axis name.

    Returns a deterministic seed for this (sequence, axis) cell.
    """
    return _sequence_seed(base_seed, index) + zlib.crc32(axis.encode()) % 9973


def iter_samples(n_sequences: int, seed: int):
    """Yield every (sample, image) pair in the sweep.

    Args:
        n_sequences: Number of distinct scenes to build.
        seed: Base seed for the whole dataset.

    Yields a tuple of (Sample, BGR image).
    """
    dictionary_size = 50  # DICT_4X4_50
    for index in range(n_sequences):
        rng = np.random.default_rng(_sequence_seed(seed, index))
        marker_id = int(rng.integers(0, dictionary_size))
        sequence_id = f"seq{index:03d}"
        clean_image, clean_corners = scene.render(marker_id, rng)

        yield (
            config.Sample(
                sample_id=f"{sequence_id}_clean",
                sequence_id=sequence_id,
                marker_id=marker_id,
                axis="clean",
                level=0.0,
                corners=clean_corners.tolist(),
            ),
            clean_image,
        )

        for axis, levels in config.SWEEP.items():
            for level in levels:
                if level <= 0.0:
                    continue
                # A fresh generator per cell keeps one axis from consuming random
                # draws that would shift another axis's choices.
                cell_rng = np.random.default_rng(_cell_seed(seed, index, axis))
                image, corners = degrade.apply(
                    clean_image.copy(), clean_corners.copy(), axis, float(level), cell_rng
                )
                yield (
                    config.Sample(
                        sample_id=f"{sequence_id}_{axis}_{level:g}",
                        sequence_id=sequence_id,
                        marker_id=marker_id,
                        axis=axis,
                        level=float(level),
                        corners=corners.tolist(),
                    ),
                    image,
                )


def generate(out_dir: Path, n_sequences: int, seed: int) -> Path:
    """Write the full sweep to disk and return the manifest path.

    Args:
        out_dir: Destination directory. Created if missing.
        n_sequences: Number of scenes.
        seed: Base seed.

    Returns a path to the written manifest.
    """
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for sample, image in iter_samples(n_sequences, seed):
        cv2.imwrite(str(images_dir / f"{sample.sample_id}.png"), image)
        record = asdict(sample)
        record["stats"] = imagestats.describe(image, np.asarray(sample.corners, dtype=np.float32))
        records.append(record)

    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "n_sequences": n_sequences,
                "image_width": config.IMAGE_WIDTH,
                "image_height": config.IMAGE_HEIGHT,
                "aruco_dict": config.ARUCO_DICT_NAME,
                "samples": records,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest_path


def load_manifest(manifest_path: Path) -> dict:
    """Read a manifest written by `generate`."""
    return json.loads(manifest_path.read_text())


def sequence_ids(manifest: dict) -> list[str]:
    """Every distinct sequence id in a manifest, sorted."""
    return sorted({record["sequence_id"] for record in manifest["samples"]})
