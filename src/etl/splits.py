"""Train / validation / test, by simulation, from the dataset's own manifest.

The test set is the manifest's `full_test` (200 airfoils), unchanged, so results
are comparable with the published benchmark. Validation is carved out of
`full_train` by simulation with a fixed seed. Splitting by mesh point instead
would put points of the same airfoil on both sides and leak its geometry.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

VAL_FRACTION = 0.1
SEED = 0


@dataclass(frozen=True)
class Splits:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    val_fraction: float


def names_sha256(names: tuple[str, ...] | list[str]) -> str:
    """A fingerprint of a set of simulation names, independent of order."""
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def make_splits(manifest: dict[str, list[str]], *, val_fraction: float = VAL_FRACTION, seed: int = SEED) -> Splits:
    pool = sorted(manifest["full_train"])
    test = tuple(sorted(manifest["full_test"]))
    overlap = set(pool) & set(test)
    if overlap:
        raise ValueError(f"full_train and full_test share {len(overlap)} simulation(s), e.g. {sorted(overlap)[0]}")
    # Sampling from the sorted list makes the result depend on the seed only,
    # not on the order the manifest happens to list names in.
    val = tuple(sorted(random.Random(seed).sample(pool, round(len(pool) * val_fraction))))
    val_set = set(val)
    train = tuple(name for name in pool if name not in val_set)
    return Splits(train=train, val=val, test=test, seed=seed, val_fraction=val_fraction)


def splits_record(splits: Splits, raw_sha256: str) -> dict[str, Any]:
    return {
        "seed": splits.seed,
        "val_fraction": splits.val_fraction,
        "source": "Dataset/manifest.json full_train / full_test",
        "raw_sha256": raw_sha256,
        "sizes": {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
        "train": list(splits.train),
        "val": list(splits.val),
        "test": list(splits.test),
    }
