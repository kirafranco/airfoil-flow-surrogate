"""Train / val / test by simulation, from the manifest."""

from __future__ import annotations

import random

import pytest

from src.etl.splits import make_splits, names_sha256, splits_record


def _manifest(n_train: int = 800, n_test: int = 200) -> dict[str, list[str]]:
    return {
        "full_train": [f"train_{i:04d}" for i in range(n_train)],
        "full_test": [f"test_{i:04d}" for i in range(n_test)],
    }


def test_sizes_on_the_real_manifest_shape() -> None:
    splits = make_splits(_manifest())
    assert (len(splits.train), len(splits.val), len(splits.test)) == (720, 80, 200)


def test_val_comes_from_full_train_and_test_is_full_test_unchanged() -> None:
    manifest = _manifest()
    splits = make_splits(manifest)
    assert set(splits.val) <= set(manifest["full_train"])
    assert set(splits.train) | set(splits.val) == set(manifest["full_train"])
    assert not set(splits.train) & set(splits.val)
    assert splits.test == tuple(sorted(manifest["full_test"]))


def test_the_manifest_order_does_not_matter_only_the_seed_does() -> None:
    manifest = _manifest()
    shuffled = {key: random.Random(1).sample(names, len(names)) for key, names in manifest.items()}
    assert make_splits(shuffled) == make_splits(manifest)
    assert make_splits(manifest, seed=1).val != make_splits(manifest).val


def test_overlapping_train_and_test_is_refused() -> None:
    manifest = _manifest(10, 2)
    manifest["full_test"].append(manifest["full_train"][0])
    with pytest.raises(ValueError, match="share 1"):
        make_splits(manifest)


def test_the_record_lists_names_and_sizes() -> None:
    record = splits_record(make_splits(_manifest(10, 2)), "ab" * 32)
    assert record["sizes"] == {"train": 9, "val": 1, "test": 2}
    assert record["seed"] == 0 and record["raw_sha256"] == "ab" * 32


def test_names_fingerprint_ignores_order() -> None:
    assert names_sha256(["b", "a"]) == names_sha256(("a", "b")) != names_sha256(["a"])
