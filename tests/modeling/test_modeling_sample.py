"""The training sample: every surface node, no validation or test simulation, reproducible."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.dataset.airfrans import Dataset, features, load_simulation, sample_points
from synthetic_interim import N_THETA, write_snapshot


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return write_snapshot(tmp_path_factory.mktemp("snapshot"))


def test_the_splits_are_the_expected_sizes(dataset: Dataset) -> None:
    assert [len(dataset.names(s)) for s in ("train", "val", "test")] == [9, 1, 2]


def test_every_surface_node_plus_the_requested_volume_points(dataset: Dataset) -> None:
    sample = sample_points(dataset, dataset.names("train"), volume_points=40, seed=0)
    assert len(sample.names) == 9
    assert int(sample.is_surface.sum()) == 9 * N_THETA  # all of them
    assert len(sample.features) == 9 * (N_THETA + 40)
    assert np.bincount(sample.simulation).tolist() == [N_THETA + 40] * 9


def test_no_validation_or_test_simulation_contributes_a_point(dataset: Dataset) -> None:
    sample = sample_points(dataset, dataset.names("train"), volume_points=40, seed=0)
    used = {sample.names[i] for i in np.unique(sample.simulation)}
    assert used == set(dataset.names("train"))
    assert not used & (set(dataset.names("val")) | set(dataset.names("test")))


def test_the_sample_depends_on_the_seed_only(dataset: Dataset) -> None:
    a = sample_points(dataset, dataset.names("train"), volume_points=40, seed=3)
    b = sample_points(dataset, list(reversed(dataset.names("train"))), volume_points=40, seed=3)
    c = sample_points(dataset, dataset.names("train"), volume_points=40, seed=4)
    np.testing.assert_array_equal(a.features, b.features)
    assert not np.array_equal(a.features, c.features)


def test_features_are_standardised_with_the_training_statistics(dataset: Dataset) -> None:
    sim = load_simulation(dataset, dataset.names("test")[0])
    x = features(sim, dataset.normalisation)
    stats = dataset.normalisation["channels"]
    assert x.shape == (sim.n_nodes, 7) and x.dtype == np.float32
    np.testing.assert_allclose(x[:, 0], (sim.xy[:, 0] - stats["x"]["mean"]) / stats["x"]["std"], rtol=1e-6)
    np.testing.assert_allclose(x[:, 3], (sim.inlet_u_x - stats["inlet_u_x"]["mean"]) / stats["inlet_u_x"]["std"], rtol=1e-6)
    np.testing.assert_array_equal(x[:, 5:], sim.normals)  # normals pass through


def test_the_fingerprint_names_the_data(dataset: Dataset, tmp_path: Path) -> None:
    fingerprint = dataset.fingerprint()
    assert fingerprint["schema_version"] == "airfrans-interim/1"
    assert set(fingerprint["split_names_sha256"]) == {"train", "val", "test"}
    assert len(fingerprint["normalisation_sha256"]) == 64
