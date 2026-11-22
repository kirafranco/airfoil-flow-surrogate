"""Targets scaled by the flow's own scale map back exactly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dataset.targets import TargetScaling, reference


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    u_inf = rng.uniform(30, 95, size=5000)
    targets = np.stack(
        [u_inf * rng.normal(1.0, 0.3, 5000), u_inf * rng.normal(0.1, 0.2, 5000),
         0.5 * u_inf**2 * rng.normal(-0.5, 1.0, 5000), 1e-4 * u_inf * rng.random(5000)],
        axis=1,
    )
    return targets.astype(np.float32), u_inf


def test_the_reference_values_are_the_free_stream_scales() -> None:
    ref = reference(np.array([10.0]))
    assert ref.tolist() == [[10.0, 10.0, 50.0, 10.0]]  # U∞, U∞, ½U∞², U∞·c with c = 1 m


def test_fitted_scaling_standardises_the_training_targets(data) -> None:  # noqa: ANN001
    targets, u_inf = data
    scaling = TargetScaling.fit(targets, u_inf)
    scaled = scaling.scale(targets.astype(np.float64), u_inf)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, rtol=1e-9)


def test_unscale_inverts_scale_in_numpy_and_torch(data) -> None:  # noqa: ANN001
    targets, u_inf = data
    scaling = TargetScaling.fit(targets, u_inf)
    back = scaling.unscale(scaling.scale(targets.astype(np.float64), u_inf), u_inf)
    np.testing.assert_allclose(back, targets, rtol=1e-12, atol=1e-12)

    # In float32 (how training targets live on the GPU) the error is relative to
    # each column's scale: a value far smaller than its column's spread can't
    # keep its own significant digits through standardisation.
    t, u = torch.from_numpy(targets), torch.from_numpy(u_inf.astype(np.float32))
    back32 = scaling.unscale(scaling.scale(t, u), u).numpy()
    scale = np.sqrt(np.mean(targets.astype(np.float64) ** 2, axis=0))
    assert np.all(np.abs(back32 - targets).max(axis=0) <= 1e-6 * scale)


def test_the_record_round_trips(data) -> None:  # noqa: ANN001
    scaling = TargetScaling.fit(*data)
    assert TargetScaling.from_dict(scaling.as_dict()) == scaling
