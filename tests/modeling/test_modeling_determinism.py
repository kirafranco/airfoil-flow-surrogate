"""On the GPU, the same seed gives bit-identical weights; a different seed doesn't.

Marked `gpu`: fails rather than skips without a CUDA device. Needs
CUBLAS_WORKSPACE_CONFIG, which the ml service sets in the compose file.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from src.train.mlp import TrainConfig, train
from synthetic_interim import write_snapshot

pytestmark = pytest.mark.gpu

CONFIG = TrainConfig(epochs=3, patience=5, batch_size=256, volume_points=200, hidden=64, layers=3, device="cuda")


def _weights(run_dir: Path) -> dict[str, torch.Tensor]:
    return torch.load(run_dir / "model.pt", weights_only=True)["state_dict"]


def test_cublas_is_configured_for_determinism() -> None:
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"}


def test_same_seed_same_weights_on_the_gpu(tmp_path: Path) -> None:
    assert torch.cuda.is_available(), "no CUDA device: run inside the ml container"
    dataset = write_snapshot(tmp_path / "snapshot")
    now = lambda: datetime(2026, 11, 23, 10, 0, tzinfo=timezone.utc)  # noqa: E731
    first = _weights(train(CONFIG, dataset, models_root=tmp_path / "a", now=now))
    again = _weights(train(CONFIG, dataset, models_root=tmp_path / "b", now=now))
    other = _weights(train(TrainConfig(**{**CONFIG.__dict__, "seed": 1}), dataset, models_root=tmp_path / "c", now=now))
    assert all(torch.equal(first[k], again[k]) for k in first)
    assert not all(torch.equal(first[k], other[k]) for k in first)
