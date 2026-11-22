"""Training end to end on a synthetic snapshot (CPU), the loss, and early stopping."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from src.dataset.airfrans import Dataset
from src.evaluate.runs import RepeatedTestEvaluation, evaluate_run
from src.train.mlp import TrainConfig, load_model, loss_fn, train
from synthetic_interim import read_json, write_snapshot

SMALL = TrainConfig(epochs=3, patience=5, batch_size=256, volume_points=120, hidden=16, layers=2, device="cpu")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return write_snapshot(tmp_path_factory.mktemp("snapshot"))


def _at(minute: int):  # noqa: ANN202
    return lambda: datetime(2026, 11, 23, 10, minute, tzinfo=timezone.utc)


def test_the_loss_weighs_volume_and_surface_separately() -> None:
    pred = torch.tensor([[1.0, 1.0], [3.0, 3.0], [0.0, 0.0]])
    target = torch.zeros(3, 2)
    surface = torch.tensor([0.0, 0.0, 1.0])
    # volume: mean of per-point MSEs 1 and 9 → 5; surface: 0.
    assert loss_fn(pred, target, surface, surface_weight=1.0).item() == pytest.approx(5.0)
    surface = torch.tensor([0.0, 1.0, 1.0])
    assert loss_fn(pred, target, surface, surface_weight=2.0).item() == pytest.approx(1.0 + 2.0 * 4.5)


def test_a_run_writes_weights_metrics_and_validation_results(dataset: Dataset, tmp_path: Path) -> None:
    run_dir = train(SMALL, dataset, models_root=tmp_path, now=_at(0))

    assert run_dir.name == "mlp-physics-s0-20261123T100000Z"
    metrics = read_json(run_dir / "metrics.json")
    assert metrics["model"]["type"] == "mlp" and metrics["model"]["parameters"] > 0
    assert metrics["targets"]["method"] == "physics"
    assert metrics["training"]["epochs_run"] == 3 and len(metrics["training"]["curve"]) == 3
    assert 1 <= metrics["training"]["best_epoch"] <= 3
    assert metrics["training"]["train_surface_points"] == 9 * 32
    assert metrics["dataset"] == dataset.fingerprint()
    assert metrics["val_metrics"]["simulations"] == 1
    assert metrics["test_metrics"] is None
    assert (run_dir / "val_metrics.parquet").is_file()
    model, scaling = load_model(run_dir, torch.device("cpu"))
    assert model.parameter_count() == metrics["model"]["parameters"]
    assert not torch.are_deterministic_algorithms_enabled()  # restored after training

    evaluate_run(run_dir, dataset, "test", device="cpu")
    assert read_json(run_dir / "metrics.json")["test_metrics"]["simulations"] == 2
    with pytest.raises(RepeatedTestEvaluation):
        evaluate_run(run_dir, dataset, "test", device="cpu")


def test_training_stops_when_validation_stops_improving(dataset: Dataset, tmp_path: Path) -> None:
    # A zero learning rate leaves the weights, and so the validation loss, unchanged.
    config = TrainConfig(**{**SMALL.__dict__, "epochs": 10, "patience": 2, "lr": 0.0, "lr_min": 0.0})
    run_dir = train(config, dataset, models_root=tmp_path, now=_at(1))
    training = read_json(run_dir / "metrics.json")["training"]
    assert training["best_epoch"] == 1
    assert training["epochs_run"] == 3


def test_the_same_seed_on_the_cpu_gives_identical_weights(dataset: Dataset, tmp_path: Path) -> None:
    a = train(SMALL, dataset, models_root=tmp_path / "a", now=_at(2))
    b = train(SMALL, dataset, models_root=tmp_path / "b", now=_at(2))
    wa = torch.load(a / "model.pt", weights_only=True)["state_dict"]
    wb = torch.load(b / "model.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
