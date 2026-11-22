"""The evaluator on synthetic snapshots: perfect and naive predictors, test discipline, the report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset.airfrans import Dataset, Simulation
from src.evaluate.evaluate import Prediction, evaluate
from src.evaluate.report import comparison, write_report
from src.evaluate.runs import DatasetMismatch, RepeatedTestEvaluation, baseline_run, evaluate_run
from src.models.baselines import free_stream, thin_airfoil
from synthetic_interim import analytic_lift, write_snapshot


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return write_snapshot(tmp_path / "snapshot", n_sims=12, n_test=3)


def perfect(sim: Simulation) -> Prediction:
    return Prediction(fields=sim.targets.astype(np.float64))


def test_a_perfect_prediction_scores_perfectly_on_every_node(dataset: Dataset) -> None:
    table, summary = evaluate(perfect, dataset, dataset.names("test"))
    assert table.n_nodes.tolist() == dataset.simulations.loc[dataset.names("test")].n_nodes.tolist()
    for target in ("u_x", "u_y", "p", "nut"):
        assert summary[f"rel_l2_{target}_mean"] == 0.0
        assert summary[f"mse_std_{target}_mean"] == 0.0
    assert summary["cl_mae"] == 0.0
    assert summary["cl_spearman"] == pytest.approx(1.0)
    assert summary["cl_within_5pct"] == 1.0


def test_the_true_lift_is_close_to_the_closed_form(dataset: Dataset) -> None:
    table, _ = evaluate(perfect, dataset, dataset.names("test"))
    expected = [analytic_lift(a) for a in table.aoa_deg]
    np.testing.assert_allclose(table.cl_true, expected, rtol=0.01)  # a 32-sided polygon, not a circle


def test_free_stream_scores_one_on_pressure_and_turbulence_and_predicts_no_lift(dataset: Dataset) -> None:
    table, summary = evaluate(free_stream, dataset, dataset.names("test"))
    assert summary["rel_l2_p_mean"] == pytest.approx(1.0)
    assert summary["rel_l2_nut_mean"] == pytest.approx(1.0)
    assert (table.cl_pred == 0).all()
    assert summary["cl_spearman"] is None  # a constant prediction has no ranking


def test_thin_airfoil_predicts_lift_only(dataset: Dataset) -> None:
    table, summary = evaluate(thin_airfoil, dataset, dataset.names("test"))
    assert not any(c.startswith("rel_l2_") for c in table.columns)
    np.testing.assert_allclose(table.cl_pred, 2 * np.pi * np.radians(table.aoa_deg))
    assert summary["cl_spearman"] == pytest.approx(1.0)  # monotonic in α, as is the synthetic lift


def test_the_test_set_is_read_once_per_run(dataset: Dataset, tmp_path: Path) -> None:
    run_dir = baseline_run("free_stream", dataset, tmp_path / "models")
    evaluate_run(run_dir, dataset, "val", device="cpu")
    evaluate_run(run_dir, dataset, "val", device="cpu")  # validation may be revisited
    evaluate_run(run_dir, dataset, "test", device="cpu")
    with pytest.raises(RepeatedTestEvaluation):
        evaluate_run(run_dir, dataset, "test", device="cpu")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_evaluated"]["simulations"] == 3
    assert (run_dir / "test_metrics.parquet").is_file()


def test_a_run_from_other_data_is_refused(dataset: Dataset, tmp_path: Path) -> None:
    run_dir = baseline_run("thin_airfoil", dataset, tmp_path / "models")
    path = run_dir / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["dataset"]["normalisation_sha256"] = "0" * 64
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(DatasetMismatch, match="normalisation_sha256"):
        evaluate_run(run_dir, dataset, "test", device="cpu")


def test_the_report_puts_baselines_first_and_combines_seeds(dataset: Dataset, tmp_path: Path) -> None:
    models = tmp_path / "models"
    for name in ("thin_airfoil", "free_stream"):
        evaluate_run(baseline_run(name, dataset, models), dataset, "test", device="cpu")
    _, free = evaluate(free_stream, dataset, dataset.names("test"))
    for seed, offset in ((0, 0.0), (1, 0.02)):
        run = models / f"mlp-physics-s{seed}-20261123T000000Z"
        run.mkdir()
        metrics = {
            "run_id": run.name,
            "model": {"type": "mlp"},
            "targets": {"method": "physics"},
            "test_metrics": {**free, "cl_mae": 0.1 + offset},
        }
        (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    table = comparison(models)
    assert table.model.tolist()[:3] == ["Free stream (no model)", "Thin airfoil, C_L = 2πα", "MLP, physics targets"]
    assert table.set_index("model").loc["MLP, physics targets", "C_L MAE"] == "0.1100 ± 0.0141"
    assert any("AirfRANS paper" in m for m in table.model)
    csv_path, md_path = write_report(models)
    assert csv_path.is_file() and md_path.read_text(encoding="utf-8").startswith("| model |")


def test_runs_of_different_configurations_are_never_averaged_together(dataset: Dataset, tmp_path: Path) -> None:
    models = tmp_path / "models"
    _, free = evaluate(free_stream, dataset, dataset.names("test"))
    configs = [(64, 60, 0.0), (64, 60, 0.0), (256, 150, 0.0), (64, 60, 1e-4)]
    for i, (hidden, epochs, wd) in enumerate(configs):
        run = models / f"mlp-physics-s{i}-20261123T00000{i}Z"
        run.mkdir(parents=True)
        metrics = {
            "run_id": run.name,
            "model": {"type": "mlp", "hidden": hidden, "layers": 4},
            "targets": {"method": "physics"},
            "training": {"epochs": epochs, "weight_decay": wd},
            "test_metrics": free,
        }
        (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    table = comparison(models).set_index("model")
    assert table.loc["MLP 4×64, physics targets, cosine 60", "runs"] == 2
    assert table.loc["MLP 4×256, physics targets, cosine 150", "runs"] == 1
    assert table.loc["MLP 4×64, physics targets, cosine 60, wd 0.0001", "runs"] == 1
