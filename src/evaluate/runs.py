"""Evaluate a run directory on a split, and enforce reading the test set once.

A run directory is data/models/<run_id>/ with a metrics.json: a trained model
(model.pt alongside) or a baseline. Evaluating it on a split writes
`<split>_metrics.parquet` (one row per simulation) and adds the aggregates to
metrics.json under `<split>_metrics`.

**The test set is read once per run.** Model choices are made on validation;
if a run already has test metrics, evaluating it on test again is refused,
so a test score can't be tuned against by re-running. A run is also refused
if it was trained on a different snapshot, split or normalisation than the
dataset being evaluated.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.dataset.airfrans import Dataset
from src.etl.schema import write_json
from src.evaluate.evaluate import Predictor, evaluate
from src.ingest.airfrans import git_commit
from src.models.baselines import BASELINES

logger = logging.getLogger(__name__)

METRICS_FILE = "metrics.json"
SPLITS = ("val", "test")


class RepeatedTestEvaluation(Exception):
    """A second evaluation of a run on the test set."""


class DatasetMismatch(Exception):
    """A run trained on different data from the dataset being evaluated."""


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_run(name: str, dataset: Dataset, models_root: Path) -> Path:
    """The run directory for a baseline, created on first use."""
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}")
    run_dir = models_root / f"baseline-{name}"
    if not (run_dir / METRICS_FILE).is_file():
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            run_dir / METRICS_FILE,
            {
                "run_id": run_dir.name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": {"type": "baseline", "name": name},
                "dataset": dataset.fingerprint(),
                "code_commit": git_commit(),
            },
        )
    return run_dir


def _predictor(run_dir: Path, metrics: dict[str, Any], dataset: Dataset, device: str) -> Predictor:
    model = metrics["model"]
    if model["type"] == "baseline":
        return BASELINES[model["name"]]
    if model["type"] == "mlp":
        from src.train.mlp import load_model, mlp_predictor  # torch-heavy; only when needed

        torch_device = torch.device(device)
        mlp, scaling = load_model(run_dir, torch_device)
        return mlp_predictor(mlp, scaling, dataset.normalisation, torch_device)
    raise ValueError(f"{run_dir.name}: unknown model type {model['type']!r}")


def _check_same_data(run_dir: Path, recorded: dict[str, Any], dataset: Dataset) -> None:
    current = dataset.fingerprint()
    for key in ("raw_sha256", "schema_version", "split_names_sha256", "normalisation_sha256"):
        if recorded.get(key) != current.get(key):
            raise DatasetMismatch(f"{run_dir.name} was built on different data ({key} differs); evaluate it on its own dataset")


def evaluate_run(run_dir: Path, dataset: Dataset, split: str, *, device: str = "cuda") -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    metrics = _read(run_dir / METRICS_FILE)
    table_path = run_dir / f"{split}_metrics.parquet"
    if split == "test" and (table_path.exists() or metrics.get("test_metrics")):
        raise RepeatedTestEvaluation(
            f"{run_dir.name} has already been evaluated on the test set; choices are made on validation, "
            "and the test set is read once per model"
        )
    _check_same_data(run_dir, metrics["dataset"], dataset)

    table, aggregate = evaluate(_predictor(run_dir, metrics, dataset, device), dataset, dataset.names(split))
    tmp = table_path.with_name(table_path.name + ".tmp")
    table.to_parquet(tmp)
    os.replace(tmp, table_path)
    metrics[f"{split}_metrics"] = aggregate
    metrics[f"{split}_evaluated"] = {
        "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(),
        "simulations": len(table),
    }
    write_json(run_dir / METRICS_FILE, metrics)
    logger.info(
        "evaluated run=%s split=%s simulations=%d cl_mae=%.4f cl_spearman=%s rel_l2_p=%s",
        run_dir.name,
        split,
        len(table),
        aggregate["cl_mae"],
        aggregate["cl_spearman"],
        aggregate.get("rel_l2_p_mean"),
    )
    return aggregate
