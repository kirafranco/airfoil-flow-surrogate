"""Run any predictor over a split at full mesh resolution, and aggregate.

A predictor takes a `Simulation` and returns a `Prediction`: either the four
fields at every node, in physical units, or (for a lift-only baseline) just a
lift coefficient. The evaluator reads each simulation's full node table, so
every node of every mesh counts and nothing is subsampled.

Per simulation it records relative L2 and standardised MSE per field, the
surface pressure error, and the true and predicted lift coefficients. Over a
split it reports means and medians of those, and for lift:

- **MAE**: the error in the units an engineer reads;
- **median relative error**;
- **mean relative error**, comparable with the AirfRANS paper but fragile:
  airfoils with lift near zero make it explode, so it is never the headline;
- **Spearman ρ_L**: does the model rank the airfoils' lift correctly;
- the share of airfoils within 5 % relative error, the paper's "accurate enough" bound.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.dataset.airfrans import TARGETS, Dataset, Simulation, load_simulation
from src.evaluate.metrics import cp_rmse, lift_coefficient, relative_l2, standardised_mse

ACCURATE_ENOUGH = 0.05


@dataclass(frozen=True)
class Prediction:
    #: (n, 4) u_x, u_y, p, nut in physical units, or None for a lift-only predictor.
    fields: np.ndarray | None
    #: Set only by a lift-only predictor; otherwise lift is integrated from the predicted pressure.
    lift: float | None = None


Predictor = Callable[[Simulation], Prediction]


def true_lift(sim: Simulation) -> float:
    return lift_coefficient(sim.xy, sim.targets[:, 2], sim.normals, sim.segments, aoa_deg=sim.aoa_deg, u_inf=sim.u_inf)


def evaluate_simulation(sim: Simulation, prediction: Prediction, std: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": sim.name,
        "u_inf": sim.u_inf,
        "aoa_deg": sim.aoa_deg,
        "n_nodes": sim.n_nodes,
        "cl_true": true_lift(sim),
    }
    if prediction.fields is None:
        row["cl_pred"] = prediction.lift
        return row
    pred, true = prediction.fields, sim.targets
    if pred.shape != true.shape:
        raise ValueError(f"{sim.name}: prediction shape {pred.shape}, expected {true.shape}")
    for j, target in enumerate(TARGETS):
        row[f"rel_l2_{target}"] = relative_l2(pred[:, j], true[:, j])
        row[f"mse_std_{target}"] = standardised_mse(pred[:, j], true[:, j], std[target])
    surface = sim.is_surface
    row["mse_std_surface_p"] = standardised_mse(pred[surface, 2], true[surface, 2], std["p"])
    row["cp_rmse_surface"] = cp_rmse(pred[surface, 2], true[surface, 2], sim.u_inf)
    row["cl_pred"] = lift_coefficient(sim.xy, pred[:, 2], sim.normals, sim.segments, aoa_deg=sim.aoa_deg, u_inf=sim.u_inf)
    return row


def _finite(value: float) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def aggregate(table: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"simulations": int(len(table))}
    field_columns = [c for c in table.columns if c.startswith(("rel_l2_", "mse_std_", "cp_rmse_"))]
    for column in field_columns:
        out[f"{column}_mean"] = float(table[column].mean())
        out[f"{column}_median"] = float(table[column].median())

    error = table.cl_pred - table.cl_true
    relative = error.abs() / table.cl_true.abs()
    rho = spearmanr(table.cl_pred, table.cl_true)[0] if table.cl_pred.nunique() > 1 else math.nan
    out.update(
        {
            "cl_mae": float(error.abs().mean()),
            "cl_rel_err_median": float(relative.median()),
            "cl_rel_err_mean": float(relative.mean()),
            "cl_within_5pct": float((relative <= ACCURATE_ENOUGH).mean()),
            "cl_spearman": _finite(rho),
        }
    )
    if "inference_ms" in table:
        out["inference_ms_median"] = float(table.inference_ms.median())
    return out


def evaluate(predict: Predictor, dataset: Dataset, names: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-simulation metrics and their aggregate over `names`, every node of every mesh."""
    std = {target: dataset.normalisation["channels"][target]["std"] for target in TARGETS}
    rows = []
    for name in names:
        sim = load_simulation(dataset, name)
        started = time.perf_counter()
        prediction = predict(sim)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        rows.append({**evaluate_simulation(sim, prediction, std), "inference_ms": elapsed_ms})
    table = pd.DataFrame(rows)
    return table, aggregate(table)
