"""The comparison table: every run's test metrics side by side, baselines first.

Runs of the same model and target scaling with different seeds are combined
into mean ± standard deviation. The AirfRANS paper's MLP is added for context,
with two caveats stated in the table itself: its force coefficients include wall
shear (ours are pressure only), and its main-text numbers come from a plain MSE
loss, which the authors note was used by mistake. Their weighted-loss results
are listed too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluate.runs import METRICS_FILE

#: (column, label, format)
COLUMNS = [
    ("rel_l2_u_x_mean", "rel. L2 u_x", "{:.4f}"),
    ("rel_l2_u_y_mean", "rel. L2 u_y", "{:.4f}"),
    ("rel_l2_p_mean", "rel. L2 p", "{:.4f}"),
    ("rel_l2_nut_mean", "rel. L2 ν_t", "{:.4f}"),
    ("cp_rmse_surface_mean", "surface Cp RMSE", "{:.4f}"),
    ("cl_mae", "C_L MAE", "{:.4f}"),
    ("cl_rel_err_median", "C_L median rel. err.", "{:.3f}"),
    ("cl_spearman", "ρ_L", "{:.4f}"),
    ("cl_within_5pct", "C_L within 5 %", "{:.1%}"),
    ("cl_rel_err_mean", "C_L mean rel. err.", "{:.3f}"),
    ("mse_std_u_x_mean", "MSE u_x (std.)", "{:.2e}"),
    ("mse_std_u_y_mean", "MSE u_y (std.)", "{:.2e}"),
    ("mse_std_p_mean", "MSE p (std.)", "{:.2e}"),
    ("mse_std_nut_mean", "MSE ν_t (std.)", "{:.2e}"),
    ("mse_std_surface_p_mean", "MSE surface p (std.)", "{:.2e}"),
    ("inference_ms_median", "ms per mesh", "{:.0f}"),
]

#: arXiv:2212.07564 v3, full data regime, mean of 5 runs. Table 2/3 (plain MSE loss,
#: main text) and Tables 18/19 (weighted loss, Appendix N). Forces include wall shear.
PAPER_ROWS = [
    {
        "label": "AirfRANS paper MLP, MSE loss (forces incl. shear)",
        "mse_std_u_x_mean": 0.95e-2,
        "mse_std_u_y_mean": 0.98e-2,
        "mse_std_p_mean": 0.74e-2,
        "mse_std_nut_mean": 1.90e-2,
        "mse_std_surface_p_mean": 1.13e-1,
        "cl_rel_err_mean": 0.767,
        "cl_spearman": 0.913,
    },
    {
        "label": "AirfRANS paper MLP, weighted loss (forces incl. shear)",
        "mse_std_u_x_mean": 1.63e-2,
        "mse_std_u_y_mean": 1.09e-2,
        "mse_std_p_mean": 0.81e-2,
        "mse_std_nut_mean": 2.59e-2,
        "mse_std_surface_p_mean": 2.00e-1,
        "cl_rel_err_mean": 0.21,
        "cl_spearman": 0.9932,
    },
]


def _label(model: dict[str, Any], targets: dict[str, Any] | None) -> str:
    if model["type"] == "baseline":
        return {"free_stream": "Free stream (no model)", "thin_airfoil": "Thin airfoil, C_L = 2πα"}[model["name"]]
    return f"{model['type'].upper()}, {targets['method'] if targets else '?'} targets"


def collect(models_root: Path, split: str = "test") -> pd.DataFrame:
    """One row per run that has metrics on `split`."""
    rows = []
    for path in sorted(models_root.glob(f"*/{METRICS_FILE}")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        aggregate = metrics.get(f"{split}_metrics")
        if aggregate:
            rows.append({"run_id": metrics["run_id"], "label": _label(metrics["model"], metrics.get("targets")), **aggregate})
    return pd.DataFrame(rows)


def comparison(models_root: Path, split: str = "test") -> pd.DataFrame:
    """Baselines, then models as mean ± std over seeds, then the paper for context."""
    runs = collect(models_root, split)
    order = {"Free stream (no model)": 0, "Thin airfoil, C_L = 2πα": 1}
    rows = []
    if not runs.empty:
        for label, group in sorted(runs.groupby("label"), key=lambda item: (order.get(item[0], 2), item[0])):
            row: dict[str, Any] = {"model": label, "runs": len(group)}
            for column, _, fmt in COLUMNS:
                values = group[column].dropna() if column in group else pd.Series(dtype=float)
                if values.empty:
                    row[column] = ""
                elif len(values) == 1:
                    row[column] = fmt.format(values.iloc[0])
                else:
                    row[column] = f"{fmt.format(values.mean())} ± {fmt.format(values.std(ddof=1))}"
            rows.append(row)
    for paper in PAPER_ROWS:
        row = {"model": paper["label"], "runs": 5}
        for column, _, fmt in COLUMNS:
            row[column] = fmt.format(paper[column]) if column in paper else ""
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.rename(columns={column: label for column, label, _ in COLUMNS})


def write_report(models_root: Path, split: str = "test") -> tuple[Path, Path]:
    table = comparison(models_root, split)
    csv_path = models_root / f"comparison_{split}.csv"
    md_path = models_root / f"comparison_{split}.md"
    table.to_csv(csv_path, index=False)
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    md_path.write_text("\n".join([header, rule, *body]) + "\n", encoding="utf-8")
    return csv_path, md_path
