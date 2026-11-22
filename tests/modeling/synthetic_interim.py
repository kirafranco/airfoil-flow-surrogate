"""Tiny interim snapshots, written straight to the ETL's schema, for the modelling tests.

Each simulation is a ring mesh around a circular "airfoil" of radius R:
N_THETA nodes per ring, N_RINGS rings, quads between neighbouring rings. The
inner ring is the surface: zero velocity (no-slip), zero distance, outward
normals pointing radially. Fields are smooth functions of position and inflow,
so a small model can learn them.

On the surface the pressure is p = −½U∞² k sin(θ − α), whose lift on a circle
is known in closed form: C_L = k π R (c = 1). The ring is a polygon, so the
integrated value differs slightly, which is what the tests allow for.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from src.dataset.airfrans import Dataset
from src.etl.mesh import quad_edges
from src.etl.normalise import fit_normalisation
from src.etl.schema import (
    EDGES_FILE,
    EDGES_SCHEMA,
    MANIFEST_SPLITS,
    NODES_FILE,
    NODES_SCHEMA,
    SCHEMA_VERSION,
    SIMULATIONS_SCHEMA,
    SURFACE_FILE,
    SURFACE_SCHEMA,
    build_table,
    write_json,
    write_parquet_atomic,
    write_simulation,
)
from src.etl.splits import make_splits, splits_record

RADIUS = 0.25
CENTRE = (0.5, 0.0)
N_THETA = 32
N_RINGS = 5
SNAPSHOT = "2026-11-20"
RAW_SHA256 = "ab" * 32


def sim_name(u_inf: float, aoa_deg: float) -> str:
    return f"airFoil2D_SST_{u_inf}_{aoa_deg}_2.0_4.0_12.0"


def lift_factor(aoa_deg: float) -> float:
    """k in p = −½U∞² k sin(θ − α) on the surface, so C_L ≈ k π R."""
    return 0.5 + 4.0 * math.sin(math.radians(aoa_deg))


def analytic_lift(aoa_deg: float) -> float:
    return lift_factor(aoa_deg) * math.pi * RADIUS


def ring_simulation(u_inf: float, aoa_deg: float) -> dict[str, np.ndarray]:
    theta = 2 * math.pi * np.arange(N_THETA) / N_THETA
    radii = RADIUS * (1 + 0.6 * np.arange(N_RINGS))
    r = np.repeat(radii, N_THETA)
    t = np.tile(theta, N_RINGS)
    xy = np.stack([CENTRE[0] + r * np.cos(t), CENTRE[1] + r * np.sin(t)], axis=1)
    ring = np.arange(N_RINGS * N_THETA) // N_THETA
    j = np.arange(N_THETA)
    quads = np.concatenate(
        [np.stack([k * N_THETA + j, k * N_THETA + (j + 1) % N_THETA, (k + 1) * N_THETA + (j + 1) % N_THETA, (k + 1) * N_THETA + j], axis=1)
         for k in range(N_RINGS - 1)]
    )
    surface = ring == 0
    normals = np.zeros((len(xy), 2), dtype=np.float32)
    normals[surface] = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    alpha = math.radians(aoa_deg)
    damp = 1 - RADIUS / r  # 0 on the surface: no-slip
    q = 0.5 * u_inf**2
    return {
        "xy": xy,
        "sdf": r - RADIUS,
        "is_surface": surface,
        "normals": normals,
        "u_x": (u_inf * math.cos(alpha) * damp * (1 + 0.2 * np.sin(t))).astype(np.float32),
        "u_y": (u_inf * math.sin(alpha) * damp + 0.1 * u_inf * damp * np.cos(t)).astype(np.float32),
        "p": (-q * lift_factor(aoa_deg) * np.sin(t - alpha) * RADIUS / r).astype(np.float32),
        "nut": (1e-5 * (r - RADIUS) * u_inf / 40).astype(np.float32),
        "edges": quad_edges(quads.ravel(), len(xy)),
        "segments": np.stack([j, (j + 1) % N_THETA], axis=1),
    }


def write_snapshot(root: Path, n_sims: int = 12, n_test: int = 2) -> Dataset:
    """A complete interim + processed snapshot of `n_sims` ring simulations; returns it opened."""
    interim = root / "interim" / "airfrans" / SNAPSHOT
    processed = root / "processed" / "airfrans" / SNAPSHOT
    sims_root = interim / "simulations"
    processed.mkdir(parents=True)
    rows, names = [], []
    for i in range(n_sims):
        u_inf, aoa = 30.0 + 5 * i, round(-4.0 + 1.7 * i, 3)
        name = sim_name(u_inf, aoa)
        s = ring_simulation(u_inf, aoa)
        n = len(s["xy"])
        nodes = build_table(
            {
                "node_id": np.arange(n, dtype=np.int32),
                "x": s["xy"][:, 0],
                "y": s["xy"][:, 1],
                "sdf": s["sdf"],
                "is_surface": s["is_surface"],
                "normal_x": s["normals"][:, 0],
                "normal_y": s["normals"][:, 1],
                **{k: s[k] for k in ("u_x", "u_y", "p", "nut")},
            },
            NODES_SCHEMA,
        )
        edges = s["edges"].astype(np.int32)
        segments = s["segments"].astype(np.int32)
        tables = {
            NODES_FILE: nodes,
            EDGES_FILE: build_table({"src": edges[:, 0], "dst": edges[:, 1]}, EDGES_SCHEMA),
            SURFACE_FILE: build_table({"node_a": segments[:, 0], "node_b": segments[:, 1]}, SURFACE_SCHEMA),
        }
        write_simulation(sims_root, name, tables, {"name": name}, time.perf_counter())
        alpha = math.radians(aoa)
        rows.append(
            {
                "name": name, "u_inf": u_inf, "aoa_deg": aoa,
                "inlet_u_x": u_inf * math.cos(alpha), "inlet_u_y": u_inf * math.sin(alpha),
                "reynolds": u_inf / 1.56e-5, "naca_digits": 4, "naca_params": [2.0, 4.0, 12.0],
                "n_nodes": n, "n_edges": len(edges), "n_surface": int(s["is_surface"].sum()),
                "stagnation_ratio": 1.0, "inlet_speed_rel_err": 0.0, "inlet_angle_err_deg": 0.0,
                "read_s": 0.0, "total_s": 0.0,
            }
        )
        names.append(name)

    manifest = {"full_train": names[: n_sims - n_test], "full_test": names[n_sims - n_test :]}
    columns = {key: [row[key] for row in rows] for key in rows[0]}
    for split in MANIFEST_SPLITS:
        columns[split] = [name in manifest.get(split, []) for name in names]
    write_parquet_atomic(build_table(columns, SIMULATIONS_SCHEMA), interim / "simulations.parquet")

    splits = make_splits(manifest)
    write_json(processed / "splits.json", splits_record(splits, RAW_SHA256))
    by_name = {row["name"]: row for row in rows}
    normalisation = fit_normalisation(
        (n, sims_root / n, by_name[n]["inlet_u_x"], by_name[n]["inlet_u_y"]) for n in splits.train
    )
    write_json(processed / "normalisation.json", normalisation)
    write_json(
        interim / "etl.json",
        {"schema_version": SCHEMA_VERSION, "raw_sha256": RAW_SHA256, "status": "complete", "etl_code_commit": None},
    )
    return Dataset.open(interim, processed)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
