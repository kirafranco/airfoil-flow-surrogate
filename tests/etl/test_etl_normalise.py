"""Streaming statistics equal the all-at-once ones."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.etl.mesh import derive
from src.etl.names import parse_simulation_name
from src.etl.normalise import RunningStats, fit_normalisation
from src.etl.schema import EDGES_FILE, NODES_FILE, SURFACE_FILE, edges_table, nodes_table, surface_table, write_simulation
from src.etl.vtk_io import read_simulation
from synthetic_vtk import dataset_of, write_sim


def test_merged_batches_equal_numpy_on_the_concatenation() -> None:
    rng = np.random.default_rng(0)
    batches = [rng.normal(5.0, 2.0, size=size) for size in (1, 17, 1000, 3)]
    stats = RunningStats()
    for batch in batches:
        stats.add(batch)
    stats.add_constant(7.5, 40)
    everything = np.concatenate([*batches, np.full(40, 7.5)])
    assert stats.count == everything.size
    assert stats.mean == pytest.approx(everything.mean(), rel=1e-12)
    assert stats.std == pytest.approx(everything.std(), rel=1e-12)
    assert (stats.min, stats.max) == (everything.min(), everything.max())


def test_a_large_offset_does_not_lose_the_variance() -> None:
    # The naive E[x²] − E[x]² cancels catastrophically here; the pairwise merge doesn't.
    rng = np.random.default_rng(1)
    batches = [1e8 + rng.normal(0.0, 1e-3, size=500) for _ in range(20)]
    stats = RunningStats()
    for batch in batches:
        stats.add(batch)
    assert stats.std == pytest.approx(np.concatenate(batches).std(), rel=1e-6)


def test_fit_reads_only_the_simulations_it_is_given(tmp_path: Path) -> None:
    sims_root = tmp_path / "simulations"
    given = []
    for sim in dataset_of(3):
        raw = read_simulation(write_sim(sim, tmp_path / "Dataset"))
        params = parse_simulation_name(sim.name)
        derived = derive(raw, params)
        tables = {NODES_FILE: nodes_table(raw, derived), EDGES_FILE: edges_table(derived), SURFACE_FILE: surface_table(derived)}
        write_simulation(sims_root, sim.name, tables, {"name": sim.name}, time.perf_counter())
        given.append((sim.name, sims_root / sim.name, params.inlet_u_x, params.inlet_u_y, raw))

    result = fit_normalisation([g[:4] for g in given[:2]], sample_per_sim=8)
    p = np.concatenate([g[4].mesh.point_data["p"].astype(np.float64) for g in given[:2]])
    assert result["fitted_on"]["simulations"] == 2
    assert result["channels"]["p"]["count"] == 32
    assert result["channels"]["p"]["mean"] == pytest.approx(p.mean(), rel=1e-12)
    assert result["channels"]["inlet_u_x"]["mean"] == pytest.approx((given[0][2] + given[1][2]) / 2)
    assert result["nut_quantile_sample"]["size"] == 16
    assert set(result["nut_quantiles"]) == {"q01", "q05", "q50", "q95", "q99"}
