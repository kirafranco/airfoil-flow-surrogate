"""The quality gate: a clean simulation passes, and each corrupted copy fails the right check."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.etl.names import parse_simulation_name
from src.etl.quality import check_simulation
from src.etl.vtk_io import read_simulation
from synthetic_vtk import AOA_DEG, GATE_CASES, NAME_CASES, U_INF, make_sim, write_sim


def _gate(sim, tmp_path: Path):  # noqa: ANN001, ANN202
    raw = read_simulation(write_sim(sim, tmp_path / "Dataset"))
    return check_simulation(raw, parse_simulation_name(sim.name))


def test_a_clean_simulation_passes(tmp_path: Path) -> None:
    result = _gate(make_sim(), tmp_path)
    assert result.passed, result.violations
    assert result.derived is not None


@pytest.mark.parametrize("case", sorted(GATE_CASES))
def test_each_corruption_fails_its_check(case: str, tmp_path: Path) -> None:
    mutate, expected = GATE_CASES[case]
    sim = make_sim()
    mutate(sim)
    result = _gate(sim, tmp_path)
    assert not result.passed
    assert expected in {v.check for v in result.violations}, result.violations


@pytest.mark.parametrize("case", sorted(NAME_CASES))
def test_a_folder_name_that_disagrees_with_the_file_fails_g9(case: str, tmp_path: Path) -> None:
    # The files hold a U_INF, AOA_DEG flow; only the name changes. A wrong speed
    # trips both the stagnation ratio and the inlet speed; nothing else fails.
    sim = make_sim(NAME_CASES[case], u_inf=U_INF, aoa_deg=AOA_DEG)
    result = _gate(sim, tmp_path)
    assert {v.check for v in result.violations} == {"G9"}, result.violations


def test_structural_failures_stop_before_deriving(tmp_path: Path) -> None:
    mutate, _ = GATE_CASES["triangle_cell"]
    sim = make_sim()
    mutate(sim)
    result = _gate(sim, tmp_path)
    assert result.derived is None
    assert [v.check for v in result.violations] == ["G3"]
