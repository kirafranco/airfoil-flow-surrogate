"""Graph edges, surface matching and outward normals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.etl.mesh import derive, match_points, outward_normals, point_index, quad_edges
from src.etl.names import parse_simulation_name
from src.etl.vtk_io import read_simulation
from synthetic_vtk import SURFACE_NODES, make_sim, write_sim


def test_quad_edges_of_a_two_by_two_grid() -> None:
    # 3×3 nodes, 4 quads: 6 horizontal + 6 vertical sides, the 4 inner ones shared.
    quads = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    edges = quad_edges(quads.ravel(), 9)
    expected = {(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8)}
    assert {tuple(e) for e in edges.tolist()} == expected
    assert len(edges) == 12
    assert np.all(edges[:, 0] < edges[:, 1])
    assert edges.tolist() == sorted(edges.tolist())  # lexsorted


def test_exact_matching_has_no_tolerance() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.5]])
    index = point_index(xy)
    query = np.array([[1.0, 0.5], [1.0 + 1e-12, 0.5], [0.0, 0.0]])
    assert match_points(index, query).tolist() == [1, -1, 0]


def test_duplicates_collapse_in_the_index() -> None:
    assert len(point_index(np.array([[0.0, 1.0], [0.0, 1.0], [2.0, 2.0]]))) == 2


def test_outward_normals_flip_and_drop_z() -> None:
    stored = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(outward_normals(stored), [[-1.0, 0.0], [0.0, 1.0]])


@pytest.fixture
def derived(tmp_path: Path):  # noqa: ANN201 - Derived
    sim = make_sim()
    raw = read_simulation(write_sim(sim, tmp_path / "Dataset"))
    return derive(raw, parse_simulation_name(sim.name))


def test_derive_on_the_synthetic_mesh(derived) -> None:  # noqa: ANN001
    # A 3×3 grid has 24 sides; removing the centre quad keeps them all (each is shared).
    assert len(derived.edges) == 24
    assert derived.surface_ids.tolist() == list(SURFACE_NODES)
    assert np.flatnonzero(derived.is_surface).tolist() == sorted(SURFACE_NODES)
    assert derived.duplicate_points == 0
    assert derived.segments.tolist() == [[5, 6], [6, 10], [10, 9], [9, 5]]
    assert np.all(derived.sdf >= 0) and np.all(derived.sdf[list(SURFACE_NODES)] == 0)


def test_normals_point_out_of_the_body(derived) -> None:  # noqa: ANN001
    # Node 5 is the hole's corner at (-0.5, -0.5): outward is towards (-1, -1).
    np.testing.assert_allclose(derived.normals[5], [-np.sqrt(0.5), -np.sqrt(0.5)], rtol=1e-6)
    assert np.all(derived.normals[~derived.is_surface] == 0)


def test_the_gate_measurements(derived) -> None:  # noqa: ANN001
    assert derived.stagnation_ratio == pytest.approx(1.001, rel=1e-6)
    assert derived.inlet_speed_rel_err < 1e-6
    assert derived.inlet_angle_err_deg < 1e-4
