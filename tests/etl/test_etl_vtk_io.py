"""Reading the three VTK files, and turning VTK's silent failures into errors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from vtkmodules.vtkCommonDataModel import VTK_QUAD

from src.etl.vtk_io import VtkReadError, read_mesh, read_polyline, read_simulation, simulation_files
from synthetic_vtk import SURFACE_NODES, make_sim, write_sim


@pytest.fixture
def sim_dir(tmp_path: Path) -> Path:
    return write_sim(make_sim(), tmp_path / "Dataset")


def test_reads_the_mesh_with_its_dtypes(sim_dir: Path) -> None:
    sim = make_sim()
    raw = read_simulation(sim_dir)

    assert raw.name == sim.name
    assert raw.mesh.n_points == 16 and raw.mesh.n_cells == 8
    assert raw.mesh.cell_types.tolist() == [VTK_QUAD]
    assert raw.mesh.points.dtype == np.float64
    np.testing.assert_array_equal(raw.mesh.points, sim.points)
    for name, dtype in (("U", np.float32), ("p", np.float32), ("nut", np.float32), ("implicit_distance", np.float64)):
        assert raw.mesh.point_data[name].dtype == dtype, name
    np.testing.assert_array_equal(raw.mesh.point_data["p"], sim.p)


def test_reads_both_boundaries_as_closed_polylines(sim_dir: Path) -> None:
    raw = read_simulation(sim_dir)
    assert raw.aerofoil.points.dtype == np.float32
    assert len(raw.aerofoil.points) == len(SURFACE_NODES)
    assert np.diff(raw.aerofoil.line_offsets).tolist() == [2, 2, 2, 2]
    assert raw.freestream.point_data["U"].shape == (12, 3)


def test_a_truncated_file_raises_instead_of_returning_an_empty_mesh(sim_dir: Path) -> None:
    mesh_path, _, _ = simulation_files(sim_dir)
    data = mesh_path.read_bytes()
    mesh_path.write_bytes(data[: len(data) // 2])
    with pytest.raises(VtkReadError, match="could not read"):
        read_mesh(mesh_path)


def test_a_file_that_is_not_vtk_raises(tmp_path: Path) -> None:
    path = tmp_path / "x_aerofoil.vtp"
    path.write_text("<html>maintenance</html>")
    with pytest.raises(VtkReadError):
        read_polyline(path)


def test_a_missing_file_raises(sim_dir: Path) -> None:
    _, _, freestream = simulation_files(sim_dir)
    freestream.unlink()
    with pytest.raises(VtkReadError, match="not found"):
        read_simulation(sim_dir)
