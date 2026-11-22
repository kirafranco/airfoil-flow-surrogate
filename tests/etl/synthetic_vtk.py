"""Tiny AirfRANS-shaped simulations, written with VTK's own XML writers, for the ETL tests.

The mesh is a 4×4-node grid (3×3 quads) with the centre quad removed: 16 nodes
and 8 quads. The hole's four corners play the airfoil. They form a closed loop
of 4 surface segments where U = 0 and implicit_distance = 0, and the stored
normals point into the hole, as they do in the real files. The outer ring of 12
nodes is the far field, carrying the inflow U∞(cos α, sin α). One surface node
holds the stagnation pressure ½U∞² · 1.001.

Dtypes match the real release: mesh points float64, boundary points float32,
U / p / nut / Normals float32, implicit_distance float64. Files are written in
the release's format: inline binary, zlib-compressed.

Every array is a plain attribute of `SyntheticSim`, so a test breaks exactly
one thing before writing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkCommonDataModel import VTK_QUAD, VTK_TRIANGLE, vtkCellArray, vtkPolyData, vtkUnstructuredGrid
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter, vtkXMLUnstructuredGridWriter

U_INF = 40.0
AOA_DEG = 5.0
COORDS = (-1.5, -0.5, 0.5, 1.5)
Z = 0.5
#: Node ids of the hole's corners, counter-clockwise from (-0.5, -0.5).
SURFACE_NODES = (5, 6, 10, 9)
#: The outer ring of the grid, counter-clockwise from (-1.5, -1.5).
FAR_FIELD_NODES = (0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4)
STAGNATION_NODE = 5
SNAPSHOT = "2026-11-20"
RAW_SHA256 = "ab" * 32


def sim_name(u_inf: float = U_INF, aoa_deg: float = AOA_DEG, naca: tuple[float, ...] = (2.0, 4.0, 12.0)) -> str:
    return "_".join(["airFoil2D_SST", f"{u_inf}", f"{aoa_deg}", *(f"{v}" for v in naca)])


def _node(i: int, j: int) -> int:
    return j * 4 + i


def _distance_to_hole(x: float, y: float) -> float:
    dx = max(abs(x) - 0.5, 0.0)
    dy = max(abs(y) - 0.5, 0.0)
    return math.hypot(dx, dy)


@dataclass
class SyntheticSim:
    name: str
    points: np.ndarray
    cells: list[tuple[int, ...]]
    U: np.ndarray
    p: np.ndarray
    nut: np.ndarray
    implicit_distance: np.ndarray
    aero_points: np.ndarray
    aero_lines: list[tuple[int, int]]
    aero_normals: np.ndarray
    free_points: np.ndarray
    free_lines: list[tuple[int, int]]
    free_U: np.ndarray
    #: Mesh point arrays to leave out of the file.
    drop: set[str] = field(default_factory=set)


def make_sim(name: str | None = None, *, u_inf: float = U_INF, aoa_deg: float = AOA_DEG) -> SyntheticSim:
    """A clean simulation. `u_inf` and `aoa_deg` set the physics; `name` can disagree with them."""
    name = name or sim_name(u_inf, aoa_deg)
    points = np.array([(x, y, Z) for y in COORDS for x in COORDS], dtype=np.float64)
    cells = [
        (_node(i, j), _node(i + 1, j), _node(i + 1, j + 1), _node(i, j + 1))
        for j in range(3)
        for i in range(3)
        if (i, j) != (1, 1)
    ]
    alpha = math.radians(aoa_deg)
    inflow = np.array([u_inf * math.cos(alpha), u_inf * math.sin(alpha), 0.0], dtype=np.float32)
    n = len(points)
    surface = list(SURFACE_NODES)

    U = np.tile(inflow, (n, 1)).astype(np.float32)
    U[surface] = 0.0
    q = 0.5 * u_inf**2
    p = np.full(n, 0.3 * q, dtype=np.float32)
    p[surface] = 0.2 * q
    p[STAGNATION_NODE] = 1.001 * q
    nut = np.full(n, 1e-4, dtype=np.float32)
    nut[surface] = 0.0
    implicit_distance = np.array([-_distance_to_hole(x, y) for x, y, _ in points], dtype=np.float64)

    aero_points = points[surface].astype(np.float32)
    centre = np.array([0.0, 0.0, Z])
    inward = centre - points[surface]
    aero_normals = (inward / np.linalg.norm(inward, axis=1, keepdims=True)).astype(np.float32)
    aero_normals[:, 2] = 0.0

    free_points = points[list(FAR_FIELD_NODES)].astype(np.float32)
    k = len(FAR_FIELD_NODES)
    return SyntheticSim(
        name=name,
        points=points,
        cells=cells,
        U=U,
        p=p,
        nut=nut,
        implicit_distance=implicit_distance,
        aero_points=aero_points,
        aero_lines=[(i, (i + 1) % 4) for i in range(4)],
        aero_normals=aero_normals,
        free_points=free_points,
        free_lines=[(i, (i + 1) % k) for i in range(k)],
        free_U=np.tile(inflow, (k, 1)).astype(np.float32),
    )


def _vtk_array(name: str, values: np.ndarray):  # noqa: ANN202 - a VTK data array
    array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    array.SetName(name)
    return array


def _vtk_points(values: np.ndarray) -> vtkPoints:
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(values), deep=True))
    return points


def _write(writer, data, path: Path) -> None:  # noqa: ANN001 - VTK objects
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise RuntimeError(f"VTK failed to write {path}")


def _polyline(points: np.ndarray, lines: list[tuple[int, int]], arrays: dict[str, np.ndarray]) -> vtkPolyData:
    poly = vtkPolyData()
    poly.SetPoints(_vtk_points(points))
    cells = vtkCellArray()
    for a, b in lines:
        ids = vtkIdList()
        ids.InsertNextId(a)
        ids.InsertNextId(b)
        cells.InsertNextCell(ids)
    poly.SetLines(cells)
    for name, values in arrays.items():
        poly.GetPointData().AddArray(_vtk_array(name, values))
    return poly


def write_sim(sim: SyntheticSim, dataset_dir: Path) -> Path:
    """Write the three files under `dataset_dir/<name>/`; returns that directory."""
    sim_dir = dataset_dir / sim.name
    sim_dir.mkdir(parents=True, exist_ok=True)

    grid = vtkUnstructuredGrid()
    grid.SetPoints(_vtk_points(sim.points))
    for cell in sim.cells:
        ids = vtkIdList()
        for node in cell:
            ids.InsertNextId(node)
        grid.InsertNextCell(VTK_QUAD if len(cell) == 4 else VTK_TRIANGLE, ids)
    arrays = {"U": sim.U, "p": sim.p, "nut": sim.nut, "implicit_distance": sim.implicit_distance}
    for name, values in arrays.items():
        if name not in sim.drop:
            grid.GetPointData().AddArray(_vtk_array(name, values))
    _write(vtkXMLUnstructuredGridWriter(), grid, sim_dir / f"{sim.name}_internal.vtu")

    aero_U = np.zeros((len(sim.aero_points), 3), dtype=np.float32)
    aero = _polyline(sim.aero_points, sim.aero_lines, {"Normals": sim.aero_normals, "U": aero_U})
    _write(vtkXMLPolyDataWriter(), aero, sim_dir / f"{sim.name}_aerofoil.vtp")
    free = _polyline(sim.free_points, sim.free_lines, {"U": sim.free_U})
    _write(vtkXMLPolyDataWriter(), free, sim_dir / f"{sim.name}_freestream.vtp")
    return sim_dir


def write_raw_snapshot(
    raw_root: Path,
    sims: list[SyntheticSim],
    *,
    train: list[str],
    test: list[str],
    extra_splits: dict[str, list[str]] | None = None,
    sha256: str = RAW_SHA256,
) -> Path:
    """A raw snapshot laid out as the ingestion leaves it: source.json and Dataset/ with a manifest."""
    snapshot = raw_root / SNAPSHOT
    dataset = snapshot / "Dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    for sim in sims:
        write_sim(sim, dataset)
    manifest = {"full_train": train, "full_test": test, **(extra_splits or {})}
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (snapshot / "source.json").write_text(json.dumps({"sha256": sha256}), encoding="utf-8")
    return snapshot


def dataset_of(n: int) -> list[SyntheticSim]:
    """`n` clean simulations with distinct names, mixing 4- and 5-digit NACA names."""
    sims = []
    for i in range(n):
        naca = (2.0, 4.0, 12.0) if i % 2 == 0 else (1.5, 3.0, 0.0, 15.0)
        u_inf, aoa = 30.0 + 7 * i, -4.0 + 1.5 * i
        sims.append(make_sim(sim_name(u_inf, aoa, naca), u_inf=u_inf, aoa_deg=aoa))
    return sims


# --- one broken copy per gate check: (mutation, check expected to fail) ---------


def _nan_pressure(s: SyntheticSim) -> None:
    s.p[3] = np.nan


def _missing_nut(s: SyntheticSim) -> None:
    s.drop.add("nut")


def _pressure_float64(s: SyntheticSim) -> None:
    s.p = s.p.astype(np.float64)


def _z_varies(s: SyntheticSim) -> None:
    s.points[0, 2] = Z + 0.1


def _out_of_plane_velocity(s: SyntheticSim) -> None:
    s.U[0, 2] = 1.0


def _triangle_cell(s: SyntheticSim) -> None:
    s.cells[0] = s.cells[0][:3]


def _duplicate_point(s: SyntheticSim) -> None:
    s.points[0] = s.points[1]


def _isolated_node(s: SyntheticSim) -> None:
    s.points = np.vstack([s.points, [[3.0, 3.0, Z]]])
    s.U = np.vstack([s.U, s.U[:1]])
    s.p = np.append(s.p, s.p[0])
    s.nut = np.append(s.nut, s.nut[0])
    s.implicit_distance = np.append(s.implicit_distance, -3.5)


def _self_loop(s: SyntheticSim) -> None:
    a, _, c, d = s.cells[0]
    s.cells[0] = (a, a, c, d)


def _surface_point_off_mesh(s: SyntheticSim) -> None:
    s.aero_points[1, 0] += np.float32(1e-3)


def _surface_node_moving(s: SyntheticSim) -> None:
    s.U[SURFACE_NODES[1]] = (0.1, 0.0, 0.0)


def _positive_distance(s: SyntheticSim) -> None:
    s.implicit_distance[0] = 0.1


def _open_surface_loop(s: SyntheticSim) -> None:
    s.aero_lines = s.aero_lines[:-1]


def _normals_stored_outward(s: SyntheticSim) -> None:
    s.aero_normals = -s.aero_normals


def _negative_nut(s: SyntheticSim) -> None:
    s.nut[0] = -1e-6


GATE_CASES = {
    "nan_pressure": (_nan_pressure, "G1"),
    "missing_nut": (_missing_nut, "G1"),
    "pressure_float64": (_pressure_float64, "G1"),
    "z_varies": (_z_varies, "G2"),
    "out_of_plane_velocity": (_out_of_plane_velocity, "G2"),
    "triangle_cell": (_triangle_cell, "G3"),
    "duplicate_point": (_duplicate_point, "G4"),
    "isolated_node": (_isolated_node, "G5"),
    "self_loop": (_self_loop, "G5"),
    "surface_point_off_mesh": (_surface_point_off_mesh, "G6"),
    "surface_node_moving": (_surface_node_moving, "G6"),
    "positive_distance": (_positive_distance, "G6"),
    "open_surface_loop": (_open_surface_loop, "G7"),
    "normals_stored_outward": (_normals_stored_outward, "G8"),
    "negative_nut": (_negative_nut, "G10"),
}

#: Folder names that disagree with the physics in the files (G9).
NAME_CASES = {
    "speed_5_percent_off": sim_name(U_INF * 1.05, AOA_DEG),
    "angle_1_degree_off": sim_name(U_INF, AOA_DEG + 1.0),
}
