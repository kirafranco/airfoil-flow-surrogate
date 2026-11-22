"""The per-simulation quality gate: nothing is written unless every check passes.

Checks run in two phases. G1–G3 (and the cell-reference half of G5) are
structural: if the arrays are missing, not 2D or not quads, the derived graph
and surface mapping would be meaningless, so the gate stops there. Otherwise
`derive` runs once and G4–G10 are all evaluated, so a report lists every
problem, not just the first.

The thresholds come from measurements on the real dataset (2026-11-22, 50
simulations), with a wide margin: for example the stagnation ratio was
1.0007–1.0018 and the gate allows 0.99–1.01.

| check | what it protects |
|---|---|
| G1 | completeness: required arrays present, expected dtypes, all finite |
| G2 | the mesh is a 2D slice: constant z, out-of-plane velocity at round-off level |
| G3 | every cell is a quad, so edges come from quad sides |
| G4 | unique node coordinates |
| G5 | edges reference real nodes; no self-loops; no isolated node |
| G6 | the airfoil surface is a subset of the mesh, and three independent markers agree |
| G7 | the surface is one closed loop of 2-point segments |
| G8 | stored normals become outward ones (leading-edge normal points upstream) |
| G9 | the folder name agrees with the physics in the file |
| G10 | turbulent viscosity is non-negative |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from vtkmodules.vtkCommonDataModel import VTK_QUAD

from src.etl.mesh import Derived, derive
from src.etl.names import SimulationParams
from src.etl.vtk_io import RawSimulation

STAGNATION_RANGE = (0.99, 1.01)
#: U_z is solver round-off, not zero: up to 1.3e-10 m/s, at most 2.8e-12 of U∞
#: (50 sims). A genuinely 3D field would be many orders of magnitude above this.
OUT_OF_PLANE_REL_TOL = 1e-6
INLET_SPEED_REL_TOL = 1e-3
INLET_ANGLE_TOL_DEG = 0.05

#: Arrays the ETL stores, with the dtype and component count the release uses.
#: A different dtype fails G1 instead of being cast, so the ETL stays lossless.
MESH_ARRAYS = {
    "U": (np.float32, 3),
    "p": (np.float32, 1),
    "nut": (np.float32, 1),
    "implicit_distance": (np.float64, 1),
}
AEROFOIL_ARRAYS = {"Normals": (np.float32, 3)}
FREESTREAM_ARRAYS = {"U": (np.float32, 3)}
#: The mesh stores coordinates as float64, but every value is an exact float32
#: number; the boundary files store float32. Comparing a float32 boundary point
#: with a float64 mesh point is therefore exact, with no rounding involved.
POINTS_DTYPES = {"mesh": np.float64, "aerofoil": np.float32, "freestream": np.float32}


@dataclass(frozen=True)
class Violation:
    check: str  # "G1" … "G10"
    detail: str


@dataclass(frozen=True)
class GateResult:
    violations: tuple[Violation, ...]
    derived: Derived | None

    @property
    def passed(self) -> bool:
        return not self.violations


def _array_violations(where: str, arrays: dict[str, np.ndarray], n: int, spec: dict) -> list[Violation]:
    found: list[Violation] = []
    for name, (dtype, components) in spec.items():
        array = arrays.get(name)
        expected_shape = (n,) if components == 1 else (n, components)
        if array is None:
            found.append(Violation("G1", f"{where}: array {name!r} missing"))
        elif array.dtype != dtype:
            found.append(Violation("G1", f"{where}: {name!r} is {array.dtype}, expected {np.dtype(dtype)}"))
        elif array.shape != expected_shape:
            found.append(Violation("G1", f"{where}: {name!r} has shape {array.shape}, expected {expected_shape}"))
        elif not np.isfinite(array).all():
            bad = int((~np.isfinite(array)).sum())
            found.append(Violation("G1", f"{where}: {name!r} has {bad} NaN or infinite value(s)"))
    return found


def _structural(raw: RawSimulation, params: SimulationParams) -> list[Violation]:
    mesh, aerofoil, freestream = raw.mesh, raw.aerofoil, raw.freestream
    found: list[Violation] = []
    for where, points in (("mesh", mesh.points), ("aerofoil", aerofoil.points), ("freestream", freestream.points)):
        dtype = np.dtype(POINTS_DTYPES[where])
        if points.dtype != dtype:
            found.append(Violation("G1", f"{where}: points are {points.dtype}, expected {dtype}"))
        elif not np.isfinite(points).all():
            found.append(Violation("G1", f"{where}: points include NaN or infinite values"))
    found += _array_violations("mesh", mesh.point_data, mesh.n_points, MESH_ARRAYS)
    found += _array_violations("aerofoil", aerofoil.point_data, len(aerofoil.points), AEROFOIL_ARRAYS)
    found += _array_violations("freestream", freestream.point_data, len(freestream.points), FREESTREAM_ARRAYS)
    if found:
        return found

    z = mesh.points[:, 2]
    if np.ptp(z) != 0:
        found.append(Violation("G2", f"z varies from {z.min()} to {z.max()}: not a 2D slice"))
    u_z = float(np.abs(mesh.point_data["U"][:, 2]).max())
    if u_z > OUT_OF_PLANE_REL_TOL * params.u_inf:
        found.append(Violation("G2", f"out-of-plane velocity up to {u_z:.3e} m/s, more than round-off"))

    sizes = np.diff(mesh.cell_offsets)
    if set(mesh.cell_types.tolist()) != {VTK_QUAD} or np.any(sizes != 4):
        found.append(
            Violation("G3", f"cell types {sorted(mesh.cell_types.tolist())}, sizes {sorted(set(sizes.tolist()))}: expected quads only")
        )
    conn = mesh.cell_connectivity
    if conn.size and (conn.min() < 0 or conn.max() >= mesh.n_points):
        found.append(Violation("G5", f"a cell references node {int(conn.max())}, but the mesh has {mesh.n_points}"))
    return found


def _derived(raw: RawSimulation, params: SimulationParams, d: Derived) -> list[Violation]:
    mesh, aerofoil = raw.mesh, raw.aerofoil
    n = mesh.n_points
    found: list[Violation] = []

    if d.duplicate_points:
        found.append(Violation("G4", f"{d.duplicate_points} duplicate (x, y) point(s)"))

    loops = int(np.sum(d.edges[:, 0] == d.edges[:, 1]))
    if loops:
        found.append(Violation("G5", f"{loops} self-loop edge(s)"))
    isolated = int(np.sum(np.bincount(d.edges.ravel(), minlength=n) == 0))
    if isolated:
        found.append(Violation("G5", f"{isolated} node(s) in no edge"))

    unmatched = int(np.sum(d.surface_ids < 0))
    matched = d.surface_ids[d.surface_ids >= 0]
    if unmatched:
        found.append(Violation("G6", f"{unmatched} airfoil point(s) match no mesh node exactly"))
    if len(np.unique(matched)) != len(matched):
        found.append(Violation("G6", "two airfoil points match the same mesh node"))
    U = mesh.point_data["U"]
    no_slip = (U[:, 0] == 0) & (U[:, 1] == 0)
    on_wall = mesh.point_data["implicit_distance"] == 0
    if not np.array_equal(no_slip, d.is_surface):
        found.append(
            Violation("G6", f"{int(no_slip.sum())} nodes with U = 0 vs {int(d.is_surface.sum())} surface nodes")
        )
    if not np.array_equal(on_wall, d.is_surface):
        found.append(
            Violation("G6", f"{int(on_wall.sum())} nodes at distance 0 vs {int(d.is_surface.sum())} surface nodes")
        )
    positive = int(np.sum(mesh.point_data["implicit_distance"] > 0))
    if positive:
        found.append(Violation("G6", f"{positive} node(s) with positive implicit_distance (inside the airfoil)"))

    line_sizes = np.diff(aerofoil.line_offsets)
    if np.any(line_sizes != 2):
        found.append(Violation("G7", f"surface lines of sizes {sorted(set(line_sizes.tolist()))}: expected 2"))
    else:
        degree = np.bincount(aerofoil.line_connectivity, minlength=len(aerofoil.points))
        if np.any(degree != 2):
            found.append(Violation("G7", f"{int(np.sum(degree != 2))} surface point(s) not in exactly 2 segments"))

    if matched.size:
        leading = matched[np.argmin(d.xy[matched, 0])]
        if not d.normals[leading, 0] < 0:
            found.append(
                Violation("G8", f"leading-edge normal x = {d.normals[leading, 0]:.4f} after the flip: expected < 0")
            )

    low, high = STAGNATION_RANGE
    if not low <= d.stagnation_ratio <= high:
        found.append(
            Violation("G9", f"stagnation ratio {d.stagnation_ratio:.4f} outside [{low}, {high}] for U∞ = {params.u_inf}")
        )
    if not d.inlet_speed_rel_err < INLET_SPEED_REL_TOL:
        found.append(Violation("G9", f"inlet speed differs from U∞ = {params.u_inf} by {d.inlet_speed_rel_err:.2e}"))
    if not d.inlet_angle_err_deg < INLET_ANGLE_TOL_DEG:
        found.append(
            Violation("G9", f"inlet angle differs from AoA = {params.aoa_deg}° by {d.inlet_angle_err_deg:.4f}°")
        )

    nut_min = float(mesh.point_data["nut"].min())
    if nut_min < 0:
        found.append(Violation("G10", f"negative turbulent viscosity, min {nut_min}"))
    return found


def check_simulation(raw: RawSimulation, params: SimulationParams) -> GateResult:
    structural = _structural(raw, params)
    if structural:
        return GateResult(tuple(structural), None)
    derived = derive(raw, params)
    return GateResult(tuple(_derived(raw, params, derived)), derived)
