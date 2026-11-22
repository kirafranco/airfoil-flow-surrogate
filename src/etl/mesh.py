"""Array transforms from a raw simulation to the columns of its tables.

Nothing here judges the data. `derive` computes the graph, the surface mapping,
the outward normals and the gate's measurements once. `src.etl.quality` decides
whether the result is acceptable, and `src.etl.schema` writes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.etl.names import SimulationParams
from src.etl.vtk_io import PolylineData, RawSimulation

#: The far-field boundary's upstream half (x < 0) carries the undisturbed inflow.
UPSTREAM_X = 0.0


@dataclass(frozen=True)
class Derived:
    xy: np.ndarray  # (n, 2) float64, the mesh points without the constant z
    sdf: np.ndarray  # (n,) float64, distance to the airfoil: -implicit_distance
    edges: np.ndarray  # (E, 2) int64, unique undirected quad sides, src < dst, lexsorted
    #: Mesh node of each airfoil-surface point, or -1 where none matches exactly.
    surface_ids: np.ndarray
    is_surface: np.ndarray  # (n,) bool
    normals: np.ndarray  # (n, 2) float32, outward on surface nodes, 0 elsewhere
    #: Surface line segments as pairs of mesh node ids; empty unless every line has 2 points.
    segments: np.ndarray
    duplicate_points: int
    #: max p / (½U∞²): 1 at the stagnation point when p is kinematic gauge pressure.
    stagnation_ratio: float
    #: Median relative speed error and median angle error (degrees) of the upstream
    #: far-field velocity against the U∞ and angle of attack in the folder name.
    inlet_speed_rel_err: float
    inlet_angle_err_deg: float


def quad_edges(connectivity: np.ndarray, n_points: int) -> np.ndarray:
    """Unique undirected edges of a mesh made of quads, as (src, dst) with src < dst, lexsorted.

    Quad a-b-c-d contributes its four sides a-b, b-c, c-d and d-a; a side shared
    by two quads appears once. Each pair is packed into one int64 (src·n + dst),
    so deduplication is a 1-D `np.unique` instead of a much slower row-wise one.
    """
    quads = connectivity.reshape(-1, 4).astype(np.int64)
    pairs = np.stack([quads, np.roll(quads, -1, axis=1)], axis=-1).reshape(-1, 2)
    keys = np.unique(pairs.min(axis=1) * n_points + pairs.max(axis=1))
    return np.stack([keys // n_points, keys % n_points], axis=1)


def point_index(xy: np.ndarray) -> dict[tuple[float, float], int]:
    """Map each (x, y) to its row. Duplicates collapse, so `len()` < rows reveals them."""
    return dict(zip(map(tuple, xy.tolist()), range(len(xy))))


def match_points(index: dict[tuple[float, float], int], query_xy: np.ndarray) -> np.ndarray:
    """The row of each query point by exact coordinate equality, -1 where there is none.

    Exact equality, no tolerance: the surface file's points are copies of mesh
    points, not recomputed coordinates, so a near-miss means a different point.
    """
    return np.array([index.get(key, -1) for key in map(tuple, query_xy.tolist())], dtype=np.int64)


def outward_normals(stored: np.ndarray) -> np.ndarray:
    """The surface file's normals point into the airfoil; flip them to point into the fluid."""
    return (-stored[:, :2]).astype(np.float32)


def _upstream_inflow_errors(freestream: PolylineData, params: SimulationParams) -> tuple[float, float]:
    upstream = freestream.points[:, 0] < UPSTREAM_X
    velocity = freestream.point_data["U"][upstream, :2].astype(np.float64)
    if velocity.size == 0:
        return float("inf"), float("inf")
    speed = np.linalg.norm(velocity, axis=1)
    angle = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0]))
    return (
        float(np.median(np.abs(speed - params.u_inf)) / params.u_inf),
        float(np.median(np.abs(angle - params.aoa_deg))),
    )


def derive(raw: RawSimulation, params: SimulationParams) -> Derived:
    """Everything the gate checks and the writer stores. Expects the structural checks to have passed."""
    mesh, aerofoil = raw.mesh, raw.aerofoil
    n = mesh.n_points
    xy = np.ascontiguousarray(mesh.points[:, :2])
    index = point_index(xy)

    surface_ids = match_points(index, aerofoil.points[:, :2])
    matched = surface_ids >= 0
    is_surface = np.zeros(n, dtype=bool)
    is_surface[surface_ids[matched]] = True
    normals = np.zeros((n, 2), dtype=np.float32)
    normals[surface_ids[matched]] = outward_normals(aerofoil.point_data["Normals"])[matched]

    if np.all(np.diff(aerofoil.line_offsets) == 2):
        segments = surface_ids[aerofoil.line_connectivity.reshape(-1, 2)]
    else:
        segments = np.empty((0, 2), dtype=np.int64)

    speed_err, angle_err = _upstream_inflow_errors(raw.freestream, params)
    p = mesh.point_data["p"]
    return Derived(
        xy=xy,
        sdf=-mesh.point_data["implicit_distance"],
        edges=quad_edges(mesh.cell_connectivity, n),
        surface_ids=surface_ids,
        is_surface=is_surface,
        normals=normals,
        segments=segments,
        duplicate_points=n - len(index),
        stagnation_ratio=float(p.max()) / (0.5 * params.u_inf**2),
        inlet_speed_rel_err=speed_err,
        inlet_angle_err_deg=angle_err,
    )
