"""Read one AirfRANS simulation's three VTK files into plain numpy arrays.

Only VTK's XML readers are used (`vtkmodules.vtkIOXML`), so nothing needs a
display or OpenGL libraries.

Each file is read into memory in one call and VTK decodes it from there. Left
to open the file itself, VTK makes many small reads, and on a Windows folder
bind-mounted into a Linux container each one crosses the mount: 1.2 s per mesh
against 0.28 s this way (median of 10 meshes, identical arrays either way).

VTK readers don't raise on a corrupt file. They log the problem and return an
empty dataset with error code 0, so a truncated download would otherwise flow
on as a mesh with no points. `_read` listens for the reader's error event and
turns it, or an empty result, into a `VtkReadError`. One bad file then fails
its own simulation and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from vtkmodules.util.misc import calldata_type
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import VTK_STRING, vtkCommand
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLUnstructuredGridReader

MESH_SUFFIX = "_internal.vtu"
AEROFOIL_SUFFIX = "_aerofoil.vtp"
FREESTREAM_SUFFIX = "_freestream.vtp"


class VtkReadError(Exception):
    """A VTK file is missing, unreadable or empty."""


@dataclass(frozen=True)
class MeshData:
    """The volume mesh, from `<name>_internal.vtu`."""

    points: np.ndarray  # (n, 3) float64
    #: The distinct VTK cell type ids present (9 is a quad).
    cell_types: np.ndarray
    #: Cell i uses connectivity[offsets[i]:offsets[i + 1]].
    cell_offsets: np.ndarray  # (m + 1,) int64
    cell_connectivity: np.ndarray  # int64
    point_data: dict[str, np.ndarray]

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cell_offsets.shape[0] - 1)


@dataclass(frozen=True)
class PolylineData:
    """A boundary curve, from `<name>_aerofoil.vtp` or `<name>_freestream.vtp`."""

    points: np.ndarray  # (k, 3) float64
    #: Line i uses connectivity[offsets[i]:offsets[i + 1]]; indices are into `points`.
    line_offsets: np.ndarray
    line_connectivity: np.ndarray
    point_data: dict[str, np.ndarray]


@dataclass(frozen=True)
class RawSimulation:
    name: str
    mesh: MeshData
    aerofoil: PolylineData
    freestream: PolylineData


def simulation_files(sim_dir: Path) -> tuple[Path, Path, Path]:
    """The mesh, airfoil-surface and far-field files of one simulation directory."""
    name = sim_dir.name
    return (
        sim_dir / f"{name}{MESH_SUFFIX}",
        sim_dir / f"{name}{AEROFOIL_SUFFIX}",
        sim_dir / f"{name}{FREESTREAM_SUFFIX}",
    )


def _copy(vtk_array: Any) -> np.ndarray:
    # A copy, so the result owns its memory and outlives the VTK object it came from.
    return np.array(vtk_to_numpy(vtk_array))


def _arrays(data: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i in range(data.GetNumberOfArrays()):
        array = data.GetArray(i)
        if array is not None:  # non-numeric arrays come back as None
            out[array.GetName()] = _copy(array)
    return out


def _read(reader: Any, path: Path) -> Any:
    if not path.is_file():
        raise VtkReadError(f"{path.name}: file not found")
    errors: list[str] = []

    @calldata_type(VTK_STRING)
    def on_error(_obj: Any, _event: str, message: str) -> None:
        errors.append(str(message).strip())

    reader.AddObserver(vtkCommand.ErrorEvent, on_error)
    reader.ReadFromInputStringOn()
    reader.SetInputString(path.read_bytes())
    reader.Update()
    output = reader.GetOutput()
    if errors or output is None or output.GetNumberOfPoints() == 0:
        detail = errors[0].splitlines()[0] if errors else "no points"
        raise VtkReadError(f"{path.name}: VTK could not read it ({detail})")
    return output


def read_mesh(path: Path) -> MeshData:
    grid = _read(vtkXMLUnstructuredGridReader(), path)
    cells = grid.GetCells()
    distinct = grid.GetDistinctCellTypesArray()
    return MeshData(
        points=_copy(grid.GetPoints().GetData()),
        cell_types=np.array([distinct.GetValue(i) for i in range(distinct.GetNumberOfTuples())]),
        cell_offsets=_copy(cells.GetOffsetsArray()).astype(np.int64),
        cell_connectivity=_copy(cells.GetConnectivityArray()).astype(np.int64),
        point_data=_arrays(grid.GetPointData()),
    )


def read_polyline(path: Path) -> PolylineData:
    poly = _read(vtkXMLPolyDataReader(), path)
    lines = poly.GetLines()
    return PolylineData(
        points=_copy(poly.GetPoints().GetData()),
        line_offsets=_copy(lines.GetOffsetsArray()).astype(np.int64),
        line_connectivity=_copy(lines.GetConnectivityArray()).astype(np.int64),
        point_data=_arrays(poly.GetPointData()),
    )


def read_simulation(sim_dir: Path) -> RawSimulation:
    mesh_path, aerofoil_path, freestream_path = simulation_files(sim_dir)
    return RawSimulation(
        name=sim_dir.name,
        mesh=read_mesh(mesh_path),
        aerofoil=read_polyline(aerofoil_path),
        freestream=read_polyline(freestream_path),
    )
