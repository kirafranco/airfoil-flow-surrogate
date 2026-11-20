"""The installed environment is exactly the pinned one, and VTK's XML IO works
in a slim image without any rendering libraries."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkCommonDataModel import VTK_QUAD, vtkUnstructuredGrid
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter

from src.config import PROJECT_ROOT


def pins() -> dict[str, str]:
    pinned: dict[str, str] = {}
    for name in ("requirements.txt", "requirements-dev.txt"):
        for line in (PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                package, release = line.strip().split("==")
                pinned[package] = release
    return pinned


@pytest.mark.parametrize("package, pinned", sorted(pins().items()))
def test_installed_version_is_the_pin(package: str, pinned: str) -> None:
    # torch reports its CUDA build as a local tag (2.15.0+cu132); the pin names the release.
    assert version(package).split("+")[0] == pinned


def test_vtk_xml_round_trip_without_rendering(tmp_path: Path) -> None:
    """Write a one-quad unstructured grid with a point array and read it back:
    the same calls the ETL and the ParaView exports make."""
    points = vtkPoints()
    for xyz in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]:
        points.InsertNextPoint(xyz)
    ids = vtkIdList()
    for i in range(4):
        ids.InsertNextId(i)
    grid = vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.InsertNextCell(VTK_QUAD, ids)
    pressure = numpy_to_vtk(np.array([1.0, 2.0, 3.0, 4.0]), deep=True)
    pressure.SetName("p")
    grid.GetPointData().AddArray(pressure)

    path = tmp_path / "quad.vtu"
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    assert writer.Write() == 1

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    out = reader.GetOutput()
    assert out.GetNumberOfPoints() == 4
    assert out.GetNumberOfCells() == 1
    np.testing.assert_array_equal(vtk_to_numpy(out.GetPointData().GetArray("p")), [1.0, 2.0, 3.0, 4.0])
