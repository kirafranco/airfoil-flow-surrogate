"""The interim tables' explicit schemas and the atomic per-simulation writer.

One directory per simulation:

    simulations/<name>/
        nodes.parquet     one row per mesh node, in VTK point order (node_id = VTK point index)
        edges.parquet     one row per undirected mesh edge, src < dst
        surface.parquet   one row per airfoil-surface segment, as mesh node ids
        meta.json         this simulation's row of simulations.parquet, plus timings

A simulation is written into `<name>.partial/` and renamed when complete, so a
directory without the suffix is always whole. That rename is the checkpoint a
re-run relies on.

Every column keeps its source dtype: coordinates and distance in float64, the
flow fields in float32. The ETL changes names and signs (documented per
column), never precision.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.etl.mesh import Derived
from src.etl.vtk_io import RawSimulation

SCHEMA_VERSION = "airfrans-interim/1"

NODES_FILE = "nodes.parquet"
EDGES_FILE = "edges.parquet"
SURFACE_FILE = "surface.parquet"
META_FILE = "meta.json"
PARTIAL_SUFFIX = ".partial"

#: The manifest's splits, as bool columns of simulations.parquet.
MANIFEST_SPLITS = (
    "full_train",
    "full_test",
    "scarce_train",
    "reynolds_train",
    "reynolds_test",
    "aoa_train",
    "aoa_test",
)


def _field(name: str, dtype: pa.DataType) -> pa.Field:
    return pa.field(name, dtype, nullable=False)


NODES_SCHEMA = pa.schema(
    [
        _field("node_id", pa.int32()),
        _field("x", pa.float64()),
        _field("y", pa.float64()),
        _field("sdf", pa.float64()),  # distance to the airfoil, = -implicit_distance
        _field("is_surface", pa.bool_()),
        _field("normal_x", pa.float32()),  # outward unit normal on the surface, 0 elsewhere
        _field("normal_y", pa.float32()),
        _field("u_x", pa.float32()),  # m/s
        _field("u_y", pa.float32()),
        _field("p", pa.float32()),  # kinematic gauge pressure, m²/s²
        _field("nut", pa.float32()),  # turbulent kinematic viscosity, m²/s
    ]
)
EDGES_SCHEMA = pa.schema([_field("src", pa.int32()), _field("dst", pa.int32())])
SURFACE_SCHEMA = pa.schema([_field("node_a", pa.int32()), _field("node_b", pa.int32())])
SIMULATIONS_SCHEMA = pa.schema(
    [
        _field("name", pa.string()),
        _field("u_inf", pa.float64()),
        _field("aoa_deg", pa.float64()),
        _field("inlet_u_x", pa.float64()),
        _field("inlet_u_y", pa.float64()),
        _field("reynolds", pa.float64()),
        _field("naca_digits", pa.int8()),
        _field("naca_params", pa.list_(pa.float64())),
        _field("n_nodes", pa.int32()),
        _field("n_edges", pa.int32()),
        _field("n_surface", pa.int32()),
        *(_field(split, pa.bool_()) for split in MANIFEST_SPLITS),
        _field("stagnation_ratio", pa.float64()),
        _field("inlet_speed_rel_err", pa.float64()),
        _field("inlet_angle_err_deg", pa.float64()),
        _field("read_s", pa.float64()),
        _field("total_s", pa.float64()),
    ]
)

#: Integer columns that are sorted or nearly so: delta encoding shrinks edges ~50×.
_DELTA_COLUMNS = {"node_id", "src", "dst", "node_a", "node_b"}


def build_table(columns: dict[str, Any], schema: pa.Schema) -> pa.Table:
    """A table with exactly the declared schema, from numpy arrays or Python lists."""
    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def nodes_table(raw: RawSimulation, derived: Derived) -> pa.Table:
    data = raw.mesh.point_data
    U = data["U"]
    return build_table(
        {
            "node_id": np.arange(raw.mesh.n_points, dtype=np.int32),
            "x": derived.xy[:, 0],
            "y": derived.xy[:, 1],
            "sdf": derived.sdf,
            "is_surface": derived.is_surface,
            "normal_x": derived.normals[:, 0],
            "normal_y": derived.normals[:, 1],
            "u_x": U[:, 0],
            "u_y": U[:, 1],
            "p": data["p"],
            "nut": data["nut"],
        },
        NODES_SCHEMA,
    )


def edges_table(derived: Derived) -> pa.Table:
    edges = derived.edges.astype(np.int32)
    return build_table({"src": edges[:, 0], "dst": edges[:, 1]}, EDGES_SCHEMA)


def surface_table(derived: Derived) -> pa.Table:
    segments = derived.segments.astype(np.int32)
    return build_table({"node_a": segments[:, 0], "node_b": segments[:, 1]}, SURFACE_SCHEMA)


def write_parquet(table: pa.Table, path: Path) -> None:
    delta = [name for name in table.column_names if name in _DELTA_COLUMNS]
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=[name for name in table.column_names if name not in _DELTA_COLUMNS],
        column_encoding={name: "DELTA_BINARY_PACKED" for name in delta} or None,
    )


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_parquet(table, tmp)
    os.replace(tmp, path)


def read_table(path: Path, schema: pa.Schema, columns: list[str] | None = None) -> pa.Table:
    """Read a table and insist on the declared schema: nothing downstream infers types."""
    table = pq.read_table(path, columns=columns)
    expected = pa.schema([schema.field(name) for name in columns]) if columns else schema
    if not table.schema.equals(expected, check_metadata=False):
        raise ValueError(f"{path}: schema differs from the declared one:\n{table.schema}\nexpected\n{expected}")
    return table


def write_json(path: Path, payload: Any) -> None:
    """Write JSON through a temporary file and a rename, so a reader never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def is_complete(sim_dir: Path) -> bool:
    return (sim_dir / META_FILE).is_file()


def write_simulation(
    sims_root: Path,
    name: str,
    tables: dict[str, pa.Table],
    meta: dict[str, Any],
    started: float,
) -> int:
    """Write one simulation's tables and meta.json atomically; returns the bytes written.

    `started` is the `time.perf_counter()` reading when work on the simulation
    began; meta.json records `total_s` from it, writing included.
    """
    final = sims_root / name
    partial = sims_root / f"{name}{PARTIAL_SUFFIX}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    for filename, table in tables.items():
        write_parquet(table, partial / filename)
    meta = {**meta, "total_s": time.perf_counter() - started}
    write_json(partial / META_FILE, meta)  # last: its presence marks a whole directory
    written = sum(path.stat().st_size for path in partial.iterdir())
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    return written
