"""Declared schemas, lossless round trip, and the atomic per-simulation writer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.etl.mesh import derive
from src.etl.names import parse_simulation_name
from src.etl.schema import (
    EDGES_FILE,
    EDGES_SCHEMA,
    META_FILE,
    NODES_FILE,
    NODES_SCHEMA,
    SURFACE_FILE,
    SURFACE_SCHEMA,
    edges_table,
    nodes_table,
    read_table,
    surface_table,
    write_parquet,
    write_simulation,
)
from src.etl.vtk_io import read_simulation
from synthetic_vtk import make_sim, write_sim


@pytest.fixture
def written(tmp_path: Path) -> tuple[Path, object]:
    sim = make_sim()
    raw = read_simulation(write_sim(sim, tmp_path / "Dataset"))
    derived = derive(raw, parse_simulation_name(sim.name))
    tables = {
        NODES_FILE: nodes_table(raw, derived),
        EDGES_FILE: edges_table(derived),
        SURFACE_FILE: surface_table(derived),
    }
    sims_root = tmp_path / "simulations"
    write_simulation(sims_root, sim.name, tables, {"name": sim.name}, time.perf_counter())
    return sims_root / sim.name, raw


def test_tables_read_back_with_the_declared_schemas(written) -> None:  # noqa: ANN001
    sim_dir, _ = written
    for filename, schema in ((NODES_FILE, NODES_SCHEMA), (EDGES_FILE, EDGES_SCHEMA), (SURFACE_FILE, SURFACE_SCHEMA)):
        assert read_table(sim_dir / filename, schema).schema.equals(schema, check_metadata=False)


def test_node_columns_equal_the_vtk_arrays_exactly(written) -> None:  # noqa: ANN001
    sim_dir, raw = written
    nodes = read_table(sim_dir / NODES_FILE, NODES_SCHEMA).to_pydict()
    data = raw.mesh.point_data
    np.testing.assert_array_equal(nodes["node_id"], np.arange(raw.mesh.n_points))
    np.testing.assert_array_equal(nodes["x"], raw.mesh.points[:, 0])
    np.testing.assert_array_equal(nodes["y"], raw.mesh.points[:, 1])
    np.testing.assert_array_equal(nodes["sdf"], -data["implicit_distance"])
    np.testing.assert_array_equal(np.float32(nodes["u_x"]), data["U"][:, 0])
    np.testing.assert_array_equal(np.float32(nodes["u_y"]), data["U"][:, 1])
    np.testing.assert_array_equal(np.float32(nodes["p"]), data["p"])
    np.testing.assert_array_equal(np.float32(nodes["nut"]), data["nut"])


def test_index_columns_are_delta_encoded(written) -> None:  # noqa: ANN001
    sim_dir, _ = written
    meta = pq.ParquetFile(sim_dir / EDGES_FILE).metadata
    for i in range(meta.num_columns):
        assert "DELTA_BINARY_PACKED" in meta.row_group(0).column(i).encodings


def test_a_table_with_another_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "edges.parquet"
    write_parquet(pa.table({"src": pa.array([0], pa.int64()), "dst": pa.array([1], pa.int64())}), path)
    with pytest.raises(ValueError, match="schema differs"):
        read_table(path, EDGES_SCHEMA)


def test_the_writer_leaves_one_whole_directory(written) -> None:  # noqa: ANN001
    sim_dir, _ = written
    assert sorted(p.name for p in sim_dir.parent.iterdir()) == [sim_dir.name]
    meta = json.loads((sim_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta["total_s"] > 0


def test_a_leftover_partial_is_replaced(tmp_path: Path) -> None:
    sims_root = tmp_path / "simulations"
    (sims_root / "sim.partial").mkdir(parents=True)
    (sims_root / "sim.partial" / "junk.parquet").write_bytes(b"half")
    table = pa.table({"src": pa.array([0], pa.int32()), "dst": pa.array([1], pa.int32())})
    write_simulation(sims_root, "sim", {EDGES_FILE: table}, {"name": "sim"}, time.perf_counter())
    assert sorted(p.name for p in sims_root.iterdir()) == ["sim"]
    assert sorted(p.name for p in (sims_root / "sim").iterdir()) == [EDGES_FILE, META_FILE]
