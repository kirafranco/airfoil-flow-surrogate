"""The ETL end to end on synthetic snapshots: outputs, fault tolerance, checkpoints, idempotency, leakage."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

import src.etl.pipeline as pipeline
from src.etl.pipeline import EtlError, EtlResult, run
from src.etl.schema import NODES_FILE, NODES_SCHEMA, SIMULATIONS_SCHEMA, read_table
from src.etl.splits import make_splits
from src.etl.vtk_io import read_simulation
from synthetic_vtk import SNAPSHOT, SURFACE_NODES, dataset_of, write_raw_snapshot, write_sim

N_SIMS = 10


@pytest.fixture(autouse=True)
def _info_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)


@pytest.fixture
def roots(tmp_path: Path) -> dict[str, Path]:
    return {
        "raw_root": tmp_path / "raw" / "airfrans",
        "interim_root": tmp_path / "interim",
        "processed_root": tmp_path / "processed",
    }


def _snapshot(roots: dict[str, Path], sims=None, **kwargs) -> list:  # noqa: ANN001
    sims = sims if sims is not None else dataset_of(N_SIMS)
    names = [s.name for s in sims]
    kwargs.setdefault("train", names[:8])
    kwargs.setdefault("test", names[8:])
    write_raw_snapshot(roots["raw_root"], sims, **kwargs)
    return sims


def _run(roots: dict[str, Path], **kwargs) -> EtlResult:  # noqa: ANN003
    return run(**roots, workers=kwargs.pop("workers", 1), **kwargs)


def _interim(roots: dict[str, Path]) -> Path:
    return roots["interim_root"] / "airfrans" / SNAPSHOT


def _processed(roots: dict[str, Path]) -> Path:
    return roots["processed_root"] / "airfrans" / SNAPSHOT


def _mtimes(*dirs: Path) -> dict[Path, int]:
    return {p: p.stat().st_mtime_ns for d in dirs for p in d.rglob("*")}


class TestCompleteRun:
    def test_writes_every_output(self, roots: dict[str, Path]) -> None:
        sims = _snapshot(roots, extra_splits={"scarce_train": [], "aoa_test": []})
        names = [s.name for s in sims]
        result = _run(roots)

        assert (result.processed, result.skipped, result.failures, result.wrote_dataset) == (10, 0, (), True)
        table = pq.read_table(_interim(roots) / "simulations.parquet")
        assert table.schema.equals(SIMULATIONS_SCHEMA, check_metadata=False)
        assert table.column("name").to_pylist() == sorted(names)
        assert sum(table.column("full_train").to_pylist()) == 8
        assert sum(table.column("full_test").to_pylist()) == 2
        assert sorted(set(table.column("naca_digits").to_pylist())) == [4, 5]
        assert table.column("n_surface").to_pylist() == [len(SURFACE_NODES)] * 10

        etl = json.loads((_interim(roots) / "etl.json").read_text(encoding="utf-8"))
        assert etl["status"] == "complete"
        assert etl["simulations"] == {"expected": 10, "processed": 10, "skipped": 0, "failed": 0}

        splits = json.loads((_processed(roots) / "splits.json").read_text(encoding="utf-8"))
        assert splits["sizes"] == {"train": 7, "val": 1, "test": 2}
        norm = json.loads((_processed(roots) / "normalisation.json").read_text(encoding="utf-8"))
        assert norm["channels"]["x"]["count"] == 7 * 16  # every node of the 7 training simulations
        assert norm["fitted_on"]["simulations"] == 7

    def test_node_tables_equal_the_raw_files(self, roots: dict[str, Path]) -> None:
        sims = _snapshot(roots)
        _run(roots)
        dataset = roots["raw_root"] / SNAPSHOT / "Dataset"
        for sim in sims:
            raw = read_simulation(dataset / sim.name)
            nodes = read_table(_interim(roots) / "simulations" / sim.name / NODES_FILE, NODES_SCHEMA)
            np.testing.assert_array_equal(nodes.column("p").to_numpy(), raw.mesh.point_data["p"])
            np.testing.assert_array_equal(nodes.column("sdf").to_numpy(), -raw.mesh.point_data["implicit_distance"])

    def test_two_worker_processes_give_the_same_tables(self, roots: dict[str, Path], tmp_path: Path) -> None:
        _snapshot(roots)
        _run(roots, workers=2)
        other = {**roots, "interim_root": tmp_path / "interim1", "processed_root": tmp_path / "processed1"}
        _run(other, workers=1)
        a = pq.read_table(_interim(roots) / "simulations.parquet").drop(["read_s", "total_s"])
        b = pq.read_table(_interim(other) / "simulations.parquet").drop(["read_s", "total_s"])
        assert a.equals(b)
        assert (_processed(roots) / "normalisation.json").read_bytes() == (_processed(other) / "normalisation.json").read_bytes()


class TestRerun:
    def test_a_second_run_writes_nothing(self, roots: dict[str, Path], caplog: pytest.LogCaptureFixture) -> None:
        _snapshot(roots)
        _run(roots)
        before = _mtimes(_interim(roots), _processed(roots))

        result = _run(roots)

        assert (result.processed, result.skipped, result.wrote_dataset) == (0, 10, False)
        assert _mtimes(_interim(roots), _processed(roots)) == before
        assert "etl_up_to_date" in caplog.text

    def test_only_the_missing_simulation_is_redone(self, roots: dict[str, Path]) -> None:
        sims = _snapshot(roots)
        _run(roots)
        sims_root = _interim(roots) / "simulations"
        shutil.rmtree(sims_root / sims[3].name)
        (sims_root / f"{sims[4].name}.partial").mkdir()  # an interrupted write

        result = _run(roots)

        assert (result.processed, result.skipped) == (1, 10 - 1)
        assert [o.name for o in result.outcomes] == [sims[3].name]
        assert not (sims_root / f"{sims[4].name}.partial").exists()
        assert result.wrote_dataset

    def test_splits_are_byte_identical_when_rebuilt(self, roots: dict[str, Path]) -> None:
        _snapshot(roots)
        _run(roots)
        first = (_processed(roots) / "splits.json").read_bytes()
        (_processed(roots) / "splits.json").unlink()
        _run(roots)
        assert (_processed(roots) / "splits.json").read_bytes() == first


class TestFaultTolerance:
    def test_bad_simulations_are_skipped_and_nothing_dataset_level_is_written(self, roots: dict[str, Path]) -> None:
        sims = dataset_of(3)
        sims[1].nut[0] = -1e-6  # fails G10
        _snapshot(roots, sims, train=[s.name for s in sims[:2]], test=[sims[2].name])
        mesh = roots["raw_root"] / SNAPSHOT / "Dataset" / sims[2].name / f"{sims[2].name}_internal.vtu"
        mesh.write_bytes(mesh.read_bytes()[:200])  # truncated

        result = _run(roots)

        assert result.processed == 1
        assert {(o.name, o.violations[0][0]) for o in result.failures} == {(sims[1].name, "G10"), (sims[2].name, "read")}
        assert (_interim(roots) / "simulations" / sims[0].name / "meta.json").is_file()
        assert not (_interim(roots) / "simulations.parquet").exists()
        assert not _processed(roots).exists()
        etl = json.loads((_interim(roots) / "etl.json").read_text(encoding="utf-8"))
        assert etl["status"] == "incomplete"
        assert {f["check"] for f in etl["failures"]} == {"G10", "read"}

        # Repair the two files; only they are processed, and the set completes.
        dataset = roots["raw_root"] / SNAPSHOT / "Dataset"
        sims[1].nut[0] = 1e-4
        write_sim(sims[1], dataset)
        write_sim(sims[2], dataset)
        result = _run(roots)
        assert (result.processed, result.skipped, result.failures, result.wrote_dataset) == (2, 1, (), True)

    def test_a_manifest_name_without_files_is_a_failure(self, roots: dict[str, Path]) -> None:
        sims = dataset_of(2)
        _snapshot(roots, sims, train=[sims[0].name, "airFoil2D_SST_50.0_1.0_2.0_4.0_12.0"], test=[sims[1].name])
        result = _run(roots)
        assert [(o.name, o.violations[0][0]) for o in result.failures] == [("airFoil2D_SST_50.0_1.0_2.0_4.0_12.0", "read")]

    def test_a_split_naming_an_unknown_simulation_stops_the_run(self, roots: dict[str, Path]) -> None:
        _snapshot(roots, extra_splits={"scarce_train": ["airFoil2D_SST_99.0_1.0_2.0_4.0_12.0"]})
        with pytest.raises(EtlError, match="scarce_train"):
            _run(roots)


class TestStops:
    def test_an_interim_directory_from_another_schema_is_not_mixed_into(self, roots: dict[str, Path]) -> None:
        _snapshot(roots)
        _run(roots)
        path = _interim(roots) / "etl.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**record, "schema_version": "airfrans-interim/0"}), encoding="utf-8")
        with pytest.raises(EtlError, match="Delete that directory"):
            _run(roots)

    def test_a_different_raw_snapshot_is_not_mixed_into(self, roots: dict[str, Path]) -> None:
        _snapshot(roots)
        _run(roots)
        source = roots["raw_root"] / SNAPSHOT / "source.json"
        source.write_text(json.dumps({"sha256": "cd" * 32}), encoding="utf-8")
        with pytest.raises(EtlError, match="raw SHA-256"):
            _run(roots)

    def test_no_complete_snapshot_stops(self, roots: dict[str, Path]) -> None:
        roots["raw_root"].mkdir(parents=True)
        with pytest.raises(EtlError, match="found none"):
            _run(roots)

    def test_a_smoke_run_writes_per_simulation_outputs_only(self, roots: dict[str, Path]) -> None:
        _snapshot(roots)
        result = _run(roots, limit=3)
        assert result.processed == 3 and not result.wrote_dataset
        assert not (_interim(roots) / "simulations.parquet").exists()
        assert json.loads((_interim(roots) / "etl.json").read_text(encoding="utf-8"))["status"] == "running"


class TestLeakage:
    def _normalisation(self, roots: dict[str, Path], sims: list) -> dict:  # noqa: ANN001
        names = [s.name for s in sims]
        _snapshot(roots, sims, train=names[:8], test=names[8:])
        _run(roots)
        return json.loads((_processed(roots) / "normalisation.json").read_text(encoding="utf-8"))

    @staticmethod
    def _perturb(sim) -> None:  # noqa: ANN001
        # Change velocity, pressure and turbulence away from the surface, in a way the gate still accepts.
        off = np.ones(len(sim.p), dtype=bool)
        off[list(SURFACE_NODES)] = False
        sim.U[off] *= 3
        sim.p[off] = np.float32(-5.0)
        sim.nut[off] *= 100

    def test_changing_val_and_test_leaves_the_statistics_unchanged(self, tmp_path: Path) -> None:
        def roots_at(label: str) -> dict[str, Path]:
            base = tmp_path / label
            return {"raw_root": base / "raw", "interim_root": base / "interim", "processed_root": base / "processed"}

        clean = dataset_of(N_SIMS)
        baseline = self._normalisation(roots_at("a"), clean)

        names = [s.name for s in clean]
        splits = make_splits({"full_train": names[:8], "full_test": names[8:]})
        held_out = set(splits.val) | set(splits.test)
        changed = dataset_of(N_SIMS)
        for sim in changed:
            if sim.name in held_out:
                self._perturb(sim)
        assert self._normalisation(roots_at("b"), changed) == baseline

        # The same change to one training simulation does move them: the test has teeth.
        moved = dataset_of(N_SIMS)
        self._perturb(next(s for s in moved if s.name in splits.train))
        assert self._normalisation(roots_at("c"), moved)["channels"]["p"] != baseline["channels"]["p"]


class TestExitCode:
    def test_failures_and_stops_exit_non_zero(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def result(failures: tuple) -> EtlResult:
            return EtlResult(tmp_path, tmp_path, 1, 1 - len(failures), 0, failures, False, 0.0)

        monkeypatch.setattr(pipeline, "run", lambda **_: result(()))
        assert pipeline.main([]) == 0
        monkeypatch.setattr(pipeline, "run", lambda **_: result((pipeline.SimOutcome("x", False, (("G1", "d"),)),)))
        assert pipeline.main([]) == 1

        def stop(**_: object) -> EtlResult:
            raise EtlError("no snapshot")

        monkeypatch.setattr(pipeline, "run", stop)
        assert pipeline.main([]) == 1
