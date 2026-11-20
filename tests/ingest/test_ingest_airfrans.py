"""The ingestion pipeline end to end, against a local server: idempotency, resume, stops."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conftest import FileServer
from src.ingest.airfrans import IngestResult, git_commit, run
from src.ingest.download import TruncatedTransfer
from src.ingest.retry import PermanentIngestError
from synthetic_archives import ADVERSARIAL, SIMULATIONS, build_dataset_zip

FIRST_DAY = datetime(2026, 11, 20, 21, 30, tzinfo=timezone.utc)
NEXT_DAY = datetime(2026, 11, 21, 8, 0, tzinfo=timezone.utc)
SOURCE_KEYS = {
    "source",
    "url",
    "etag",
    "last_modified",
    "size_bytes",
    "sha256",
    "license",
    "citation",
    "downloaded_at_utc",
    "manifest_splits",
    "ingest_code_commit",
}


def _no_sleep(_: float) -> None:
    pass


def _run(server: FileServer, raw_root: Path, *, now: datetime = FIRST_DAY, **kwargs: object) -> IngestResult:
    return run(
        url=server.url,
        raw_root=raw_root,
        sleep=_no_sleep,
        chunk_bytes=16 * 1024,
        now=lambda: now,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _info_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)


@pytest.fixture
def served(file_server: FileServer, tmp_path: Path) -> FileServer:
    file_server.payload = build_dataset_zip(tmp_path / "served.zip").read_bytes()
    return file_server


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    return tmp_path / "raw" / "airfrans"


def _snapshot_dirs(raw_root: Path) -> list[str]:
    return sorted(p.name for p in raw_root.iterdir())


class TestFirstRun:
    def test_builds_a_complete_recorded_snapshot(self, served: FileServer, raw_root: Path) -> None:
        result = _run(served, raw_root)
        snap = raw_root / "2026-11-20"

        assert result.action == "ingested"
        assert result.snapshot_dir == snap
        assert result.sha256 == hashlib.sha256(served.payload).hexdigest()
        assert (snap / "Dataset.zip").read_bytes() == served.payload

        source = json.loads((snap / "source.json").read_text(encoding="utf-8"))
        assert set(source) == SOURCE_KEYS
        assert source["sha256"] == result.sha256
        assert source["etag"] == '"v1"'
        assert source["size_bytes"] == len(served.payload)
        assert source["license"] == "ODbL-1.0"
        assert source["manifest_splits"] == {"full_train": 1, "full_test": 1}
        assert source["downloaded_at_utc"] == "2026-11-20T21:30:00+00:00"

        listing = json.loads((snap / "listing.json").read_text(encoding="utf-8"))
        assert len(listing) == 2 + 4 * len(SIMULATIONS)
        for record in listing:
            if not record["is_dir"]:
                extracted = snap / record["name"]  # "Dataset/<sim>/<file>"
                assert extracted.stat().st_size == record["size"]

        # Nothing transient is left behind.
        for leftover in ("download.json", "Dataset.zip.part", "Dataset.partial"):
            assert not (snap / leftover).exists(), leftover


class TestIdempotency:
    def test_a_second_run_makes_one_head_request_and_nothing_else(
        self, served: FileServer, raw_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = _run(served, raw_root)
        served.requests.clear()
        before = {p: p.stat().st_mtime_ns for p in raw_root.rglob("*")}

        second = _run(served, raw_root, now=NEXT_DAY)

        assert second.action == "snapshot_exists"
        assert second.sha256 == first.sha256
        assert served.requests == [("HEAD", None)]
        assert _snapshot_dirs(raw_root) == ["2026-11-20"]
        assert {p: p.stat().st_mtime_ns for p in raw_root.rglob("*")} == before
        assert "snapshot_exists" in caplog.text

    def test_a_missing_file_is_restored_from_the_zip_without_downloading(
        self, served: FileServer, raw_root: Path
    ) -> None:
        _run(served, raw_root)
        sim = SIMULATIONS[1]
        victim = raw_root / "2026-11-20" / "Dataset" / sim / f"{sim}_freestream.vtp"
        victim.unlink()
        served.requests.clear()

        result = _run(served, raw_root)

        assert result.action == "extracted"
        assert victim.exists()
        assert served.gets() == []

    def test_an_interrupted_extraction_is_redone(self, served: FileServer, raw_root: Path) -> None:
        _run(served, raw_root)
        snap = raw_root / "2026-11-20"
        shutil.rmtree(snap / "Dataset")
        (snap / "Dataset.partial").mkdir()
        (snap / "Dataset.partial" / "half_written.vtu").write_bytes(b"...")
        served.requests.clear()

        result = _run(served, raw_root)

        assert result.action == "extracted"
        assert not (snap / "Dataset.partial").exists()
        assert (snap / "Dataset" / SIMULATIONS[0]).is_dir()
        assert served.gets() == []

    def test_verify_rehashes_and_catches_a_corrupted_archive(self, served: FileServer, raw_root: Path) -> None:
        _run(served, raw_root)
        zip_path = raw_root / "2026-11-20" / "Dataset.zip"
        raw = bytearray(zip_path.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        zip_path.write_bytes(bytes(raw))

        # The default run checks size and tree only, so it doesn't notice.
        assert _run(served, raw_root).action == "snapshot_exists"
        with pytest.raises(PermanentIngestError, match="SHA-256"):
            _run(served, raw_root, verify=True)


class TestResume:
    def test_an_interrupted_download_resumes_in_its_own_directory_the_next_day(
        self, served: FileServer, raw_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cut = len(served.payload) // 2
        served.drop_after = cut
        with pytest.raises(TruncatedTransfer):
            _run(served, raw_root, download_attempts=1)

        snap = raw_root / "2026-11-20"
        assert (snap / "download.json").is_file()
        assert (snap / "Dataset.zip.part").stat().st_size == cut
        assert not (snap / "source.json").exists()

        result = _run(served, raw_root, now=NEXT_DAY)

        assert result.snapshot_dir == snap
        assert result.sha256 == hashlib.sha256(served.payload).hexdigest()
        assert f"download_resume offset={cut}" in caplog.text
        assert "sha256_rehash reason=resumed_across_runs" in caplog.text
        assert _snapshot_dirs(raw_root) == ["2026-11-20"]


class TestStops:
    def test_a_changed_remote_never_becomes_a_second_snapshot(self, served: FileServer, raw_root: Path) -> None:
        _run(served, raw_root)
        served.etag = '"v2"'
        with pytest.raises(PermanentIngestError, match="differs from snapshot 2026-11-20"):
            _run(served, raw_root, now=NEXT_DAY)
        assert _snapshot_dirs(raw_root) == ["2026-11-20"]

    def test_a_changed_remote_stops_an_unfinished_download(self, served: FileServer, raw_root: Path) -> None:
        served.drop_after = len(served.payload) // 2
        with pytest.raises(TruncatedTransfer):
            _run(served, raw_root, download_attempts=1)
        served.etag = '"v2"'
        with pytest.raises(PermanentIngestError, match="changed since the unfinished download"):
            _run(served, raw_root)

    @pytest.mark.parametrize("case", sorted(ADVERSARIAL))
    def test_a_rejected_archive_leaves_nothing_extracted(
        self, case: str, file_server: FileServer, raw_root: Path, tmp_path: Path
    ) -> None:
        build, fragment = ADVERSARIAL[case]
        file_server.payload = build(tmp_path / "bad.zip").read_bytes()

        with pytest.raises(PermanentIngestError, match=re.escape(fragment)):
            _run(file_server, raw_root)

        snap = raw_root / "2026-11-20"
        assert sorted(p.name for p in snap.iterdir()) == ["Dataset.zip", "download.json"]


class TestGitCommit:
    SHA = "0123456789abcdef0123456789abcdef01234567"

    def test_reads_a_loose_branch_ref(self, tmp_path: Path) -> None:
        (tmp_path / ".git" / "refs" / "heads").mkdir(parents=True)
        (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (tmp_path / ".git" / "refs" / "heads" / "main").write_text(self.SHA + "\n")
        assert git_commit(tmp_path) == self.SHA

    def test_reads_a_packed_ref(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (tmp_path / ".git" / "packed-refs").write_text(f"# pack-refs with: peeled\n{self.SHA} refs/heads/main\n")
        assert git_commit(tmp_path) == self.SHA

    def test_reads_a_detached_head(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text(self.SHA + "\n")
        assert git_commit(tmp_path) == self.SHA

    def test_no_repository_gives_none(self, tmp_path: Path) -> None:
        assert git_commit(tmp_path) is None
