"""Resumable download against a local server that drops, ignores Range, or changes the file."""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path

import pytest

from conftest import FileServer
from src.ingest.download import DownloadResult, TruncatedTransfer, download, head, part_path, sha256_file
from src.ingest.retry import PermanentIngestError

PAYLOAD = random.Random(7).randbytes(3 * 1024 * 1024 + 123)
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
CHUNK = 256 * 1024
CUT = 1_000_003  # deliberately not on a chunk boundary


def _no_sleep(_: float) -> None:
    pass


@pytest.fixture
def server(file_server: FileServer) -> FileServer:
    file_server.payload = PAYLOAD
    return file_server


@pytest.fixture(autouse=True)
def _info_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)


def _download(server: FileServer, dest: Path, **kwargs: object):
    remote = head(server.url, sleep=_no_sleep)
    return download(remote, dest, chunk_bytes=CHUNK, sleep=_no_sleep, **kwargs)


class TestHead:
    def test_reports_size_etag_and_last_modified(self, server: FileServer) -> None:
        remote = head(server.url, sleep=_no_sleep)
        assert remote.size_bytes == len(PAYLOAD)
        assert remote.etag == '"v1"'
        assert remote.last_modified == "Thu, 11 Aug 2022 15:01:09 GMT"
        assert server.requests == [("HEAD", None)]

    def test_a_server_without_etag_is_refused(self, server: FileServer) -> None:
        server.send_etag = False
        with pytest.raises(PermanentIngestError, match="no ETag"):
            head(server.url, sleep=_no_sleep)


class TestDownload:
    def test_clean_download_matches_the_sha256(
        self, server: FileServer, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        dest = tmp_path / "Dataset.zip"
        result = _download(server, dest)

        assert result.sha256 == SHA256
        assert dest.read_bytes() == PAYLOAD
        assert not part_path(dest).exists()
        assert result.resumed_from == ()
        assert result.transferred_bytes == len(PAYLOAD)
        assert server.gets() == [None]
        assert "download_progress pct=100" in caplog.text

    def test_a_dropped_connection_resumes_from_the_offset(
        self, server: FileServer, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        server.drop_after = CUT
        dest = tmp_path / "Dataset.zip"
        result = _download(server, dest)

        assert result.sha256 == SHA256
        assert result.resumed_from == (CUT,)
        assert server.gets() == [None, f"bytes={CUT}-"]
        assert f"download_resume offset={CUT}" in caplog.text
        # Same process: the streamed hash already covers every byte.
        assert "sha256_rehash" not in caplog.text

    def test_a_server_ignoring_range_forces_a_clean_restart(
        self, server: FileServer, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        server.drop_after = CUT
        server.ignore_range = True
        dest = tmp_path / "Dataset.zip"
        result = _download(server, dest)

        assert result.sha256 == SHA256
        assert result.restarts == 1
        # The first CUT bytes were fetched twice; throughput counts both.
        assert result.transferred_bytes == CUT + len(PAYLOAD)
        assert dest.stat().st_size == len(PAYLOAD)
        assert "download_restart_no_range" in caplog.text

    def test_an_etag_change_mid_download_stops_it(self, server: FileServer, tmp_path: Path) -> None:
        server.drop_after = CUT
        server.etag_after_get = '"v2"'
        dest = tmp_path / "Dataset.zip"
        with pytest.raises(PermanentIngestError, match="source changed during download"):
            _download(server, dest)
        assert not dest.exists()

    def test_a_range_answered_at_the_wrong_offset_stops_it(self, server: FileServer, tmp_path: Path) -> None:
        server.drop_after = CUT
        server.range_skew = 1
        dest = tmp_path / "Dataset.zip"
        with pytest.raises(PermanentIngestError, match="Content-Range"):
            _download(server, dest)
        assert not dest.exists()
        assert part_path(dest).stat().st_size == CUT  # nothing misplaced was appended

    def test_a_part_left_by_an_earlier_run_is_resumed_and_rehashed(
        self, server: FileServer, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        dest = tmp_path / "Dataset.zip"
        part_path(dest).write_bytes(PAYLOAD[:CUT])
        result = _download(server, dest)

        assert result.sha256 == SHA256
        assert server.gets() == [f"bytes={CUT}-"]
        assert "sha256_rehash reason=resumed_across_runs" in caplog.text
        # Only the remainder came over the network in this run.
        assert result.size_bytes == len(PAYLOAD)
        assert result.transferred_bytes == len(PAYLOAD) - CUT
        assert result.transfer_seconds <= result.seconds

    def test_an_oversized_part_is_discarded(self, server: FileServer, tmp_path: Path) -> None:
        dest = tmp_path / "Dataset.zip"
        part_path(dest).write_bytes(PAYLOAD + b"trailing garbage")
        result = _download(server, dest)

        assert result.sha256 == SHA256
        assert server.gets() == [None]

    def test_exhausted_attempts_keep_the_part_for_the_next_run(self, server: FileServer, tmp_path: Path) -> None:
        server.drop_after = CUT
        dest = tmp_path / "Dataset.zip"
        with pytest.raises(TruncatedTransfer):
            _download(server, dest, attempts=1)
        assert not dest.exists()
        assert part_path(dest).stat().st_size == CUT


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    path.write_bytes(PAYLOAD)
    assert sha256_file(path, chunk_bytes=CHUNK) == SHA256


class TestThroughput:
    MIB = 1024 * 1024

    def _result(self, **overrides: object) -> DownloadResult:
        values: dict[str, object] = {
            "path": Path("Dataset.zip"),
            "size_bytes": 10 * self.MIB,
            "sha256": "0" * 64,
            "seconds": 20.0,
            "transferred_bytes": 4 * self.MIB,
            "transfer_seconds": 2.0,
            "resumed_from": (6 * self.MIB,),
            "restarts": 0,
        }
        values.update(overrides)
        return DownloadResult(**values)  # type: ignore[arg-type]

    def test_counts_only_what_this_run_fetched_over_the_fetch_time(self) -> None:
        # Not 10 MiB / 20 s: 6 MiB came from an earlier run, 18 s was the re-hash.
        assert self._result().mib_per_s == pytest.approx(2.0)

    def test_no_transfer_time_gives_zero(self) -> None:
        assert self._result(transferred_bytes=0, transfer_seconds=0.0).mib_per_s == 0.0
