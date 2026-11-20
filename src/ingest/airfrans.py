"""AirfRANS raw snapshot: download, verify, gate and extract the dataset archive once.

    python -m src.ingest.airfrans            # ingest, or confirm the snapshot is intact
    python -m src.ingest.airfrans --verify   # also re-hash the snapshot's Dataset.zip

In the container:

    docker compose --profile ml run --rm ml python -m src.ingest.airfrans

Layout of one snapshot, immutable once complete:

    data/raw/airfrans/<snapshot-date>/
        Dataset.zip     the archive, byte for byte as served
        source.json     provenance: URL, ETag, Last-Modified, size, SHA-256, license
        listing.json    every archive entry: name, sizes, CRC-32
        Dataset/        the extracted tree, written only after the archive gate passes

The run is idempotent. When a snapshot already matches the remote file and its
tree is complete, the only network call is one HEAD. An unfinished download is
resumed in the directory it started in. If the remote file differs from a
snapshot already on disk, the run stops rather than silently ingesting a
different dataset under the same name.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import AIRFRANS_RAW_ROOT, PROJECT_ROOT
from src.ingest.archive import ArchiveReport, ZipEntry, check_tree, inspect_zip, safe_extract
from src.ingest.download import (
    DOWNLOAD_ATTEMPTS,
    USER_AGENT,
    RemoteFile,
    download,
    head,
    part_path,
    sha256_file,
)
from src.ingest.retry import PermanentIngestError

logger = logging.getLogger(__name__)

AIRFRANS_URL = "https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip"
SOURCE_NAME = "AirfRANS"
LICENSE = "ODbL-1.0"
CITATION = "Bonnet et al., AirfRANS, NeurIPS 2022 Datasets and Benchmarks Track, arXiv:2212.07564"

#: The release this pipeline was written and checked against (measured 2026-11-20).
RELEASE_ETAG = '"62f519b5-255c76d39"'
RELEASE_SIZE_BYTES = 10_029_067_577
RELEASE_SUFFIX_COUNTS = {".json": 1, ".vtp": 2000, ".vtu": 1000}
RELEASE_SPLIT_SIZES = {"full_train": 800, "full_test": 200}

ZIP_NAME = "Dataset.zip"
DATASET_DIR = "Dataset"
SOURCE_FILE = "source.json"
LISTING_FILE = "listing.json"
#: Written when a download starts and removed once source.json exists. It ties
#: an unfinished `.part` to the ETag it was started against.
PENDING_FILE = "download.json"
#: Free space required on top of the bytes about to be written.
FREE_SPACE_MARGIN = 1.05


@dataclass(frozen=True)
class IngestResult:
    snapshot_dir: Path
    #: "ingested" (downloaded and extracted), "extracted" (tree rebuilt from the
    #: snapshot's zip) or "snapshot_exists" (nothing to do).
    action: str
    sha256: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON through a temporary file and a rename, so a reader never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(root: Path = PROJECT_ROOT) -> str | None:
    """The checked-out commit, read from `.git` directly: the container has no git binary."""
    git_dir = root / ".git"
    try:
        head_ref = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head_ref.startswith("ref: "):
            return head_ref or None
        ref = head_ref.removeprefix("ref: ")
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip() or None
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except OSError:
        return None
    return None


def _require_free_space(path: Path, need_bytes: int, what: str) -> None:
    free = shutil.disk_usage(path).free
    logger.info("disk_free path=%s free_bytes=%d need_bytes=%d for=%s", path, free, need_bytes, what)
    if free < need_bytes * FREE_SPACE_MARGIN:
        raise PermanentIngestError(
            f"{what} needs {need_bytes / 1e9:.1f} GB plus margin, {free / 1e9:.1f} GB free at {path}"
        )


def _changed_source_message(snapshot_dir: Path, recorded: dict[str, Any], remote: RemoteFile) -> str:
    return (
        f"the remote file differs from snapshot {snapshot_dir.name}: ETag "
        f"{recorded.get('etag')} and {recorded.get('size_bytes')} bytes on disk, "
        f"{remote.etag} and {remote.size_bytes} bytes at {remote.url}. Adopting a new "
        "release is a deliberate decision: review what changed, then move the old "
        "snapshot out of the raw directory to ingest the new one."
    )


def _log_report(report: ArchiveReport) -> None:
    files = report.files
    logger.info(
        "archive_gate_passed entries=%d files=%d dirs=%d suffixes=%s unsafe=0 "
        "uncompressed_bytes=%d ratio_overall=%.2f simulations=%d splits=%s train_test_overlap=0",
        len(report.entries),
        len(files),
        len(report.entries) - len(files),
        report.suffix_counts,
        report.total_uncompressed,
        report.total_uncompressed / report.total_compressed if report.total_compressed else 0.0,
        len(report.simulations),
        report.split_sizes,
    )
    if report.unlisted_dirs:
        logger.warning("archive_unlisted_dirs count=%d first=%r", len(report.unlisted_dirs), report.unlisted_dirs[0])

    # Asserted against the known release and logged, but not part of the gate:
    # a correct re-release must not be rejected for being different.
    counts_match = report.suffix_counts == RELEASE_SUFFIX_COUNTS
    splits_match = all(report.split_sizes.get(k) == v for k, v in RELEASE_SPLIT_SIZES.items())
    level = logging.INFO if counts_match and splits_match else logging.WARNING
    logger.log(
        level,
        "archive_matches_known_release counts=%s splits=%s",
        counts_match,
        splits_match,
    )


def _source_record(
    remote: RemoteFile, sha256: str, downloaded_at: datetime, report: ArchiveReport
) -> dict[str, Any]:
    return {
        "source": SOURCE_NAME,
        "url": remote.url,
        "etag": remote.etag,
        "last_modified": remote.last_modified,
        "size_bytes": remote.size_bytes,
        "sha256": sha256,
        "license": LICENSE,
        "citation": CITATION,
        "downloaded_at_utc": downloaded_at.isoformat(timespec="seconds"),
        "manifest_splits": report.split_sizes,
        "ingest_code_commit": git_commit(),
    }


def _scan(raw_root: Path) -> tuple[list[tuple[Path, dict[str, Any]]], list[Path]]:
    """Completed snapshots (with source.json) and unfinished ones (download.json only)."""
    complete: list[tuple[Path, dict[str, Any]]] = []
    pending: list[Path] = []
    if not raw_root.exists():
        return complete, pending
    for snapshot_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        if (snapshot_dir / SOURCE_FILE).is_file():
            complete.append((snapshot_dir, _read_json(snapshot_dir / SOURCE_FILE)))
        elif (snapshot_dir / PENDING_FILE).is_file():
            pending.append(snapshot_dir)
    return complete, pending


def _extract(snapshot_dir: Path, zip_path: Path, report: ArchiveReport) -> None:
    _require_free_space(snapshot_dir, report.total_uncompressed, "extraction")
    safe_extract(zip_path, report, snapshot_dir / DATASET_DIR)
    tree = check_tree(snapshot_dir / DATASET_DIR, report.entries)
    if not tree.complete:
        raise PermanentIngestError(f"extracted tree incomplete: {tree}")
    logger.info("tree_complete files=%d missing=0 wrong_size=0", tree.expected_files)


def _confirm_snapshot(
    snapshot_dir: Path, source: dict[str, Any], *, verify: bool, started: float
) -> IngestResult:
    """A completed snapshot matches the remote: check it is intact, rebuild the tree if it isn't."""
    zip_path = snapshot_dir / ZIP_NAME
    if not zip_path.is_file():
        raise PermanentIngestError(
            f"{snapshot_dir.name} has {SOURCE_FILE} but no {ZIP_NAME}. Raw snapshots are "
            "immutable: restore the archive, or move the directory aside to download again."
        )
    size = zip_path.stat().st_size
    if size != source["size_bytes"]:
        raise PermanentIngestError(f"{zip_path}: {size:,} bytes, {SOURCE_FILE} records {source['size_bytes']:,}")
    if verify:
        digest = sha256_file(zip_path)
        if digest != source["sha256"]:
            raise PermanentIngestError(f"{zip_path}: SHA-256 {digest}, {SOURCE_FILE} records {source['sha256']}")
        logger.info("snapshot_verified sha256=%s", digest)

    # Left behind only if a run stopped between recording the snapshot and tidying up.
    (snapshot_dir / PENDING_FILE).unlink(missing_ok=True)

    entries = tuple(ZipEntry.from_dict(r) for r in _read_json(snapshot_dir / LISTING_FILE))
    tree = check_tree(snapshot_dir / DATASET_DIR, entries)
    if tree.complete:
        logger.info(
            "snapshot_exists dir=%s files=%d seconds=%.1f",
            snapshot_dir,
            tree.expected_files,
            time.monotonic() - started,
        )
        return IngestResult(snapshot_dir, "snapshot_exists", source["sha256"])

    logger.warning("snapshot_tree_incomplete missing=%d wrong_size=%d", tree.missing, tree.wrong_size)
    report = inspect_zip(zip_path)
    _extract(snapshot_dir, zip_path, report)
    return IngestResult(snapshot_dir, "extracted", source["sha256"])


def _pending_dir(raw_root: Path, pending: list[Path], remote: RemoteFile, now: datetime) -> Path:
    """The directory to download into: an unfinished one for this file, or a new one."""
    if len(pending) > 1:
        raise PermanentIngestError(f"more than one unfinished snapshot: {[p.name for p in pending]}")
    if pending:
        snapshot_dir = pending[0]
        started_for = _read_json(snapshot_dir / PENDING_FILE)
        if started_for.get("etag") != remote.etag or started_for.get("size_bytes") != remote.size_bytes:
            raise PermanentIngestError(
                f"the remote file changed since the unfinished download in {snapshot_dir.name} "
                f"started (ETag {started_for.get('etag')} then, {remote.etag} now). Delete that "
                "directory to start over from the new file."
            )
        logger.info("snapshot_unfinished dir=%s", snapshot_dir)
        return snapshot_dir

    # Named after the UTC date of the first byte, and kept on resume.
    snapshot_dir = raw_root / now.date().isoformat()
    if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
        raise PermanentIngestError(f"{snapshot_dir} exists with unrecognised content; not writing into it")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        snapshot_dir / PENDING_FILE,
        {
            "url": remote.url,
            "etag": remote.etag,
            "size_bytes": remote.size_bytes,
            "last_modified": remote.last_modified,
            "started_at_utc": now.isoformat(timespec="seconds"),
        },
    )
    logger.info("snapshot_new dir=%s", snapshot_dir)
    return snapshot_dir


def run(
    *,
    url: str = AIRFRANS_URL,
    raw_root: Path = AIRFRANS_RAW_ROOT,
    verify: bool = False,
    user_agent: str = USER_AGENT,
    download_attempts: int = DOWNLOAD_ATTEMPTS,
    chunk_bytes: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _utc_now,
) -> IngestResult:
    """Make sure one complete, verified snapshot of the remote archive exists under `raw_root`."""
    started = time.monotonic()
    remote = head(url, user_agent=user_agent, sleep=sleep)
    logger.info(
        "remote url=%s size_bytes=%d etag=%s last_modified=%r known_release=%s",
        remote.url,
        remote.size_bytes,
        remote.etag,
        remote.last_modified,
        remote.etag == RELEASE_ETAG and remote.size_bytes == RELEASE_SIZE_BYTES,
    )

    complete, pending = _scan(raw_root)
    matching = [
        (d, s) for d, s in complete if s.get("etag") == remote.etag and s.get("size_bytes") == remote.size_bytes
    ]
    if matching:
        snapshot_dir, source = matching[-1]
        return _confirm_snapshot(snapshot_dir, source, verify=verify, started=started)
    if complete:
        snapshot_dir, source = complete[-1]
        raise PermanentIngestError(_changed_source_message(snapshot_dir, source, remote))

    raw_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir = _pending_dir(raw_root, pending, remote, now())
    zip_path = snapshot_dir / ZIP_NAME

    if zip_path.is_file():
        # The download finished but the run stopped before recording it.
        size = zip_path.stat().st_size
        if size != remote.size_bytes:
            raise PermanentIngestError(f"{zip_path}: {size:,} bytes, {remote.size_bytes:,} expected")
        logger.info("download_found_unrecorded path=%s action=rehash", zip_path)
        digest = sha256_file(zip_path)
        downloaded_at = datetime.fromtimestamp(zip_path.stat().st_mtime, timezone.utc)
    else:
        part = part_path(zip_path)
        already = part.stat().st_size if part.exists() else 0
        _require_free_space(snapshot_dir, remote.size_bytes - already, "download")
        extra = {"chunk_bytes": chunk_bytes} if chunk_bytes else {}
        result = download(
            remote,
            zip_path,
            user_agent=user_agent,
            attempts=download_attempts,
            sleep=sleep,
            **extra,
        )
        digest = result.sha256
        downloaded_at = now()

    report = inspect_zip(zip_path)
    _log_report(report)
    _write_json(snapshot_dir / LISTING_FILE, report.listing())
    _write_json(snapshot_dir / SOURCE_FILE, _source_record(remote, digest, downloaded_at, report))
    (snapshot_dir / PENDING_FILE).unlink()
    logger.info("snapshot_recorded dir=%s sha256=%s", snapshot_dir, digest)

    _extract(snapshot_dir, zip_path, report)
    logger.info("ingest_done dir=%s seconds=%.1f", snapshot_dir, time.monotonic() - started)
    return IngestResult(snapshot_dir, "ingested", digest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-hash an existing snapshot's Dataset.zip against source.json (about a minute)",
    )
    args = parser.parse_args(argv)

    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    try:
        result = run(verify=args.verify)
    except PermanentIngestError as exc:
        logger.error("ingest_failed error=%s", exc)
        return 1
    except (OSError, http.client.HTTPException) as exc:
        # Transient failures that outlasted every retry. Nothing is lost: the
        # bytes already downloaded stay in Dataset.zip.part.
        logger.error("ingest_interrupted error=%r: rerun to resume from the bytes on disk", exc)
        return 1
    logger.info("ingest_result action=%s dir=%s sha256=%s", result.action, result.snapshot_dir, result.sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
