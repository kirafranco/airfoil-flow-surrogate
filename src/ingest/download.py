"""Resumable HTTP download of one large file, pinned to the version the server had at the start.

The file is streamed into `<name>.part`. When the connection drops, the next
attempt asks only for the missing bytes (`Range: bytes=<offset>-`) and appends
them. The download refuses to continue if the server's ETag changes, because
splicing two versions of a file together would produce something neither of
them is. When every byte has arrived, the size is checked against the
advertised length and `.part` is renamed into place in a single atomic step,
so a file without the `.part` suffix is always complete.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.ingest.retry import PermanentIngestError, retry_with_backoff

logger = logging.getLogger(__name__)

#: Descriptive, and carries no personal or account details.
USER_AGENT = "airfrans-ingest/1.0 (python-urllib)"

CHUNK_BYTES = 8 * 1024 * 1024
TIMEOUT_S = 60.0
#: Failed attempts tolerated in a row. Every attempt resumes where the last one
#: stopped, so even an unreliable connection eventually finishes the file.
DOWNLOAD_ATTEMPTS = 8
PROGRESS_STEP_PCT = 5

_MIB = 1024 * 1024
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class TruncatedTransfer(ConnectionError):
    """The server closed the connection before sending the whole body.

    A ConnectionError, so the retry logic treats it as transient: the next
    attempt resumes from the bytes already on disk.
    """


@dataclass(frozen=True)
class RemoteFile:
    """What the server says about the file before any of it is downloaded."""

    url: str
    size_bytes: int
    etag: str
    last_modified: str | None


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size_bytes: int
    sha256: str
    #: Whole call, including a re-hash from disk if one was needed.
    seconds: float
    #: Bytes this call fetched over the network. Less than `size_bytes` when an
    #: earlier run left a `.part`; more when a server ignored Range.
    transferred_bytes: int
    #: Time spent fetching, without the re-hash.
    transfer_seconds: float
    #: Byte offset at which each resumed request started.
    resumed_from: tuple[int, ...]
    #: Times the server ignored a Range request, forcing a restart from zero.
    restarts: int

    @property
    def mib_per_s(self) -> float:
        """Network throughput of this call: what it fetched over the time it spent fetching."""
        if self.transfer_seconds <= 0:
            return 0.0
        return self.transferred_bytes / _MIB / self.transfer_seconds


@dataclass
class _Transfer:
    """Mutable state shared by the attempts of one download."""

    hasher: "hashlib._Hash" = field(default_factory=hashlib.sha256)
    hashed_bytes: int = 0
    #: False once bytes on disk were written by another process (or skipped by
    #: the hasher), so the streamed hash no longer covers the whole file.
    hash_covers_file: bool = True
    #: Every byte received over the network, across attempts and restarts.
    transferred_bytes: int = 0
    resumed_from: list[int] = field(default_factory=list)
    restarts: int = 0
    next_progress_pct: int = PROGRESS_STEP_PCT

    def reset_hash(self) -> None:
        self.hasher = hashlib.sha256()
        self.hashed_bytes = 0
        self.hash_covers_file = True


def part_path(dest: Path) -> Path:
    """Where `dest` is assembled while incomplete."""
    return dest.with_name(dest.name + ".part")


def sha256_file(path: Path, *, chunk_bytes: int = CHUNK_BYTES) -> str:
    """SHA-256 of a file on disk, read in chunks so memory stays flat."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_bytes):
            hasher.update(chunk)
    return hasher.hexdigest()


def head(
    url: str,
    *,
    user_agent: str = USER_AGENT,
    timeout_s: float = TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> RemoteFile:
    """HEAD `url` and return its size, ETag and Last-Modified.

    Both Content-Length and ETag are required. Without the length a truncated
    file can't be detected, and without the ETag a resume can't prove it is
    appending to the same file.
    """

    def _once() -> RemoteFile:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            length = response.headers.get("Content-Length")
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
        if length is None or not length.isdigit():
            raise PermanentIngestError(f"HEAD {url}: no usable Content-Length ({length!r})")
        if not etag:
            raise PermanentIngestError(
                f"HEAD {url}: no ETag, so a resumed download could not prove "
                "it is appending to the same file"
            )
        return RemoteFile(url=url, size_bytes=int(length), etag=etag, last_modified=last_modified)

    return retry_with_backoff(_once, description=f"HEAD {url}", sleep=sleep)


def _check_content_range(header: str | None, offset: int, size_bytes: int) -> None:
    """A 206 must cover exactly [offset, size): anything else would misplace bytes."""
    match = _CONTENT_RANGE.match(header or "")
    if (
        match is None
        or int(match.group(1)) != offset
        or int(match.group(2)) != size_bytes - 1
        or int(match.group(3)) != size_bytes
    ):
        raise PermanentIngestError(
            f"server answered Range bytes={offset}- with Content-Range {header!r}; "
            f"expected 'bytes {offset}-{size_bytes - 1}/{size_bytes}'"
        )


def download(
    remote: RemoteFile,
    dest: Path,
    *,
    user_agent: str = USER_AGENT,
    timeout_s: float = TIMEOUT_S,
    chunk_bytes: int = CHUNK_BYTES,
    attempts: int = DOWNLOAD_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadResult:
    """Download `remote` to `dest`, resuming from `dest.part` if it exists."""
    part = part_path(dest)
    state = _Transfer()
    started = time.monotonic()

    initial = part.stat().st_size if part.exists() else 0
    if initial > remote.size_bytes:
        logger.warning(
            "download_part_oversized part_bytes=%d size_bytes=%d action=restart",
            initial,
            remote.size_bytes,
        )
        part.unlink()
        initial = 0
    if initial:
        # These bytes came from an earlier run; this process never hashed them.
        state.hash_covers_file = False

    def _attempt() -> None:
        offset = part.stat().st_size if part.exists() else 0
        if offset == remote.size_bytes:
            return

        headers = {"User-Agent": user_agent}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            # Serve the range only if the file is still this version.
            headers["If-Range"] = remote.etag
            state.resumed_from.append(offset)
            logger.info(
                "download_resume offset=%d remaining_bytes=%d",
                offset,
                remote.size_bytes - offset,
            )

        request = urllib.request.Request(remote.url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            etag = response.headers.get("ETag")
            if etag != remote.etag:
                raise PermanentIngestError(
                    f"source changed during download: ETag was {remote.etag} at the "
                    f"start and is {etag} now. Adopting a new release is a deliberate "
                    "decision, not something to resume into."
                )

            if offset and response.status == 200:
                logger.warning("download_restart_no_range offset=%d", offset)
                state.restarts += 1
                offset = 0
            elif offset and response.status == 206:
                _check_content_range(response.headers.get("Content-Range"), offset, remote.size_bytes)
            elif response.status != 200:
                raise PermanentIngestError(f"GET {remote.url}: unexpected HTTP {response.status}")

            if offset == 0:
                state.reset_hash()
            elif state.hashed_bytes != offset:
                state.hash_covers_file = False

            received = 0
            with open(part, "ab" if offset else "wb") as fh:
                while chunk := response.read(chunk_bytes):
                    fh.write(chunk)
                    state.hasher.update(chunk)
                    state.hashed_bytes += len(chunk)
                    state.transferred_bytes += len(chunk)
                    received += len(chunk)
                    _log_progress(state, offset + received, remote.size_bytes)
                fh.flush()
                os.fsync(fh.fileno())

        end = offset + received
        if end < remote.size_bytes:
            raise TruncatedTransfer(f"connection closed at byte {end:,} of {remote.size_bytes:,}")
        if end > remote.size_bytes:
            raise PermanentIngestError(f"server sent {end:,} bytes, {remote.size_bytes:,} advertised")

    retry_with_backoff(_attempt, description=f"GET {remote.url}", attempts=attempts, sleep=sleep)
    transfer_seconds = time.monotonic() - started

    size_on_disk = part.stat().st_size
    if size_on_disk != remote.size_bytes:
        raise PermanentIngestError(f"{part}: {size_on_disk:,} bytes on disk, {remote.size_bytes:,} expected")

    if state.hash_covers_file and state.hashed_bytes == size_on_disk:
        digest = state.hasher.hexdigest()
    else:
        # A hash can't be continued across processes, so a file finished by a
        # resumed run is hashed again from disk, from its first byte.
        logger.info("sha256_rehash reason=resumed_across_runs bytes=%d", size_on_disk)
        digest = sha256_file(part, chunk_bytes=chunk_bytes)

    os.replace(part, dest)
    result = DownloadResult(
        path=dest,
        size_bytes=size_on_disk,
        sha256=digest,
        seconds=time.monotonic() - started,
        transferred_bytes=state.transferred_bytes,
        transfer_seconds=transfer_seconds,
        resumed_from=tuple(state.resumed_from),
        restarts=state.restarts,
    )
    logger.info(
        "download_done bytes=%d transferred_bytes=%d seconds=%.1f transfer_seconds=%.1f "
        "mib_s=%.2f resumes=%d restarts=%d sha256=%s",
        result.size_bytes,
        result.transferred_bytes,
        result.seconds,
        result.transfer_seconds,
        result.mib_per_s,
        len(result.resumed_from),
        result.restarts,
        result.sha256,
    )
    return result


def _log_progress(state: _Transfer, done: int, total: int) -> None:
    pct = done * 100 // total if total else 100
    if pct >= state.next_progress_pct:
        logger.info("download_progress pct=%d bytes=%d of=%d", pct, done, total)
        state.next_progress_pct = (pct // PROGRESS_STEP_PCT + 1) * PROGRESS_STEP_PCT
