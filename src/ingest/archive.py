"""Archive gate: decide from a zip's table of contents whether it is safe to extract.

A zip's central directory lists every entry with its path, sizes, CRC-32 and
file attributes, and it can be read without decompressing anything. The gate
reads it and rejects the archive before a single byte is written if any entry
could escape the extraction directory (absolute path, `..`, drive letter,
backslash, symlink: the "zip-slip" family), could expand far beyond its
compressed size (a zip bomb), is encrypted, or has a type outside the allowlist.
It then checks the dataset's own structure: a manifest with disjoint train and
test splits, and exactly three files for every simulation it names.

Extraction writes every member itself instead of calling `extractall`, so each
target path is checked again at write time and each member is capped at its
declared size. The tree is built in `Dataset.partial/` and renamed to
`Dataset/` only when complete, so an interrupted extraction can't be mistaken
for a finished one.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from src.ingest.retry import PermanentIngestError

logger = logging.getLogger(__name__)

TOP_LEVEL_DIR = "Dataset"
MANIFEST_NAME = f"{TOP_LEVEL_DIR}/manifest.json"
ALLOWED_SUFFIXES = frozenset({".vtu", ".vtp", ".json"})
#: Every simulation directory holds the volume mesh and two boundary surfaces.
SIMULATION_FILE_SUFFIXES = ("_internal.vtu", "_aerofoil.vtp", "_freestream.vtp")
REQUIRED_SPLITS = ("full_train", "full_test")

#: AirfRANS is 15.0 GB uncompressed; the cap leaves room without being unbounded.
MAX_TOTAL_UNCOMPRESSED_BYTES = 20 * 10**9
#: AirfRANS meshes compress about 1.5x. Deflate tops out near 1000x, which is
#: what a bomb uses.
MAX_COMPRESSION_RATIO = 20.0
#: A ratio only signals a bomb when the expansion is large in absolute terms.
#: Small, repetitive text such as the manifest can legitimately compress more.
RATIO_CHECK_MIN_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_VIOLATIONS_LOGGED = 10
EXTRACT_CHUNK_BYTES = 8 * 1024 * 1024
EXTRACT_PROGRESS_STEP_PCT = 10

_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")
_S_IFMT = 0o170000
_S_IFLNK = 0o120000
_ENCRYPTED_FLAG = 0x1
_MIB = 1024 * 1024


@dataclass(frozen=True)
class ZipEntry:
    """One central-directory record, as written to listing.json."""

    name: str
    size: int
    compressed_size: int
    crc32: int
    is_dir: bool

    @classmethod
    def from_info(cls, info: zipfile.ZipInfo) -> ZipEntry:
        return cls(
            name=info.filename,
            size=info.file_size,
            compressed_size=info.compress_size,
            crc32=info.CRC,
            is_dir=info.is_dir(),
        )

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> ZipEntry:
        return cls(
            name=str(record["name"]),
            size=int(record["size"]),
            compressed_size=int(record["compressed_size"]),
            crc32=int(record["crc32"]),
            is_dir=bool(record["is_dir"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "compressed_size": self.compressed_size,
            "crc32": self.crc32,
            "is_dir": self.is_dir,
        }

    @property
    def relative_path(self) -> PurePosixPath:
        """Path inside the extracted tree, without the archive's top-level directory."""
        return PurePosixPath(self.name).relative_to(TOP_LEVEL_DIR)


@dataclass(frozen=True)
class ArchiveReport:
    """What the gate saw in an archive it accepted."""

    entries: tuple[ZipEntry, ...]
    manifest: dict[str, list[str]]
    simulations: tuple[str, ...]
    #: Simulation directories that no manifest split names. Logged, not rejected.
    unlisted_dirs: tuple[str, ...]

    @property
    def files(self) -> tuple[ZipEntry, ...]:
        return tuple(entry for entry in self.entries if not entry.is_dir)

    @property
    def suffix_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(PurePosixPath(e.name).suffix for e in self.files).items()))

    @property
    def total_uncompressed(self) -> int:
        return sum(entry.size for entry in self.entries)

    @property
    def total_compressed(self) -> int:
        return sum(entry.compressed_size for entry in self.entries)

    @property
    def split_sizes(self) -> dict[str, int]:
        return {name: len(members) for name, members in self.manifest.items()}

    def listing(self) -> list[dict[str, Any]]:
        return [entry.as_dict() for entry in self.entries]


@dataclass(frozen=True)
class ExtractResult:
    files: int
    bytes_written: int
    seconds: float

    @property
    def mib_per_s(self) -> float:
        return self.bytes_written / _MIB / self.seconds if self.seconds > 0 else 0.0


@dataclass(frozen=True)
class TreeCheck:
    """How an extracted tree compares with the listing it was extracted from."""

    expected_files: int
    missing: int
    wrong_size: int

    @property
    def complete(self) -> bool:
        return self.missing == 0 and self.wrong_size == 0


def entry_violations(info: zipfile.ZipInfo, *, max_ratio: float = MAX_COMPRESSION_RATIO) -> list[str]:
    """Every reason this single entry is unsafe to extract; empty when it is fine."""
    name = info.filename
    problems: list[str] = []

    if "\\" in name:
        problems.append("backslash in path")
    if name.startswith("/"):
        problems.append("absolute path")
    if _DRIVE_LETTER.match(name):
        problems.append("drive letter")
    parts = name.rstrip("/").split("/")
    if ".." in parts:
        problems.append("'..' component")
    if any(part in ("", ".") for part in parts):
        problems.append("empty or '.' component")
    if parts[0] != TOP_LEVEL_DIR:
        problems.append(f"outside {TOP_LEVEL_DIR}/")
    if (info.external_attr >> 16) & _S_IFMT == _S_IFLNK:
        problems.append("symlink")
    if info.flag_bits & _ENCRYPTED_FLAG:
        problems.append("encrypted")

    if not info.is_dir():
        suffix = PurePosixPath(name).suffix
        if suffix not in ALLOWED_SUFFIXES:
            problems.append(f"extension {suffix or '(none)'} not in {sorted(ALLOWED_SUFFIXES)}")
        if info.file_size >= RATIO_CHECK_MIN_BYTES:
            ratio = info.file_size / info.compress_size if info.compress_size else float("inf")
            if ratio > max_ratio:
                problems.append(f"compression ratio {ratio:.0f} > {max_ratio:.0f}")

    return [f"{name!r}: {problem}" for problem in problems]


def _reject(problems: list[str]) -> None:
    for problem in problems[:MAX_VIOLATIONS_LOGGED]:
        logger.error("archive_violation %s", problem)
    raise PermanentIngestError(
        f"archive rejected, nothing extracted: {len(problems)} problem(s), first: {problems[0]}"
    )


def _read_manifest(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    try:
        info = zf.getinfo(MANIFEST_NAME)
    except KeyError:
        raise PermanentIngestError(f"archive rejected: {MANIFEST_NAME} is missing") from None
    if info.file_size > MAX_MANIFEST_BYTES:
        raise PermanentIngestError(f"archive rejected: {MANIFEST_NAME} is {info.file_size:,} bytes")
    try:
        manifest = json.loads(zf.read(info))
    except (ValueError, UnicodeDecodeError) as exc:
        raise PermanentIngestError(f"archive rejected: {MANIFEST_NAME} is not valid JSON ({exc})") from None

    if not isinstance(manifest, dict):
        raise PermanentIngestError(f"archive rejected: {MANIFEST_NAME} is not a JSON object")
    for split in REQUIRED_SPLITS:
        if split not in manifest:
            raise PermanentIngestError(f"archive rejected: {MANIFEST_NAME} has no {split!r} split")
    for split, members in manifest.items():
        if not isinstance(members, list) or not all(isinstance(m, str) for m in members):
            raise PermanentIngestError(f"archive rejected: split {split!r} is not a list of names")
        if len(set(members)) != len(members):
            raise PermanentIngestError(f"archive rejected: split {split!r} names a simulation twice")

    overlap = set(manifest["full_train"]) & set(manifest["full_test"])
    if overlap:
        raise PermanentIngestError(
            f"archive rejected: {len(overlap)} simulation(s) in both full_train and full_test, "
            f"e.g. {sorted(overlap)[0]!r}"
        )
    return {split: list(members) for split, members in manifest.items()}


def _simulation_violations(
    manifest: dict[str, list[str]], infos: list[zipfile.ZipInfo]
) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
    """Check that every simulation the manifest names has exactly its three files."""
    files_by_dir: dict[str, set[str]] = defaultdict(set)
    for info in infos:
        parts = info.filename.split("/")
        if not info.is_dir() and len(parts) == 3:
            files_by_dir[parts[1]].add(parts[2])

    simulations = tuple(sorted(set().union(*manifest.values())))
    problems: list[str] = []
    for simulation in simulations:
        expected = {simulation + suffix for suffix in SIMULATION_FILE_SUFFIXES}
        actual = files_by_dir.get(simulation, set())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            problems.append(f"simulation {simulation!r}: missing {missing}, unexpected {extra}")

    unlisted = tuple(sorted(set(files_by_dir) - set(simulations)))
    return problems, simulations, unlisted


def inspect_zip(
    path: Path,
    *,
    max_total_bytes: int = MAX_TOTAL_UNCOMPRESSED_BYTES,
    max_ratio: float = MAX_COMPRESSION_RATIO,
) -> ArchiveReport:
    """Gate the archive at `path`. Raises PermanentIngestError, having extracted nothing."""
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise PermanentIngestError(f"{path} is not a readable zip: {exc}") from None

    with zf:
        infos = zf.infolist()
        problems: list[str] = []
        for info in infos:
            problems.extend(entry_violations(info, max_ratio=max_ratio))

        duplicates = sorted(name for name, n in Counter(i.filename for i in infos).items() if n > 1)
        problems.extend(f"{name!r}: duplicate entry" for name in duplicates)

        total = sum(info.file_size for info in infos)
        if total > max_total_bytes:
            problems.append(f"total uncompressed size {total:,} bytes > {max_total_bytes:,}")

        # Nothing is read from an archive with unsafe entries, the manifest included.
        if problems:
            _reject(problems)

        manifest = _read_manifest(zf)

    sim_problems, simulations, unlisted = _simulation_violations(manifest, infos)
    if sim_problems:
        _reject(sim_problems)

    return ArchiveReport(
        entries=tuple(ZipEntry.from_info(info) for info in infos),
        manifest=manifest,
        simulations=simulations,
        unlisted_dirs=unlisted,
    )


def _copy_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> int:
    """Stream one member to disk, refusing to write past its declared size."""
    written = 0
    try:
        with zf.open(info) as source, open(target, "wb") as sink:
            # Reading to the end makes zipfile verify the member's CRC-32.
            while chunk := source.read(EXTRACT_CHUNK_BYTES):
                written += len(chunk)
                if written > info.file_size:
                    raise PermanentIngestError(
                        f"{info.filename!r} expands beyond its declared {info.file_size:,} bytes"
                    )
                sink.write(chunk)
    except zipfile.BadZipFile as exc:
        raise PermanentIngestError(f"{info.filename!r} failed its integrity check: {exc}") from None
    return written


def safe_extract(zip_path: Path, report: ArchiveReport, dest: Path) -> ExtractResult:
    """Extract the gated archive into `dest`, through `dest.partial` and a final rename.

    `dest` is the archive's top-level directory: entry `Dataset/a/b.vtu`
    lands at `dest/a/b.vtu`.
    """
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        logger.warning("extract_partial_removed path=%s", partial)
        shutil.rmtree(partial)
    if dest.exists():
        # Only called when the tree failed its completeness check; it is
        # derived from the zip and is rebuilt from it.
        logger.warning("extract_incomplete_tree_removed path=%s", dest)
        shutil.rmtree(dest)

    partial.mkdir(parents=True)
    root = partial.resolve()
    files = report.files
    total_bytes = sum(entry.size for entry in files)
    written_total = 0
    next_pct = EXTRACT_PROGRESS_STEP_PCT
    started = time.monotonic()

    with zipfile.ZipFile(zip_path) as zf:
        for entry in files:
            target = (root / entry.relative_path).resolve()
            if root not in target.parents:
                raise PermanentIngestError(f"{entry.name!r} would be written outside {root}")
            target.parent.mkdir(parents=True, exist_ok=True)

            written = _copy_member(zf, zf.getinfo(entry.name), target)
            if written != entry.size:
                raise PermanentIngestError(
                    f"{entry.name!r}: wrote {written:,} bytes, the listing says {entry.size:,}"
                )
            written_total += written

            pct = written_total * 100 // total_bytes if total_bytes else 100
            if pct >= next_pct:
                logger.info("extract_progress pct=%d bytes=%d of=%d", pct, written_total, total_bytes)
                next_pct = (pct // EXTRACT_PROGRESS_STEP_PCT + 1) * EXTRACT_PROGRESS_STEP_PCT

    os.replace(partial, dest)
    result = ExtractResult(files=len(files), bytes_written=written_total, seconds=time.monotonic() - started)
    logger.info(
        "extract_done files=%d bytes=%d seconds=%.1f mib_s=%.1f",
        result.files,
        result.bytes_written,
        result.seconds,
        result.mib_per_s,
    )
    return result


def check_tree(dest: Path, entries: tuple[ZipEntry, ...]) -> TreeCheck:
    """Compare an extracted tree with the listing: every file present, every size equal."""
    files = [entry for entry in entries if not entry.is_dir]
    missing = 0
    wrong_size = 0
    for entry in files:
        try:
            size = (dest / entry.relative_path).stat().st_size
        except FileNotFoundError:
            missing += 1
            continue
        if size != entry.size:
            wrong_size += 1
    return TreeCheck(expected_files=len(files), missing=missing, wrong_size=wrong_size)
