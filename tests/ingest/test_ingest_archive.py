"""Archive gate and extraction on synthetic AirfRANS-shaped zips."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest

from src.ingest.archive import (
    check_tree,
    entry_violations,
    inspect_zip,
    safe_extract,
)
from src.ingest.retry import PermanentIngestError
from synthetic_archives import (
    ADVERSARIAL,
    SIMULATIONS,
    build_dataset_zip,
    member_bytes,
    member_name,
)


def _info(name: str, *, attr: int = 0, flags: int = 0, size: int = 10, compressed: int = 10) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.external_attr = attr
    info.flag_bits = flags
    info.file_size = size
    info.compress_size = compressed
    return info


class TestEntryRules:
    def test_a_normal_member_passes(self) -> None:
        assert entry_violations(_info("Dataset/sim/sim_internal.vtu")) == []

    @pytest.mark.parametrize(
        ("info", "fragment"),
        [
            (_info("Dataset/x.vtu", flags=0x1), "encrypted"),
            (_info("Dataset/x.vtu", attr=0o120777 << 16), "symlink"),
            (_info("Dataset\\x.vtu"), "backslash"),
            (_info("D:/Dataset/x.vtu"), "drive letter"),
            (_info("Dataset/./x.vtu"), "'.' component"),
            (_info("Other/x.vtu"), "outside Dataset/"),
            (_info("Dataset/x"), "extension (none)"),
            (_info("Dataset/x.VTU"), "extension .VTU"),
        ],
    )
    def test_unsafe_members_are_named(self, info: zipfile.ZipInfo, fragment: str) -> None:
        assert any(fragment in problem for problem in entry_violations(info))

    def test_ratio_is_only_judged_on_large_members(self) -> None:
        """A small, repetitive manifest may compress 50x; that's not a bomb."""
        assert entry_violations(_info("Dataset/manifest.json", size=500_000, compressed=10_000)) == []
        assert entry_violations(_info("Dataset/x.vtu", size=2 * 1024 * 1024, compressed=1_000)) != []


class TestInspectZip:
    def test_a_well_formed_archive_passes(self, tmp_path: Path) -> None:
        report = inspect_zip(build_dataset_zip(tmp_path / "ok.zip"))

        assert report.simulations == tuple(sorted(SIMULATIONS))
        assert report.suffix_counts == {".json": 1, ".vtp": 4, ".vtu": 2}
        assert report.split_sizes == {"full_train": 1, "full_test": 1}
        # Dataset/, the manifest, and per simulation one directory and three files.
        assert len(report.entries) == 2 + 4 * len(SIMULATIONS)
        assert report.unlisted_dirs == ()

    @pytest.mark.parametrize("case", sorted(ADVERSARIAL))
    def test_a_hostile_archive_is_rejected(self, case: str, tmp_path: Path) -> None:
        build, fragment = ADVERSARIAL[case]
        with pytest.raises(PermanentIngestError, match=re.escape(fragment)):
            inspect_zip(build(tmp_path / "bad.zip"))

    def test_duplicate_entries_are_rejected(self, tmp_path: Path) -> None:
        name = member_name(SIMULATIONS[0], "_internal.vtu")
        with pytest.warns(UserWarning, match="Duplicate name"):
            path = build_dataset_zip(tmp_path / "dup.zip", extra=((name, b"shadow"),))
        with pytest.raises(PermanentIngestError, match="duplicate entry"):
            inspect_zip(path)

    def test_the_total_size_cap_applies(self, tmp_path: Path) -> None:
        with pytest.raises(PermanentIngestError, match="total uncompressed size"):
            inspect_zip(build_dataset_zip(tmp_path / "ok.zip"), max_total_bytes=1_000)

    def test_a_file_that_is_not_a_zip_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "Dataset.zip"
        path.write_bytes(b"<html>maintenance page</html>")
        with pytest.raises(PermanentIngestError, match="not a readable zip"):
            inspect_zip(path)


class TestSafeExtract:
    def test_extracts_every_member_byte_for_byte(self, tmp_path: Path) -> None:
        zip_path = build_dataset_zip(tmp_path / "ok.zip")
        report = inspect_zip(zip_path)
        dest = tmp_path / "snap" / "Dataset"

        result = safe_extract(zip_path, report, dest)

        assert result.files == len(report.files) == 7
        assert check_tree(dest, report.entries).complete
        name = member_name(SIMULATIONS[0], "_internal.vtu")
        assert (dest / SIMULATIONS[0] / f"{SIMULATIONS[0]}_internal.vtu").read_bytes() == member_bytes(name)
        assert not (dest.parent / "Dataset.partial").exists()

    def test_a_leftover_partial_tree_is_removed_and_redone(self, tmp_path: Path) -> None:
        zip_path = build_dataset_zip(tmp_path / "ok.zip")
        report = inspect_zip(zip_path)
        dest = tmp_path / "snap" / "Dataset"
        partial = dest.parent / "Dataset.partial"
        (partial / "stale").mkdir(parents=True)
        (partial / "stale" / "junk.vtu").write_bytes(b"from an interrupted run")

        safe_extract(zip_path, report, dest)

        assert not partial.exists()
        assert not (dest / "stale").exists()
        assert check_tree(dest, report.entries).complete

    def test_check_tree_counts_missing_and_resized_files(self, tmp_path: Path) -> None:
        zip_path = build_dataset_zip(tmp_path / "ok.zip")
        report = inspect_zip(zip_path)
        dest = tmp_path / "snap" / "Dataset"
        safe_extract(zip_path, report, dest)

        s0, s1 = SIMULATIONS
        (dest / s0 / f"{s0}_aerofoil.vtp").unlink()
        (dest / s1 / f"{s1}_internal.vtu").write_bytes(b"short")

        tree = check_tree(dest, report.entries)
        assert (tree.expected_files, tree.missing, tree.wrong_size) == (7, 1, 1)
        assert not tree.complete

    def test_a_corrupted_member_fails_its_crc_check(self, tmp_path: Path) -> None:
        # One real member is stored uncompressed, so its bytes can be found and flipped.
        name = member_name(SIMULATIONS[0], "_internal.vtu")
        content = member_bytes(name)
        zip_path = build_dataset_zip(tmp_path / "ok.zip", skip=(name,))
        with zipfile.ZipFile(zip_path, "a") as zf:
            zf.writestr(name, content, compress_type=zipfile.ZIP_STORED)
        report_before_corruption = inspect_zip(zip_path)

        raw = bytearray(zip_path.read_bytes())
        offset = raw.find(content)
        raw[offset + 100] ^= 0xFF  # same size, different bytes: only the CRC can tell
        zip_path.write_bytes(bytes(raw))

        with pytest.raises(PermanentIngestError, match="integrity check"):
            safe_extract(zip_path, report_before_corruption, tmp_path / "snap" / "Dataset")
        assert not (tmp_path / "snap" / "Dataset").exists()
