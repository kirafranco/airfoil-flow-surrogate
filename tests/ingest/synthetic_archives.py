"""Synthetic AirfRANS-shaped archives for the ingestion tests: one well-formed, many hostile.

The well-formed archive has the real layout in miniature: `Dataset/manifest.json`
plus one directory per simulation holding `_internal.vtu`, `_aerofoil.vtp` and
`_freestream.vtp`. Member contents are deterministic random bytes, so they
compress about as poorly as real meshes do.
"""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from typing import Callable

SIMULATIONS = (
    "airFoil2D_SST_31.677_-0.742_2.926_5.236_0.0",
    "airFoil2D_SST_57.932_8.393_0.0_0.0_4.012_8.584",
)
SIM_SUFFIXES = ("_internal.vtu", "_aerofoil.vtp", "_freestream.vtp")


def member_bytes(name: str, size: int = 4096) -> bytes:
    return random.Random(name).randbytes(size)


def member_name(simulation: str, suffix: str) -> str:
    return f"Dataset/{simulation}/{simulation}{suffix}"


def build_dataset_zip(
    path: Path,
    *,
    simulations: tuple[str, ...] = SIMULATIONS,
    manifest: dict[str, list[str]] | None = None,
    with_manifest: bool = True,
    skip: tuple[str, ...] = (),
    extra: tuple[tuple[str | zipfile.ZipInfo, bytes], ...] = (),
) -> Path:
    if manifest is None:
        manifest = {"full_train": [simulations[0]], "full_test": list(simulations[1:])}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.mkdir("Dataset")
        if with_manifest:
            zf.writestr("Dataset/manifest.json", json.dumps(manifest))
        for simulation in simulations:
            zf.mkdir(f"Dataset/{simulation}")
            for suffix in SIM_SUFFIXES:
                name = member_name(simulation, suffix)
                if name not in skip:
                    zf.writestr(name, member_bytes(name))
        for item, data in extra:
            zf.writestr(item, data)
    return path


def _symlink(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3  # Unix, so the mode bits in external_attr apply
    info.external_attr = 0o120777 << 16
    return info


S0, S1 = SIMULATIONS

#: name -> (builder, fragment the rejection message must contain)
ADVERSARIAL: dict[str, tuple[Callable[[Path], Path], str]] = {
    "absolute path": (
        lambda p: build_dataset_zip(p, extra=(("/Dataset/evil.vtu", b"x"),)),
        "absolute path",
    ),
    "dot-dot component": (
        lambda p: build_dataset_zip(p, extra=(("Dataset/../evil.vtu", b"x"),)),
        "'..' component",
    ),
    "drive letter": (
        lambda p: build_dataset_zip(p, extra=(("C:/Dataset/evil.vtu", b"x"),)),
        "drive letter",
    ),
    "backslash": (
        lambda p: build_dataset_zip(p, extra=(("Dataset\\evil.vtu", b"x"),)),
        "backslash",
    ),
    "symlink": (
        lambda p: build_dataset_zip(p, extra=((_symlink("Dataset/link.vtu"), b"/etc/passwd"),)),
        "symlink",
    ),
    "executable": (
        lambda p: build_dataset_zip(p, extra=(("Dataset/run.exe", b"MZ"),)),
        "extension .exe",
    ),
    "ratio bomb": (
        lambda p: build_dataset_zip(p, extra=(("Dataset/bomb.vtu", bytes(4 * 1024 * 1024)),)),
        "compression ratio",
    ),
    "missing manifest": (
        lambda p: build_dataset_zip(p, with_manifest=False),
        "manifest.json is missing",
    ),
    "manifest not JSON": (
        lambda p: build_dataset_zip(p, with_manifest=False, extra=(("Dataset/manifest.json", b"{not json"),)),
        "not valid JSON",
    ),
    "train/test overlap": (
        lambda p: build_dataset_zip(p, manifest={"full_train": [S0, S1], "full_test": [S1]}),
        "both full_train and full_test",
    ),
    "simulation missing its .vtp": (
        lambda p: build_dataset_zip(p, skip=(member_name(S0, "_aerofoil.vtp"),)),
        "_aerofoil.vtp'], unexpected []",
    ),
    "simulation with an extra file": (
        lambda p: build_dataset_zip(p, extra=((f"Dataset/{S0}/{S0}_notes.json", b"{}"),)),
        "_notes.json']",
    ),
    "manifest names an absent simulation": (
        lambda p: build_dataset_zip(p, manifest={"full_train": [S0, "airFoil2D_absent"], "full_test": [S1]}),
        "'airFoil2D_absent': missing",
    ),
}
