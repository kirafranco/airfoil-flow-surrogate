"""Project-wide paths.

Everything resolves from the repository root, so a module behaves the same
whether it is imported from a test, a notebook or a pipeline stage in the
container (where the root is /opt/project).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_ROOT: Path = PROJECT_ROOT / "data"

#: Immutable raw snapshots, by source and snapshot date:
#: data/raw/airfrans/<snapshot-date>/
AIRFRANS_RAW_ROOT: Path = DATA_ROOT / "raw" / "airfrans"

#: Cleaned and typed per-simulation tables.
INTERIM_ROOT: Path = DATA_ROOT / "interim"

#: Model-ready datasets and the normalisation statistics fitted on them.
PROCESSED_ROOT: Path = DATA_ROOT / "processed"

#: One directory per training run: weights plus metrics.json.
MODELS_ROOT: Path = DATA_ROOT / "models"

#: Predictions written back to VTK for ParaView.
EXPORTS_ROOT: Path = DATA_ROOT / "exports"
