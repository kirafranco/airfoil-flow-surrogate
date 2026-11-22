"""The interim AirfRANS tables as model inputs and targets.

Reads what the ETL wrote (see `src.etl.schema`) and gives every model the same
view of a simulation:

- 7 node features, the AirfRANS paper's baseline inputs: position, distance to
  the airfoil, inlet velocity, and the outward surface normal (0 off the
  surface). The first five are standardised with statistics fitted on the
  training simulations only (`normalisation.json`); normals are unit vectors
  already and pass through.
- 4 targets in physical units: u_x and u_y (m/s), kinematic pressure p (m²/s²)
  and turbulent viscosity ν_t (m²/s).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import INTERIM_ROOT, PROCESSED_ROOT
from src.etl.schema import NODES_FILE, NODES_SCHEMA, SIMULATIONS_SCHEMA, SURFACE_FILE, SURFACE_SCHEMA, read_table
from src.etl.splits import names_sha256

FEATURES = ("x", "y", "sdf", "inlet_u_x", "inlet_u_y", "normal_x", "normal_y")
STANDARDISED_FEATURES = FEATURES[:5]
TARGETS = ("u_x", "u_y", "p", "nut")

ETL_FILE = "etl.json"
SIMULATIONS_FILE = "simulations.parquet"
SPLITS_FILE = "splits.json"
NORMALISATION_FILE = "normalisation.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Dataset:
    """One complete interim snapshot, with its splits and normalisation statistics."""

    interim_dir: Path
    processed_dir: Path
    simulations: pd.DataFrame  # one row per simulation, indexed by name
    splits: dict[str, list[str]]
    normalisation: dict[str, Any]
    etl: dict[str, Any]

    @classmethod
    def open(cls, interim_dir: Path | None = None, processed_dir: Path | None = None) -> Dataset:
        """By default, the single complete snapshot under data/interim/airfrans/."""
        if interim_dir is None:
            root = INTERIM_ROOT / "airfrans"
            complete = sorted(
                d for d in root.iterdir() if (d / ETL_FILE).is_file() and _read_json(d / ETL_FILE).get("status") == "complete"
            ) if root.is_dir() else []
            if len(complete) != 1:
                raise FileNotFoundError(
                    f"expected one complete interim snapshot under {root}, found {[d.name for d in complete]}: "
                    "run python -m src.etl.pipeline"
                )
            interim_dir = complete[0]
        if processed_dir is None:
            processed_dir = PROCESSED_ROOT / "airfrans" / interim_dir.name
        simulations = read_table(interim_dir / SIMULATIONS_FILE, SIMULATIONS_SCHEMA).to_pandas().set_index("name")
        return cls(
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            simulations=simulations,
            splits=_read_json(processed_dir / SPLITS_FILE),
            normalisation=_read_json(processed_dir / NORMALISATION_FILE),
            etl=_read_json(interim_dir / ETL_FILE),
        )

    def names(self, split: str) -> list[str]:
        return list(self.splits[split])

    def fingerprint(self) -> dict[str, Any]:
        """What a trained model or a set of metrics was computed from."""
        return {
            "interim_snapshot": self.interim_dir.name,
            "raw_sha256": self.etl.get("raw_sha256"),
            "schema_version": self.etl.get("schema_version"),
            "etl_code_commit": self.etl.get("etl_code_commit"),
            "split_names_sha256": {split: names_sha256(self.splits[split]) for split in ("train", "val", "test")},
            "normalisation_sha256": hashlib.sha256((self.processed_dir / NORMALISATION_FILE).read_bytes()).hexdigest(),
        }


@dataclass(frozen=True)
class Simulation:
    name: str
    u_inf: float
    aoa_deg: float
    inlet_u_x: float
    inlet_u_y: float
    xy: np.ndarray  # (n, 2) float64
    sdf: np.ndarray  # (n,) float64
    is_surface: np.ndarray  # (n,) bool
    normals: np.ndarray  # (n, 2) float32, outward on the surface, 0 elsewhere
    targets: np.ndarray  # (n, 4) float32: u_x, u_y, p, nut in physical units, as stored
    #: Airfoil surface segments as pairs of node ids.
    segments: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.xy.shape[0])


def load_simulation(dataset: Dataset, name: str) -> Simulation:
    sim_dir = dataset.interim_dir / "simulations" / name
    nodes = read_table(sim_dir / NODES_FILE, NODES_SCHEMA)
    surface = read_table(sim_dir / SURFACE_FILE, SURFACE_SCHEMA)
    column = lambda c: nodes.column(c).to_numpy()  # noqa: E731
    row = dataset.simulations.loc[name]
    return Simulation(
        name=name,
        u_inf=float(row.u_inf),
        aoa_deg=float(row.aoa_deg),
        inlet_u_x=float(row.inlet_u_x),
        inlet_u_y=float(row.inlet_u_y),
        xy=np.stack([column("x"), column("y")], axis=1),
        sdf=column("sdf"),
        is_surface=column("is_surface"),
        normals=np.stack([column("normal_x"), column("normal_y")], axis=1),
        targets=np.stack([column(c) for c in TARGETS], axis=1),
        segments=np.stack([surface.column("node_a").to_numpy(), surface.column("node_b").to_numpy()], axis=1).astype(np.int64),
    )


def features(sim: Simulation, normalisation: dict[str, Any]) -> np.ndarray:
    """The 7 node features, (n, 7) float32."""
    stats = normalisation["channels"]

    def standardised(name: str, values: np.ndarray | float) -> np.ndarray:
        return (np.broadcast_to(values, (sim.n_nodes,)) - stats[name]["mean"]) / stats[name]["std"]

    columns = [
        standardised("x", sim.xy[:, 0]),
        standardised("y", sim.xy[:, 1]),
        standardised("sdf", sim.sdf),
        standardised("inlet_u_x", sim.inlet_u_x),
        standardised("inlet_u_y", sim.inlet_u_y),
        sim.normals[:, 0],
        sim.normals[:, 1],
    ]
    return np.stack(columns, axis=1).astype(np.float32)


@dataclass(frozen=True)
class Sample:
    """Training points drawn from a set of simulations, concatenated."""

    names: tuple[str, ...]
    features: np.ndarray  # (N, 7) float32
    targets: np.ndarray  # (N, 4) float32, physical units
    is_surface: np.ndarray  # (N,) bool
    simulation: np.ndarray  # (N,) int32, index into `names`
    u_inf: np.ndarray  # (len(names),) float64

    @property
    def u_inf_per_point(self) -> np.ndarray:
        return self.u_inf[self.simulation]


def sample_points(
    dataset: Dataset, names: list[str] | tuple[str, ...], *, volume_points: int, seed: int
) -> Sample:
    """Every surface node plus `volume_points` random other nodes from each simulation.

    Surface nodes are under 1 % of a mesh, so a uniform sample would hold a few
    hundred of them and the surface term of the loss would be noise. Arrays are
    preallocated from the node counts in simulations.parquet, so the sample is
    never held twice in memory.
    """
    names = tuple(sorted(names))
    rows = dataset.simulations.loc[list(names)]
    sizes = rows.n_surface.to_numpy() + np.minimum(volume_points, (rows.n_nodes - rows.n_surface).to_numpy())
    total = int(sizes.sum())
    out_features = np.empty((total, len(FEATURES)), dtype=np.float32)
    out_targets = np.empty((total, len(TARGETS)), dtype=np.float32)
    out_surface = np.empty(total, dtype=bool)
    out_simulation = np.empty(total, dtype=np.int32)

    rng = np.random.default_rng(seed)
    start = 0
    for i, name in enumerate(names):
        sim = load_simulation(dataset, name)
        volume = np.flatnonzero(~sim.is_surface)
        chosen = np.sort(rng.choice(volume, size=min(volume_points, len(volume)), replace=False))
        index = np.concatenate([np.flatnonzero(sim.is_surface), chosen])
        stop = start + len(index)
        out_features[start:stop] = features(sim, dataset.normalisation)[index]
        out_targets[start:stop] = sim.targets[index]
        out_surface[start:stop] = sim.is_surface[index]
        out_simulation[start:stop] = i
        start = stop
    assert start == total, (start, total)
    return Sample(
        names=names,
        features=out_features,
        targets=out_targets,
        is_surface=out_surface,
        simulation=out_simulation,
        u_inf=rows.u_inf.to_numpy(dtype=np.float64),
    )
