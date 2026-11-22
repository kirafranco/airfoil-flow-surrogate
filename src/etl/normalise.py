"""Normalisation statistics, fitted on the training simulations' nodes only.

The 720 training meshes hold about 130 million nodes, more than fits in the
container's memory at once, so statistics are accumulated one simulation at a
time. Each batch's mean and sum of squared deviations are merged into the
running totals with Chan et al.'s pairwise formula, which stays accurate where
the naive "sum of squares minus square of sums" loses digits.

Quantiles can't be merged that way. Those of `nut` (very skewed: most values
are near 0, a few reach about 0.03) are computed on a fixed-seed sample of
nodes from each training simulation, so they are reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.etl.schema import NODES_FILE, NODES_SCHEMA, read_table

#: Columns read from nodes.parquet.
NODE_CHANNELS = ("x", "y", "sdf", "u_x", "u_y", "p", "nut")
#: Per-simulation constants, broadcast to every node as the model sees them.
INLET_CHANNELS = ("inlet_u_x", "inlet_u_y")
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)
QUANTILE_SAMPLE_PER_SIM = 20_000
SEED = 0


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations from the mean
    min: float = math.inf
    max: float = -math.inf

    def _merge(self, count: int, mean: float, m2: float, low: float, high: float) -> None:
        if count == 0:
            return
        total = self.count + count
        delta = mean - self.mean
        self.mean += delta * count / total
        self.m2 += m2 + delta * delta * self.count * count / total
        self.count = total
        self.min = min(self.min, low)
        self.max = max(self.max, high)

    def add(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64)
        if v.size:
            mean = float(v.mean())
            self._merge(v.size, mean, float(np.square(v - mean).sum()), float(v.min()), float(v.max()))

    def add_constant(self, value: float, count: int) -> None:
        self._merge(count, float(value), 0.0, float(value), float(value))

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.m2 / self.count) if self.count else math.nan

    def as_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "std": self.std, "min": self.min, "max": self.max, "count": self.count}


def fit_normalisation(
    simulations: Iterable[tuple[str, Path, float, float]],
    *,
    seed: int = SEED,
    sample_per_sim: int = QUANTILE_SAMPLE_PER_SIM,
) -> dict[str, Any]:
    """Statistics over the nodes of the given simulations: (name, directory, inlet_u_x, inlet_u_y).

    The caller passes the training simulations and nothing else; that is what
    keeps validation and test data out of the statistics.
    """
    stats = {channel: RunningStats() for channel in (*NODE_CHANNELS, *INLET_CHANNELS)}
    rng = np.random.default_rng(seed)
    nut_samples: list[np.ndarray] = []
    names: list[str] = []
    for name, sim_dir, inlet_u_x, inlet_u_y in sorted(simulations):
        nodes = read_table(sim_dir / NODES_FILE, NODES_SCHEMA, columns=list(NODE_CHANNELS))
        for channel in NODE_CHANNELS:
            stats[channel].add(nodes.column(channel).to_numpy())
        n = nodes.num_rows
        stats["inlet_u_x"].add_constant(inlet_u_x, n)
        stats["inlet_u_y"].add_constant(inlet_u_y, n)
        nut = nodes.column("nut").to_numpy()
        nut_samples.append(nut[rng.choice(n, size=min(sample_per_sim, n), replace=False)])
        names.append(name)

    sample = np.concatenate(nut_samples) if nut_samples else np.empty(0)
    return {
        "channels": {channel: s.as_dict() for channel, s in stats.items()},
        "nut_quantiles": {
            f"q{round(q * 100):02d}": float(np.quantile(sample, q)) if sample.size else None for q in QUANTILES
        },
        "nut_quantile_sample": {"nodes_per_simulation": sample_per_sim, "seed": seed, "size": int(sample.size)},
        "fitted_on": {"split": "train", "simulations": len(names)},
    }
