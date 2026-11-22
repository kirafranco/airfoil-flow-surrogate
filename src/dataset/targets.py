"""Targets scaled by the flow's own scale, and the exact inverse.

Each target is divided by the reference value the free stream sets for it:

    u_x / U∞,   u_y / U∞,   p / (½U∞²),   ν_t / (U∞·c)

The pressure one is the pressure coefficient Cp. The exploration of the data
showed why this helps: in m²/s² the pressure range grows with U∞², while Cp
depends on the angle of attack and not on the speed. A network predicting
Cp doesn't have to learn the U∞² scaling as well.

The scaled values are then standardised (mean 0, std 1 per target) with
statistics of the training sample, so the four terms of the loss are on a
comparable scale. Predictions are mapped back with `unscale` before any metric
is computed, so every reported number is in physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.etl.names import CHORD_M

Array = np.ndarray | torch.Tensor


def reference_column(u_inf: Array, j: int) -> Array:
    """Reference value of target j (u_x, u_y, p, ν_t) from per-point U∞."""
    return (u_inf, u_inf, 0.5 * u_inf * u_inf, u_inf * CHORD_M)[j]


def reference(u_inf: Array) -> Array:
    """(n, 4) reference values for (u_x, u_y, p, ν_t), from per-point U∞."""
    columns = [reference_column(u_inf, j) for j in range(4)]
    return torch.stack(columns, dim=1) if isinstance(u_inf, torch.Tensor) else np.stack(columns, axis=1)


@dataclass(frozen=True)
class TargetScaling:
    mean: tuple[float, float, float, float]
    std: tuple[float, float, float, float]
    method: str = "physics"

    @classmethod
    def fit(cls, targets: np.ndarray, u_inf: np.ndarray) -> TargetScaling:
        """Mean and std of the scaled targets, one column at a time in float64 to bound memory."""
        u_inf = u_inf.astype(np.float64)
        means, stds = [], []
        for j in range(targets.shape[1]):
            scaled = targets[:, j].astype(np.float64) / reference_column(u_inf, j)
            means.append(float(scaled.mean()))
            stds.append(float(scaled.std()))
        return cls(mean=tuple(means), std=tuple(stds))  # type: ignore[arg-type]

    def _stats(self, like: Array) -> tuple[Array, Array]:
        if isinstance(like, torch.Tensor):
            return (
                torch.as_tensor(self.mean, dtype=like.dtype, device=like.device),
                torch.as_tensor(self.std, dtype=like.dtype, device=like.device),
            )
        return np.asarray(self.mean, dtype=like.dtype), np.asarray(self.std, dtype=like.dtype)

    def scale(self, targets: Array, u_inf: Array) -> Array:
        mean, std = self._stats(targets)
        return (targets / reference(u_inf) - mean) / std

    def unscale(self, scaled: Array, u_inf: Array) -> Array:
        mean, std = self._stats(scaled)
        return (scaled * std + mean) * reference(u_inf)

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TargetScaling:
        return cls(mean=tuple(payload["mean"]), std=tuple(payload["std"]), method=payload["method"])  # type: ignore[arg-type]
