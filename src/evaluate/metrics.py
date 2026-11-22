"""Field errors, surface pressure error and the lift coefficient.

Everything here takes physical units and computes in float64.

Lift comes from the pressure force on the airfoil, F = −∮ p n dl with n the
outward normal, integrated over the surface segments with the trapezoid rule,
then projected on the direction perpendicular to the inflow:

    C_L = F · (−sin α, cos α) / (½ U∞² c)

`p` is kinematic pressure (pressure / density), so no density appears. The data
carries no wall shear stress, so this is the pressure part of lift only. For
lift, unlike drag, that is by far the larger part.
"""

from __future__ import annotations

import math

import numpy as np

from src.etl.names import CHORD_M


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """‖pred − true‖₂ / ‖true‖₂: 0 is perfect, 1 is as far off as predicting zero."""
    diff = np.linalg.norm(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64))
    norm = np.linalg.norm(np.asarray(true, dtype=np.float64))
    return float(diff / norm) if norm > 0 else math.nan


def standardised_mse(pred: np.ndarray, true: np.ndarray, std: float) -> float:
    """Mean squared error in units of the training standard deviation (the AirfRANS paper's metric)."""
    diff = (np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)) / std
    return float(np.mean(diff * diff))


def cp_rmse(p_pred: np.ndarray, p_true: np.ndarray, u_inf: float) -> float:
    """Root mean square error of the pressure coefficient p / (½U∞²)."""
    diff = (np.asarray(p_pred, dtype=np.float64) - np.asarray(p_true, dtype=np.float64)) / (0.5 * u_inf**2)
    return float(np.sqrt(np.mean(diff * diff)))


def lift_coefficient(
    xy: np.ndarray,
    p: np.ndarray,
    normals: np.ndarray,
    segments: np.ndarray,
    *,
    aoa_deg: float,
    u_inf: float,
    chord: float = CHORD_M,
) -> float:
    """Pressure lift coefficient from surface node pressures.

    Each segment's normal is taken from its geometry (exactly perpendicular,
    with the segment's length) and oriented to agree with the outward normals
    stored at its two end nodes, so the result doesn't depend on which way the
    surface loop happens to be walked.
    """
    xy = np.asarray(xy, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    a, b = segments[:, 0], segments[:, 1]
    tangent = xy[b] - xy[a]
    perpendicular = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)  # |perpendicular| = segment length
    outward = np.asarray(normals, dtype=np.float64)[a] + np.asarray(normals, dtype=np.float64)[b]
    perpendicular *= np.sign(np.einsum("ij,ij->i", perpendicular, outward))[:, None]
    force = -(0.5 * (p[a] + p[b]))[:, None] * perpendicular
    alpha = math.radians(aoa_deg)
    lift_direction = np.array([-math.sin(alpha), math.cos(alpha)])
    return float(force.sum(axis=0) @ lift_direction / (0.5 * u_inf**2 * chord))
