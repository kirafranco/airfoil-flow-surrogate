"""Simulation parameters from an AirfRANS folder name.

Every simulation directory is named

    airFoil2D_SST_<U∞ in m/s>_<angle of attack in degrees>_<NACA parameters>

with 3 NACA parameters for a 4-digit section (camber, camber position,
thickness) or 4 for a 5-digit one (design lift, camber position, reflex,
thickness). The Reynolds number isn't in the name. It follows from the
dataset's setup: a 1 m chord in air with kinematic viscosity 1.56e-5 m²/s,
which puts every simulation between 2 and 6 million, as the AirfRANS paper
states.

The quality gate cross-checks these parsed values against the physics in the
files, so a mislabelled simulation is caught rather than trusted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

PREFIX = "airFoil2D_SST_"
CHORD_M = 1.0
NU_M2_S = 1.56e-5

#: Plain decimals only: "nan", "inf" and exponents are not how the dataset writes numbers.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_NACA_PARAM_COUNTS = {3: 4, 4: 5}  # parameters in the name → NACA digits


class SimulationNameError(ValueError):
    """A directory name that doesn't follow the AirfRANS pattern."""


@dataclass(frozen=True)
class SimulationParams:
    name: str
    u_inf: float
    aoa_deg: float
    naca_params: tuple[float, ...]

    @property
    def naca_digits(self) -> int:
        return _NACA_PARAM_COUNTS[len(self.naca_params)]

    @property
    def reynolds(self) -> float:
        return self.u_inf * CHORD_M / NU_M2_S

    @property
    def inlet_u_x(self) -> float:
        return self.u_inf * math.cos(math.radians(self.aoa_deg))

    @property
    def inlet_u_y(self) -> float:
        return self.u_inf * math.sin(math.radians(self.aoa_deg))


def parse_simulation_name(name: str) -> SimulationParams:
    if not name.startswith(PREFIX):
        raise SimulationNameError(f"{name!r}: does not start with {PREFIX!r}")
    tokens = name[len(PREFIX) :].split("_")
    if len(tokens) - 2 not in _NACA_PARAM_COUNTS:
        raise SimulationNameError(
            f"{name!r}: expected U∞, angle of attack and 3 or 4 NACA parameters, got {len(tokens)} values"
        )
    bad = [token for token in tokens if not _NUMBER.fullmatch(token)]
    if bad:
        raise SimulationNameError(f"{name!r}: not a plain decimal: {bad[0]!r}")
    u_inf, aoa_deg, *naca = (float(token) for token in tokens)
    if u_inf <= 0:
        raise SimulationNameError(f"{name!r}: free-stream speed must be positive, got {u_inf}")
    return SimulationParams(name=name, u_inf=u_inf, aoa_deg=aoa_deg, naca_params=tuple(naca))
