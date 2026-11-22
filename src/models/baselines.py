"""Two baselines that need no training, so every model has a floor to beat.

- **Free stream:** the flow as if the airfoil weren't there. The inlet velocity
  everywhere, zero gauge pressure, zero turbulent viscosity. Its relative L2
  error on p and ν_t is exactly 1 by construction, and it predicts no lift.
- **Thin-airfoil theory:** C_L = 2πα (α in radians), the classic result for a
  thin, uncambered airfoil. It predicts lift only. Because lift depends mostly
  on the angle of attack, it already ranks the test airfoils' lift well, and a
  model's lift ranking only means something next to it.
"""

from __future__ import annotations

import math

import numpy as np

from src.dataset.airfrans import TARGETS, Simulation
from src.evaluate.evaluate import Prediction


def free_stream(sim: Simulation) -> Prediction:
    fields = np.zeros((sim.n_nodes, len(TARGETS)), dtype=np.float64)
    fields[:, 0] = sim.inlet_u_x
    fields[:, 1] = sim.inlet_u_y
    return Prediction(fields=fields)


def thin_airfoil(sim: Simulation) -> Prediction:
    return Prediction(fields=None, lift=2 * math.pi * math.radians(sim.aoa_deg))


BASELINES = {"free_stream": free_stream, "thin_airfoil": thin_airfoil}
