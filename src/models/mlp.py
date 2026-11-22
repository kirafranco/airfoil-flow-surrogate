"""A per-node MLP: each mesh node's 7 features in, its 4 targets out.

It sees one node at a time, so it knows a node's position, its distance to the
airfoil, the inflow and (on the surface) the outward normal, but nothing about
the node's neighbours. The graph network that follows adds exactly that.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from src.dataset.airfrans import FEATURES, TARGETS


@dataclass(frozen=True)
class MlpConfig:
    hidden: int = 256
    layers: int = 4  # hidden layers
    in_features: int = len(FEATURES)
    out_features: int = len(TARGETS)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class MLP(nn.Module):
    def __init__(self, config: MlpConfig) -> None:
        super().__init__()
        self.config = config
        modules: list[nn.Module] = [nn.Linear(config.in_features, config.hidden), nn.ReLU()]
        for _ in range(config.layers - 1):
            modules += [nn.Linear(config.hidden, config.hidden), nn.ReLU()]
        modules.append(nn.Linear(config.hidden, config.out_features))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
