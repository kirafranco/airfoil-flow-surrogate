"""The GPU is reachable from inside the ml container, and the stack runs on it.

Marked `gpu`. These tests fail rather than skip without a CUDA device: a GPU
that silently disappeared from the container would otherwise turn a training
run into a CPU run that takes days.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.nn import GraphSAGE

pytestmark = pytest.mark.gpu

#: The VRAM later features plan against: a 6 GB card, ~1 GB of it held by Windows.
MIN_TOTAL_VRAM_GIB = 4.0

#: One AirfRANS-sized mesh through a 4-layer GraphSAGE, forward and backward.
#: Measured at 0.94 GiB on 2026-11-20; the ceiling leaves room for batching.
MAX_SAGE_PEAK_GIB = 2.0
MESH_NODES = 250_000
MESH_EDGES = 1_500_000


def test_cuda_is_available() -> None:
    assert torch.cuda.is_available(), (
        "no CUDA device: run inside the ml container, which reserves the GPU in servicios/docker-compose.yml"
    )


def test_the_device_has_the_vram_later_features_plan_for() -> None:
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 2**30
    print(f"device={props.name!r} total_vram_gib={total_gib:.2f} torch_cuda={torch.version.cuda}")
    assert total_gib >= MIN_TOTAL_VRAM_GIB


def test_graphsage_trains_one_mesh_sized_graph_on_the_gpu() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(MESH_NODES, 7, device=device)
    edge_index = torch.randint(0, MESH_NODES, (2, MESH_EDGES), device=device)
    model = GraphSAGE(7, 64, num_layers=4, out_channels=4).to(device)

    torch.cuda.reset_peak_memory_stats()
    out = model(x, edge_index)
    out.pow(2).mean().backward()
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(f"graphsage nodes={MESH_NODES} edges={MESH_EDGES} peak_gib={peak_gib:.3f}")

    assert out.device.type == "cuda"
    assert out.shape == (MESH_NODES, 4)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert peak_gib <= MAX_SAGE_PEAK_GIB
