"""Field metrics by hand, and the lift integral against closed-form answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluate.metrics import cp_rmse, lift_coefficient, relative_l2, standardised_mse


def test_relative_l2_by_hand() -> None:
    true = np.array([3.0, 4.0])  # norm 5
    assert relative_l2(true, true) == 0.0
    assert relative_l2(np.zeros(2), true) == 1.0
    assert relative_l2(np.array([3.0, 7.0]), true) == pytest.approx(3 / 5)
    assert math.isnan(relative_l2(np.ones(2), np.zeros(2)))


def test_standardised_mse_and_cp_rmse_by_hand() -> None:
    assert standardised_mse(np.array([1.0, 3.0]), np.array([0.0, 0.0]), std=2.0) == pytest.approx((0.25 + 2.25) / 2)
    # U∞ = 10 → ½U∞² = 50; errors of 50 and −50 in p are Cp errors of ±1.
    assert cp_rmse(np.array([50.0, -50.0]), np.zeros(2), u_inf=10.0) == pytest.approx(1.0)


def polygon(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counter-clockwise vertices → (xy, outward vertex normals, segments)."""
    n = len(vertices)
    segments = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    edge = vertices[segments[:, 1]] - vertices[segments[:, 0]]
    edge_normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)  # outward for a counter-clockwise loop
    edge_normal /= np.linalg.norm(edge_normal, axis=1, keepdims=True)
    vertex_normal = edge_normal + np.roll(edge_normal, 1, axis=0)
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertices, vertex_normal, segments


def area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
CIRCLE = np.stack([np.cos(2 * np.pi * np.arange(64) / 64), np.sin(2 * np.pi * np.arange(64) / 64)], axis=1) * 0.3


@pytest.mark.parametrize("shape", [SQUARE, CIRCLE], ids=["square", "circle"])
@pytest.mark.parametrize("aoa_deg", [-5.0, 0.0, 12.0])
@pytest.mark.parametrize("a, b", [(0.0, 0.0), (2.0, 0.0), (0.0, -3.0), (1.5, 4.0)])
def test_linear_pressure_gives_the_divergence_theorem_force(shape: np.ndarray, aoa_deg: float, a: float, b: float) -> None:
    # −∮ p n dl = −∫∫ ∇p dA = −(a, b) · area, exactly for a polygon and a linear p.
    xy, normals, segments = polygon(shape)
    p = a * xy[:, 0] + b * xy[:, 1] + 7.0  # a constant adds no force
    u_inf = 20.0
    alpha = math.radians(aoa_deg)
    expected = -np.array([a, b]) * area(shape) @ np.array([-math.sin(alpha), math.cos(alpha)]) / (0.5 * u_inf**2)
    got = lift_coefficient(xy, p, normals, segments, aoa_deg=aoa_deg, u_inf=u_inf)
    assert got == pytest.approx(expected, abs=1e-12)


def test_the_result_does_not_depend_on_which_way_the_loop_is_walked() -> None:
    xy, normals, segments = polygon(CIRCLE)
    p = 3.0 * xy[:, 1] - xy[:, 0]
    forward = lift_coefficient(xy, p, normals, segments, aoa_deg=4.0, u_inf=30.0)
    backward = lift_coefficient(xy, p, normals, segments[::-1, ::-1], aoa_deg=4.0, u_inf=30.0)
    assert forward == pytest.approx(backward, abs=1e-14)


def test_inward_normals_flip_the_sign() -> None:
    # The convention check: the stored-file normals pointed inward and would have flipped every C_L.
    xy, normals, segments = polygon(SQUARE)
    p = -2.0 * xy[:, 1]  # suction on top: lifts
    outward = lift_coefficient(xy, p, normals, segments, aoa_deg=0.0, u_inf=10.0)
    inward = lift_coefficient(xy, p, -normals, segments, aoa_deg=0.0, u_inf=10.0)
    assert outward > 0 and inward == pytest.approx(-outward)
