"""Folder names → simulation parameters."""

from __future__ import annotations

import math

import pytest

from src.etl.names import NU_M2_S, SimulationNameError, parse_simulation_name


def test_a_four_digit_name() -> None:
    params = parse_simulation_name("airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32")
    assert params.u_inf == 31.283
    assert params.aoa_deg == -4.156
    assert params.naca_params == (0.919, 6.98, 14.32)
    assert params.naca_digits == 4


def test_a_five_digit_name() -> None:
    params = parse_simulation_name("airFoil2D_SST_36.622_11.319_3.941_5.424_1.0_16.283")
    assert params.naca_params == (3.941, 5.424, 1.0, 16.283)
    assert params.naca_digits == 5


def test_reynolds_and_inlet_velocity_follow_from_speed_and_angle() -> None:
    params = parse_simulation_name("airFoil2D_SST_31.2_30.0_2.0_4.0_12.0")
    assert params.reynolds == pytest.approx(31.2 / NU_M2_S)
    assert params.reynolds == pytest.approx(2.0e6)
    assert params.inlet_u_x == pytest.approx(31.2 * math.sqrt(3) / 2)
    assert params.inlet_u_y == pytest.approx(31.2 / 2)


@pytest.mark.parametrize(
    "name",
    [
        "airFoil3D_SST_31.2_3.0_2.0_4.0_12.0",  # wrong prefix
        "airFoil2D_SST_31.2_3.0_2.0_4.0",  # only 2 NACA parameters
        "airFoil2D_SST_31.2_3.0_2.0_4.0_12.0_1.0_9.0",  # 5 NACA parameters
        "airFoil2D_SST_31.2_nan_2.0_4.0_12.0",  # not a plain decimal
        "airFoil2D_SST_3e1_3.0_2.0_4.0_12.0",  # exponent
        "airFoil2D_SST_0.0_3.0_2.0_4.0_12.0",  # no flow
        "airFoil2D_SST__3.0_2.0_4.0_12.0",  # empty field
    ],
)
def test_malformed_names_are_rejected(name: str) -> None:
    with pytest.raises(SimulationNameError):
        parse_simulation_name(name)
