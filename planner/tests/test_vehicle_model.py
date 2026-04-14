"""Tests for vehicle_model.py — Phase 2 validation."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from planner.vehicle_model import VehicleModel, PlanningProblem


class TestVehicleModel:
    """Test VehicleModel data and properties."""

    def test_default_values(self):
        v = VehicleModel()
        assert v.length == 5.0
        assert v.width == 2.5
        assert v.v_max == 15.0
        assert v.v_min == 0.5
        assert v.a_max == 3.0
        assert v.a_lat_max == 2.0
        assert v.kappa_max == 0.15

    def test_half_length(self):
        v = VehicleModel(length=10.0)
        assert v.half_length == 5.0

    def test_half_width(self):
        v = VehicleModel(width=4.0)
        assert v.half_width == 2.0

    def test_obb_corners_at_identity(self):
        """OBB corners at theta=0 are axis-aligned."""
        v = VehicleModel(length=6.0, width=3.0)
        hl, hw = v.half_length, v.half_width
        # Center at origin, theta=0 — corners at (±hl, ±hw)
        corners = np.array([
            [ hl,  hw],
            [ hl, -hw],
            [-hl,  hw],
            [-hl, -hw],
        ])
        assert corners.shape == (4, 2)
        assert abs(corners[0, 0] - 3.0) < 1e-10
        assert abs(corners[0, 1] - 1.5) < 1e-10

    def test_obb_corners_rotated_90(self):
        """OBB corners at theta=pi/2 are transposed."""
        v = VehicleModel(length=6.0, width=3.0)
        hl, hw = v.half_length, v.half_width
        theta = np.pi / 2
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        local_corner = np.array([hl, hw])
        world_corner = R @ local_corner
        # After 90° rotation, [hl, hw] → [-hw, hl]
        assert abs(world_corner[0] - (-hw)) < 1e-10
        assert abs(world_corner[1] - hl) < 1e-10

    def test_custom_vehicle(self):
        v = VehicleModel(length=8.0, width=3.0, v_max=20.0, kappa_max=0.1)
        assert v.half_length == 4.0
        assert v.half_width == 1.5
        assert v.v_max == 20.0
        assert v.kappa_max == 0.1


class TestPlanningProblem:
    """Test PlanningProblem construction."""

    def test_construction(self):
        v = VehicleModel()
        p = PlanningProblem(
            vehicle=v,
            x_init=10.0, y_init=10.0,
            theta_init=np.pi / 4, v_init=5.0,
            x_goal=90.0, y_goal=90.0,
            goal_radius=5.0,
        )
        assert p.n_waypoints == 20
        assert p.obstacle_cost_threshold == 100.0
        assert p.progress_epsilon == 0.0

    def test_default_n_waypoints(self):
        v = VehicleModel()
        p = PlanningProblem(
            vehicle=v, x_init=0, y_init=0,
            theta_init=0, v_init=1,
            x_goal=50, y_goal=50, goal_radius=3.0,
        )
        assert p.n_waypoints == 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
