"""
vehicle_model.py — Vehicle kinematics and planning problem definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VehicleModel:
    length: float = 5.0        # OBB total length [m]
    width: float = 2.5         # OBB total width [m]
    v_max: float = 15.0        # max speed [m/s]
    v_min: float = 0.5         # min speed [m/s]
    a_max: float = 3.0         # max longitudinal accel [m/s²]
    a_lat_max: float = 2.0     # max lateral accel [m/s²]
    kappa_max: float = 0.15    # max curvature [1/m] → min turn radius ≈ 6.7m

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0


@dataclass
class PlanningProblem:
    vehicle: VehicleModel

    # Initial state (all pinned as equality constraints)
    x_init: float
    y_init: float
    theta_init: float
    v_init: float

    # Terminal constraint (inequality — arrival disk)
    x_goal: float
    y_goal: float
    goal_radius: float

    n_waypoints: int = 20
    obstacle_cost_threshold: float = 100.0
    progress_epsilon: float = 0.0   # forward progress slack [m]
                                     # 0.0 means auto-set to 0.1 × straight-line / n_waypoints
