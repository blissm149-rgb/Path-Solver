"""Tests for optimizer.py — Phase 5 validation."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from planner.cost_landscape import CostLandscape, CostLandscapeConfig, RBFSource
from planner.vehicle_model import VehicleModel, PlanningProblem
from planner.graph_search import resample_path
from planner.optimizer import (
    generate_candidate_plans, _compute_epsilon, _unpack, _wrap_pi
)


def make_straight_seed(x0, y0, x1, y1, N):
    """Create a straight-line seed path of N points."""
    xs = np.linspace(x0, x1, N)
    ys = np.linspace(y0, y1, N)
    return np.column_stack([xs, ys])


def make_problem(x0=10.0, y0=10.0, theta0=np.pi/4, v0=5.0,
                 xg=90.0, yg=90.0, r=5.0, N=15):
    """Create a standard planning problem."""
    v = VehicleModel()
    return PlanningProblem(
        vehicle=v,
        x_init=x0, y_init=y0,
        theta_init=theta0, v_init=v0,
        x_goal=xg, y_goal=yg,
        goal_radius=r,
        n_waypoints=N,
    )


class TestEpsilonComputation:
    def test_auto_epsilon(self):
        """Auto epsilon is positive and proportional to 1/N."""
        prob = make_problem(N=20)
        eps = _compute_epsilon(prob)
        straight = np.sqrt((90-10)**2 + (90-10)**2)
        expected = 0.1 * straight / 20
        assert abs(eps - expected) < 1e-10

    def test_manual_epsilon(self):
        """Non-zero progress_epsilon overrides auto."""
        v = VehicleModel()
        prob = PlanningProblem(
            vehicle=v, x_init=0, y_init=0, theta_init=0, v_init=1,
            x_goal=50, y_goal=50, goal_radius=3, n_waypoints=10,
            progress_epsilon=2.5,
        )
        assert _compute_epsilon(prob) == 2.5


class TestWrapPi:
    def test_wrap_pi(self):
        assert abs(_wrap_pi(np.array([np.pi + 0.1])) - (-np.pi + 0.1)) < 1e-10
        assert abs(_wrap_pi(np.array([-np.pi - 0.1])) - (np.pi - 0.1)) < 1e-10
        assert abs(_wrap_pi(np.array([0.0])) - 0.0) < 1e-10


class TestOptimizerOnUniformField:
    """Test optimizer on a uniform (background-only) cost field."""

    def test_straight_line_seed_stays_feasible(self):
        """On a uniform field, a straight-line seed optimizes to a feasible plan."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)

        N = 15
        prob = make_problem(N=N)
        seed = make_straight_seed(10, 10, 90, 90, N)

        candidates = generate_candidate_plans(landscape, prob, [seed], maxiter=200)
        assert len(candidates) == 1

        plan = candidates[0]
        assert plan['label'] == 'Plan A'
        assert plan['feasible'], "Straight-line on uniform field should be feasible"

    def test_dist_to_goal_within_radius(self):
        """Feasible plan ends within goal radius."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)

        N = 15
        prob = make_problem(N=N)
        seed = make_straight_seed(10, 10, 90, 90, N)

        candidates = generate_candidate_plans(landscape, prob, [seed], maxiter=200)
        plan = candidates[0]
        assert plan['dist_to_goal'] <= prob.goal_radius + 1.0

    def test_multiple_seeds_sorted_by_cost(self):
        """Multiple candidates are returned sorted (feasible first, cost ascending)."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)

        N = 12
        prob = make_problem(N=N)
        seed1 = make_straight_seed(10, 10, 90, 90, N)
        # Same seed duplicated — should produce similar plans
        seed2 = make_straight_seed(10, 10, 90, 90, N) + np.array([0, 2])

        candidates = generate_candidate_plans(landscape, prob, [seed1, seed2], maxiter=150)
        assert len(candidates) == 2
        assert candidates[0]['label'] == 'Plan A'
        assert candidates[1]['label'] == 'Plan B'

        # Feasible plans come before infeasible
        for i in range(len(candidates) - 1):
            if not candidates[i]['feasible'] and candidates[i+1]['feasible']:
                pytest.fail("Feasible plan ranked after infeasible plan")

    def test_speed_bounds_satisfied(self):
        """All waypoint speeds in [v_min, v_max]."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)

        N = 12
        prob = make_problem(N=N)
        seed = make_straight_seed(10, 10, 90, 90, N)

        candidates = generate_candidate_plans(landscape, prob, [seed], maxiter=200)
        plan = candidates[0]
        v = plan['v']
        assert np.all(v >= prob.vehicle.v_min - 0.1)
        assert np.all(v <= prob.vehicle.v_max + 0.1)


class TestOptimizerWithObstacle:
    """Test optimizer deviates around obstacles."""

    def test_deviates_around_obstacle(self):
        """Path should not pass through a mid-path obstacle."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        # Place obstacle in middle of straight-line path
        landscape._obstacles = [RBFSource(
            cx=50.0, cy=50.0, sigma_x=5.0, sigma_y=5.0,
            amplitude=1e4, theta=0.0
        )]

        N = 15
        prob = make_problem(N=N)
        # Slightly offset seed to help optimizer find a way around
        xs = np.linspace(10, 90, N)
        ys = np.linspace(10, 90, N)
        # Add lateral offset to give optimizer a hint
        offset = np.linspace(0, 5, N)
        seed = np.column_stack([xs, ys + offset])

        candidates = generate_candidate_plans(landscape, prob, [seed], maxiter=150)
        assert len(candidates) == 1

        plan = candidates[0]
        x, y = plan['x'], plan['y']
        # Check no waypoint sits directly in the obstacle
        for xi, yi in zip(x, y):
            dist = np.sqrt((xi - 50)**2 + (yi - 50)**2)
            assert dist > 5.0, f"Waypoint at ({xi:.1f},{yi:.1f}) too close to obstacle center"

    def test_forward_progress_non_increasing_distance(self):
        """Distance-to-goal sequence is non-increasing within ε tolerance."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)

        N = 12
        prob = make_problem(N=N)
        seed = make_straight_seed(10, 10, 90, 90, N)

        candidates = generate_candidate_plans(landscape, prob, [seed], maxiter=200)
        plan = candidates[0]

        x, y = plan['x'], plan['y']
        d = np.sqrt((x - prob.x_goal)**2 + (y - prob.y_goal)**2)
        eps = _compute_epsilon(prob)

        violations = np.sum(d[1:] > d[:-1] + eps + 0.5)  # generous tolerance
        assert violations == 0, f"Forward progress violated at {violations} steps"


class TestPlanMetrics:
    """Test metric fields in returned candidates."""

    def test_metric_fields_present(self):
        """All required fields are present in candidate dict."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        N = 10
        prob = make_problem(N=N)
        seed = make_straight_seed(10, 10, 90, 90, N)
        candidates = generate_candidate_plans(landscape, prob, [seed], maxiter=100)
        required = {'label', 'x', 'y', 'theta', 'v', 'integrated_cost',
                    'peak_cost', 'total_length', 'total_time', 'feasible',
                    'dist_to_goal', 'seed_index'}
        assert required.issubset(set(candidates[0].keys()))

    def test_total_length_positive(self):
        """total_length is positive."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        N = 10
        prob = make_problem(N=N)
        seed = make_straight_seed(10, 10, 90, 90, N)
        candidates = generate_candidate_plans(landscape, prob, [seed], maxiter=100)
        assert candidates[0]['total_length'] > 0

    def test_total_time_positive(self):
        """total_time is positive."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        N = 10
        prob = make_problem(N=N)
        seed = make_straight_seed(10, 10, 90, 90, N)
        candidates = generate_candidate_plans(landscape, prob, [seed], maxiter=100)
        assert candidates[0]['total_time'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
