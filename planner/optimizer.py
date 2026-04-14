"""
optimizer.py — Constrained NLP path optimizer using scipy SLSQP.

Decision variables: z = [x_0, y_0, θ_0, v_0,  x_1, y_1, θ_1, v_1, ..., x_{N-1}, y_{N-1}, θ_{N-1}, v_{N-1}]
Total: 4N variables.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Optional

from .cost_landscape import CostLandscape
from .vehicle_model import VehicleModel, PlanningProblem

# Objective weights
W_HEADING = 10.0
W_SMOOTH = 2.0
W_SPEED = 0.5


def _unpack(z: np.ndarray, N: int):
    """Unpack flat decision vector into (x, y, theta, v) arrays of length N."""
    z = z.reshape(N, 4)
    return z[:, 0], z[:, 1], z[:, 2], z[:, 3]


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _build_initial_z(
    seed_path: np.ndarray,
    N: int,
    v_init: float,
    v_max: float,
) -> np.ndarray:
    """Build initial decision vector from resampled seed path."""
    x = seed_path[:, 0]
    y = seed_path[:, 1]

    # Compute headings from consecutive differences
    dx = np.diff(x)
    dy = np.diff(y)
    theta = np.zeros(N)
    theta[:-1] = np.arctan2(dy, dx)
    theta[-1] = theta[-2]  # replicate last

    # Uniform speed initialization
    v = np.full(N, 0.6 * v_max)
    v[0] = v_init

    z = np.column_stack([x, y, theta, v]).ravel()
    return z


def _objective(z: np.ndarray, N: int, landscape: CostLandscape,
                vehicle: VehicleModel) -> float:
    """NLP objective: integrated footprint cost + heading/smoothness/speed penalties."""
    x, y, theta, v = _unpack(z, N)
    hl, hw = vehicle.half_length, vehicle.half_width

    total = 0.0
    for i in range(N - 1):
        # Segment length
        ds = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
        # Footprint cost (mean)
        mean_c, _ = landscape.evaluate_footprint(x[i], y[i], theta[i], hl, hw)
        total += mean_c * ds

    # Heading consistency: align theta with path direction
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    ref_heading = np.arctan2(dy, dx)
    heading_err = _wrap_pi(theta[:-1] - ref_heading)
    total += W_HEADING * float(np.sum(heading_err ** 2))

    # Heading smoothness
    dtheta = _wrap_pi(theta[1:] - theta[:-1])
    total += W_SMOOTH * float(np.sum(dtheta ** 2))

    # Speed smoothness
    dv = v[1:] - v[:-1]
    total += W_SPEED * float(np.sum(dv ** 2))

    return total


def _make_constraints(
    N: int,
    problem: PlanningProblem,
    landscape: CostLandscape,
    epsilon: float,
) -> list[dict]:
    """Build scipy constraint dicts for SLSQP."""
    vehicle = problem.vehicle
    hl, hw = vehicle.half_length, vehicle.half_width
    cfg = landscape.config

    constraints = []

    # --- Equality: initial state pinned ---
    def eq_init(z):
        x, y, theta, v = _unpack(z, N)
        return np.array([
            x[0] - problem.x_init,
            y[0] - problem.y_init,
            _wrap_pi(np.array([theta[0] - problem.theta_init]))[0],
            v[0] - problem.v_init,
        ])

    constraints.append({'type': 'eq', 'fun': eq_init})

    # --- Inequality: terminal in goal disk ---
    def ineq_goal(z):
        x, y, _, _ = _unpack(z, N)
        dist2 = (x[-1] - problem.x_goal)**2 + (y[-1] - problem.y_goal)**2
        return np.array([problem.goal_radius**2 - dist2])

    constraints.append({'type': 'ineq', 'fun': ineq_goal})

    # --- Inequality: forward progress ---
    def ineq_progress(z):
        x, y, _, _ = _unpack(z, N)
        d = np.sqrt((x - problem.x_goal)**2 + (y - problem.y_goal)**2)
        return d[:-1] + epsilon - d[1:]

    constraints.append({'type': 'ineq', 'fun': ineq_progress})

    # --- Inequality: speed bounds ---
    def ineq_v_min(z):
        _, _, _, v = _unpack(z, N)
        return v - vehicle.v_min

    def ineq_v_max(z):
        _, _, _, v = _unpack(z, N)
        return vehicle.v_max - v

    constraints.append({'type': 'ineq', 'fun': ineq_v_min})
    constraints.append({'type': 'ineq', 'fun': ineq_v_max})

    # --- Inequality: longitudinal acceleration bounds ---
    def ineq_accel(z):
        x, y, _, v = _unpack(z, N)
        ds = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        ds = np.maximum(ds, 0.01)  # avoid division by zero
        dv = v[1:] - v[:-1]
        accel = dv / ds  # approximate a ≈ Δv/Δs (not exact but standard for path planning)
        # Properly: a = v * Δv / Δs (kinematic)
        v_avg = 0.5 * (v[:-1] + v[1:])
        a_long = v_avg * dv / ds
        return np.concatenate([
            vehicle.a_max - a_long,
            vehicle.a_max + a_long,
        ])

    constraints.append({'type': 'ineq', 'fun': ineq_accel})

    # --- Inequality: curvature bounds ---
    def ineq_curvature(z):
        x, y, theta, _ = _unpack(z, N)
        ds = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        ds = np.maximum(ds, 0.01)
        dtheta = _wrap_pi(theta[1:] - theta[:-1])
        kappa = np.abs(dtheta / ds)
        return vehicle.kappa_max - kappa

    constraints.append({'type': 'ineq', 'fun': ineq_curvature})

    # --- Inequality: lateral acceleration bounds ---
    def ineq_lat_accel(z):
        x, y, theta, v = _unpack(z, N)
        ds = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        ds = np.maximum(ds, 0.01)
        dtheta = _wrap_pi(theta[1:] - theta[:-1])
        kappa = np.abs(dtheta / ds)
        v_avg = 0.5 * (v[:-1] + v[1:])
        a_lat = v_avg**2 * kappa
        return vehicle.a_lat_max - a_lat

    constraints.append({'type': 'ineq', 'fun': ineq_lat_accel})

    # --- Inequality: obstacle clearance ---
    def ineq_obstacle(z):
        x, y, theta, _ = _unpack(z, N)
        vals = []
        for i in range(N):
            _, max_c = landscape.evaluate_footprint(x[i], y[i], theta[i], hl, hw)
            vals.append(problem.obstacle_cost_threshold - max_c)
        return np.array(vals)

    constraints.append({'type': 'ineq', 'fun': ineq_obstacle})

    # --- Inequality: domain bounds (OBB must fit) ---
    margin = max(hl, hw)

    def ineq_domain(z):
        x, y, _, _ = _unpack(z, N)
        return np.concatenate([
            x - (cfg.x_min + margin),
            (cfg.x_max - margin) - x,
            y - (cfg.y_min + margin),
            (cfg.y_max - margin) - y,
        ])

    constraints.append({'type': 'ineq', 'fun': ineq_domain})

    return constraints


def _compute_epsilon(problem: PlanningProblem) -> float:
    """Compute forward progress slack epsilon."""
    if problem.progress_epsilon != 0.0:
        return problem.progress_epsilon
    straight_line_dist = np.sqrt(
        (problem.x_goal - problem.x_init)**2 + (problem.y_goal - problem.y_init)**2
    )
    return 0.1 * straight_line_dist / problem.n_waypoints


def _check_feasibility(
    z: np.ndarray,
    N: int,
    problem: PlanningProblem,
    landscape: CostLandscape,
    epsilon: float,
    tol: float = 1e-3,
) -> bool:
    """Check all hard constraints are satisfied within tolerance."""
    x, y, theta, v = _unpack(z, N)
    vehicle = problem.vehicle
    hl, hw = vehicle.half_length, vehicle.half_width
    cfg = landscape.config

    # Goal disk
    dist_goal = np.sqrt((x[-1] - problem.x_goal)**2 + (y[-1] - problem.y_goal)**2)
    if dist_goal > problem.goal_radius + tol:
        return False

    # Speed bounds
    if np.any(v < vehicle.v_min - tol) or np.any(v > vehicle.v_max + tol):
        return False

    # Forward progress
    d = np.sqrt((x - problem.x_goal)**2 + (y - problem.y_goal)**2)
    if np.any(d[:-1] + epsilon + tol < d[1:]):
        return False

    # Accel / curvature / lateral
    ds = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    ds = np.maximum(ds, 0.01)
    dtheta = _wrap_pi(theta[1:] - theta[:-1])
    kappa = np.abs(dtheta / ds)

    if np.any(kappa > vehicle.kappa_max + tol):
        return False

    v_avg = 0.5 * (v[:-1] + v[1:])
    a_lat = v_avg**2 * kappa
    if np.any(a_lat > vehicle.a_lat_max + tol):
        return False

    dv = v[1:] - v[:-1]
    a_long = v_avg * dv / ds
    if np.any(np.abs(a_long) > vehicle.a_max + tol):
        return False

    # Obstacle clearance
    for i in range(N):
        _, max_c = landscape.evaluate_footprint(x[i], y[i], theta[i], hl, hw)
        if max_c > problem.obstacle_cost_threshold + tol:
            return False

    # Domain bounds
    margin = max(hl, hw)
    if np.any(x < cfg.x_min + margin - tol) or np.any(x > cfg.x_max - margin + tol):
        return False
    if np.any(y < cfg.y_min + margin - tol) or np.any(y > cfg.y_max - margin + tol):
        return False

    return True


def _compute_plan_metrics(
    z: np.ndarray,
    N: int,
    landscape: CostLandscape,
    vehicle: VehicleModel,
    problem: PlanningProblem,
) -> dict:
    """Compute scalar metrics for a solved plan."""
    x, y, theta, v = _unpack(z, N)
    hl, hw = vehicle.half_length, vehicle.half_width

    ds_arr = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    total_length = float(ds_arr.sum())

    integrated_cost = 0.0
    peak_cost = 0.0
    for i in range(N - 1):
        mean_c, max_c = landscape.evaluate_footprint(x[i], y[i], theta[i], hl, hw)
        integrated_cost += mean_c * ds_arr[i]
        peak_cost = max(peak_cost, max_c)

    # Also check last waypoint
    _, last_max = landscape.evaluate_footprint(x[-1], y[-1], theta[-1], hl, hw)
    peak_cost = max(peak_cost, last_max)

    # Time: sum of Δs / v_avg for each segment
    v_avg = 0.5 * (v[:-1] + v[1:])
    v_avg = np.maximum(v_avg, 0.01)
    total_time = float(np.sum(ds_arr / v_avg))

    dist_to_goal = float(np.sqrt((x[-1] - problem.x_goal)**2 + (y[-1] - problem.y_goal)**2))

    return {
        'x': x.copy(),
        'y': y.copy(),
        'theta': theta.copy(),
        'v': v.copy(),
        'integrated_cost': integrated_cost,
        'peak_cost': peak_cost,
        'total_length': total_length,
        'total_time': total_time,
        'dist_to_goal': dist_to_goal,
    }


def generate_candidate_plans(
    landscape: CostLandscape,
    problem: PlanningProblem,
    seed_paths: list[np.ndarray],
    maxiter: int = 100,
) -> list[dict]:
    """
    Optimize each seed path, rank by (feasibility desc, integrated_cost asc),
    label as Plan A, B, C, ...

    Args:
        landscape: Cost field
        problem: Planning problem specification
        seed_paths: List of (N, 2) resampled seed paths
        maxiter: Max SLSQP iterations

    Returns:
        Sorted list of candidate dicts.
    """
    vehicle = problem.vehicle
    N = problem.n_waypoints
    epsilon = _compute_epsilon(problem)

    constraints = _make_constraints(N, problem, landscape, epsilon)

    candidates = []

    for seed_idx, seed_path in enumerate(seed_paths):
        z0 = _build_initial_z(seed_path, N, problem.v_init, vehicle.v_max)

        result = minimize(
            _objective,
            z0,
            args=(N, landscape, vehicle),
            method='SLSQP',
            constraints=constraints,
            options={
                'maxiter': maxiter,
                'ftol': 1e-6,
                'disp': False,
            },
        )

        z_sol = result.x
        feasible = _check_feasibility(z_sol, N, problem, landscape, epsilon)

        metrics = _compute_plan_metrics(z_sol, N, landscape, vehicle, problem)
        metrics['feasible'] = feasible
        metrics['seed_index'] = seed_idx

        candidates.append(metrics)

    # Sort: feasible first, then by integrated cost
    candidates.sort(key=lambda c: (not c['feasible'], c['integrated_cost']))

    # Label
    for i, cand in enumerate(candidates):
        cand['label'] = f"Plan {chr(ord('A') + i)}"

    return candidates
