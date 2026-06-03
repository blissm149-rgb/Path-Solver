"""
optimizer.py - Constrained NLP path optimizer using scipy trust-constr.

Decision variables: z = [x_0, y_0, theta_0, v_0,  x_1, y_1, theta_1, v_1, ..., x_{N-1}, y_{N-1}, theta_{N-1}, v_{N-1}]
Total: 4N variables.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.sparse import csr_matrix, diags
from typing import Optional

from .cost_landscape import CostLandscape, _OBB_TEMPLATE_UNIT
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
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _field_gradient(
    landscape: CostLandscape,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic gradient of the cost field at query points.

    Returns (gx, gy) each shape (M,), where M = len(x_arr).
    Derivation: C = sum(A*exp(-Q)), dC/dx = -A*exp(-Q)*dQ/dx,
    dQ/dx = (u/sigma_x^2)*cos(theta) - (v/sigma_y^2)*sin(theta).
    """
    gx = np.zeros_like(x_arr, dtype=np.float64)
    gy = np.zeros_like(y_arr, dtype=np.float64)
    for src in landscape._sources + landscape._obstacles:
        dx = x_arr - src.cx
        dy = y_arr - src.cy
        c, s = np.cos(src.theta), np.sin(src.theta)
        u = c * dx + s * dy
        v = -s * dx + c * dy
        Q = 0.5 * ((u / src.sigma_x) ** 2 + (v / src.sigma_y) ** 2)
        e = src.amplitude * np.exp(-Q)
        gx -= e * ((u / src.sigma_x ** 2) * c - (v / src.sigma_y ** 2) * s)
        gy -= e * ((u / src.sigma_x ** 2) * s + (v / src.sigma_y ** 2) * c)
    return gx, gy


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
        ds = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
        mean_c, _ = landscape.evaluate_footprint(x[i], y[i], theta[i], hl, hw)
        total += mean_c * ds

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    ref_heading = np.arctan2(dy, dx)
    heading_err = _wrap_pi(theta[:-1] - ref_heading)
    total += W_HEADING * float(np.sum(heading_err ** 2))

    dtheta = _wrap_pi(theta[1:] - theta[:-1])
    total += W_SMOOTH * float(np.sum(dtheta ** 2))

    dv = v[1:] - v[:-1]
    total += W_SPEED * float(np.sum(dv ** 2))

    return total


def _objective_and_grad(
    z: np.ndarray,
    N: int,
    landscape: CostLandscape,
    vehicle: VehicleModel,
) -> tuple[float, np.ndarray]:
    """Return (J, dJ/dz) analytically. Eliminates finite-difference overhead.

    All four objective terms are differentiated analytically:
      J_cost    - arc-length-weighted integrated footprint cost
      J_heading - heading-to-chord alignment penalty
      J_smooth  - heading smoothness penalty
      J_speed   - speed smoothness penalty
    """
    x, y, theta, v = _unpack(z, N)
    hl, hw = vehicle.half_length, vehicle.half_width

    # Scaled OBB template: (5, 2) local footprint points
    local_pts = _OBB_TEMPLATE_UNIT * np.array([hl, hw])
    lx = local_pts[:, 0]  # (5,)
    ly = local_pts[:, 1]

    cos_t = np.cos(theta)  # (N,)
    sin_t = np.sin(theta)

    # World coords of all 5N footprint points: (N, 5)
    px = cos_t[:, None] * lx[None, :] - sin_t[:, None] * ly[None, :] + x[:, None]
    py = sin_t[:, None] * lx[None, :] + cos_t[:, None] * ly[None, :] + y[:, None]

    # Evaluate field value and gradient at all 5N points in one vectorized pass each
    costs_5N = landscape.evaluate(px.ravel(), py.ravel())
    gx_5N, gy_5N = _field_gradient(landscape, px.ravel(), py.ravel())

    costs = costs_5N.reshape(N, 5)
    gx_fp = gx_5N.reshape(N, 5)
    gy_fp = gy_5N.reshape(N, 5)

    mean_c = costs.mean(axis=1)   # (N,) - mean footprint cost per waypoint
    d_mx = gx_fp.mean(axis=1)    # (N,) - d(mean_c)/d(x_i)
    d_my = gy_fp.mean(axis=1)    # (N,) - d(mean_c)/d(y_i)

    # d(mean_c)/d(theta_i): chain rule through OBB rotation
    # dp_k/dtheta = dR/dtheta @ local_k,  dR/dtheta = [[-sin(theta), -cos(theta)], [cos(theta), -sin(theta)]]
    dp_x_dt = -sin_t[:, None] * lx[None, :] - cos_t[:, None] * ly[None, :]  # (N,5)
    dp_y_dt =  cos_t[:, None] * lx[None, :] - sin_t[:, None] * ly[None, :]
    d_mt = (gx_fp * dp_x_dt + gy_fp * dp_y_dt).mean(axis=1)  # (N,)

    # Segment quantities
    dx_seg = x[1:] - x[:-1]    # (N-1,)
    dy_seg = y[1:] - y[:-1]
    ds_seg = np.maximum(np.sqrt(dx_seg ** 2 + dy_seg ** 2), 1e-8)  # (N-1,)

    # --- J_cost ---
    J_cost = float((mean_c[:N-1] * ds_seg).sum())

    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    grad_t = np.zeros(N)
    grad_v = np.zeros(N)

    # Footprint cost gradient (waypoint i only; cross-terms for ds below)
    grad_x[:N-1] += d_mx[:N-1] * ds_seg
    grad_y[:N-1] += d_my[:N-1] * ds_seg
    grad_t[:N-1] += d_mt[:N-1] * ds_seg

    # Arc-length gradient: d(ds_i)/d(x_i) = -dx_i/ds_i, d(ds_i)/d(x_{i+1}) = +dx_i/ds_i
    inv_ds = 1.0 / ds_seg
    grad_x[:N-1] += mean_c[:N-1] * (-dx_seg * inv_ds)
    grad_x[1:]   += mean_c[:N-1] * ( dx_seg * inv_ds)
    grad_y[:N-1] += mean_c[:N-1] * (-dy_seg * inv_ds)
    grad_y[1:]   += mean_c[:N-1] * ( dy_seg * inv_ds)

    # --- J_heading ---
    ref_heading = np.arctan2(dy_seg, dx_seg)  # (N-1,)
    e_H = _wrap_pi(theta[:-1] - ref_heading)
    J_heading = W_HEADING * float(np.sum(e_H ** 2))

    # d(ref_i)/d(x_i)     = +dy_i/ds2_i  (arctan2 chain rule)
    # d(ref_i)/d(x_{i+1}) = -dy_i/ds2_i
    # d(ref_i)/d(y_i)     = -dx_i/ds2_i
    # d(ref_i)/d(y_{i+1}) = +dx_i/ds2_i
    # e_i = wrap_pi(theta_i - ref_i)  ->  d(e_i)/d(z) = -d(ref_i)/d(z)
    ds2_seg = np.maximum(dx_seg ** 2 + dy_seg ** 2, 1e-16)
    coeff_H = 2.0 * W_HEADING * e_H   # (N-1,)
    grad_t[:N-1] += coeff_H
    grad_x[:N-1] += coeff_H * (-dy_seg / ds2_seg)
    grad_x[1:]   += coeff_H * ( dy_seg / ds2_seg)
    grad_y[:N-1] += coeff_H * ( dx_seg / ds2_seg)
    grad_y[1:]   += coeff_H * (-dx_seg / ds2_seg)

    # --- J_smooth ---
    dtheta = _wrap_pi(theta[1:] - theta[:-1])  # (N-1,)
    J_smooth = W_SMOOTH * float(np.sum(dtheta ** 2))

    coeff_S = 2.0 * W_SMOOTH * dtheta
    grad_t[:N-1] -= coeff_S   # d(dtheta_i)/d(theta_i) = -1
    grad_t[1:]   += coeff_S   # d(dtheta_i)/d(theta_{i+1}) = +1

    # --- J_speed ---
    dv = v[1:] - v[:-1]  # (N-1,)
    J_speed = W_SPEED * float(np.sum(dv ** 2))

    coeff_V = 2.0 * W_SPEED * dv
    grad_v[:N-1] -= coeff_V
    grad_v[1:]   += coeff_V

    # Pack gradient into flat 4N vector
    grad = np.empty(4 * N)
    grad[0::4] = grad_x
    grad[1::4] = grad_y
    grad[2::4] = grad_t
    grad[3::4] = grad_v

    return J_cost + J_heading + J_smooth + J_speed, grad


def _make_constraints(
    N: int,
    problem: PlanningProblem,
    landscape: CostLandscape,
    epsilon: float,
) -> list[dict]:
    """Build nonlinear constraint dicts with analytic sparse Jacobians.

    Box constraints (initial state, speed bounds, domain) are handled via the
    bounds parameter in generate_candidate_plans and are NOT included here.
    """
    vehicle = problem.vehicle
    hl, hw = vehicle.half_length, vehicle.half_width

    # Scaled OBB template used in obstacle Jacobian
    local_pts = _OBB_TEMPLATE_UNIT * np.array([hl, hw])  # (5, 2)
    lx_obb = local_pts[:, 0]  # (5,)
    ly_obb = local_pts[:, 1]

    constraints = []

    # ------------------------------------------------------------------ #
    # Inequality: terminal in goal disk
    # ------------------------------------------------------------------ #
    def ineq_goal(z):
        x, y, _, _ = _unpack(z, N)
        dist2 = (x[-1] - problem.x_goal) ** 2 + (y[-1] - problem.y_goal) ** 2
        return np.array([problem.goal_radius ** 2 - dist2])

    def goal_jac(z):
        x, y, _, _ = _unpack(z, N)
        rows = [0, 0]
        cols = [4 * (N - 1), 4 * (N - 1) + 1]
        data = [-2.0 * (x[-1] - problem.x_goal), -2.0 * (y[-1] - problem.y_goal)]
        return csr_matrix((data, (rows, cols)), shape=(1, 4 * N))

    constraints.append({'type': 'ineq', 'fun': ineq_goal, 'jac': goal_jac})

    # ------------------------------------------------------------------ #
    # Inequality: forward progress
    # ------------------------------------------------------------------ #
    def ineq_progress(z):
        x, y, _, _ = _unpack(z, N)
        d = np.sqrt((x - problem.x_goal) ** 2 + (y - problem.y_goal) ** 2)
        return d[:-1] + epsilon - d[1:]

    def progress_jac(z):
        x, y, _, _ = _unpack(z, N)
        d = np.sqrt((x - problem.x_goal) ** 2 + (y - problem.y_goal) ** 2)
        d_safe = np.maximum(d, 1e-8)
        ii = np.arange(N - 1)
        # g_i = d_i + eps - d_{i+1}: 4 nonzeros per row
        rows = np.repeat(ii, 4)
        cols = np.column_stack([4*ii, 4*ii+1, 4*(ii+1), 4*(ii+1)+1]).ravel()
        data = np.column_stack([
             (x[:-1] - problem.x_goal) / d_safe[:-1],
             (y[:-1] - problem.y_goal) / d_safe[:-1],
            -(x[1:]  - problem.x_goal) / d_safe[1:],
            -(y[1:]  - problem.y_goal) / d_safe[1:],
        ]).ravel()
        return csr_matrix((data, (rows, cols)), shape=(N - 1, 4 * N))

    constraints.append({'type': 'ineq', 'fun': ineq_progress,
                         'jac': progress_jac})

    # ------------------------------------------------------------------ #
    # Inequality: longitudinal acceleration bounds
    # ------------------------------------------------------------------ #
    def ineq_accel(z):
        x, y, _, v = _unpack(z, N)
        ds = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
        ds = np.maximum(ds, 0.01)
        dv = v[1:] - v[:-1]
        v_avg = 0.5 * (v[:-1] + v[1:])
        a_long = v_avg * dv / ds
        return np.concatenate([vehicle.a_max - a_long, vehicle.a_max + a_long])

    def accel_jac(z):
        x, y, _, v = _unpack(z, N)
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        ds_raw = np.sqrt(dx ** 2 + dy ** 2)
        clamped = ds_raw < 0.01
        ds = np.maximum(ds_raw, 0.01)
        dv = v[1:] - v[:-1]
        v_avg = 0.5 * (v[:-1] + v[1:])

        # d(a_long)/d(v_i)     = -v_i / ds
        # d(a_long)/d(v_{i+1}) = +v_{i+1} / ds
        da_dvi  = -v[:-1] / ds
        da_dvi1 =  v[1:]  / ds

        # d(a_long)/d(x_i) = v_avg * dv * dx / ds^3  (zero when clamped)
        ds3 = ds ** 3
        da_dxi  = np.where(clamped, 0.0,  v_avg * dv * dx / ds3)
        da_dxi1 = np.where(clamped, 0.0, -v_avg * dv * dx / ds3)
        da_dyi  = np.where(clamped, 0.0,  v_avg * dv * dy / ds3)
        da_dyi1 = np.where(clamped, 0.0, -v_avg * dv * dy / ds3)

        m = N - 1
        ii = np.arange(m)

        # Rows 0..N-2: a_max - a_long >= 0  ->  Jacobian = -d(a_long)/d(z)
        r0 = np.repeat(ii, 6)
        c0 = np.column_stack([4*ii, 4*ii+1, 4*ii+3, 4*(ii+1), 4*(ii+1)+1, 4*(ii+1)+3]).ravel()
        d0 = np.column_stack([-da_dxi, -da_dyi, -da_dvi, -da_dxi1, -da_dyi1, -da_dvi1]).ravel()

        # Rows N-1..2N-3: a_max + a_long >= 0  ->  Jacobian = +d(a_long)/d(z)
        r1 = np.repeat(m + ii, 6)
        c1 = c0  # same column pattern
        d1 = np.column_stack([da_dxi, da_dyi, da_dvi, da_dxi1, da_dyi1, da_dvi1]).ravel()

        rows = np.concatenate([r0, r1])
        cols = np.concatenate([c0, c1])
        data = np.concatenate([d0, d1])
        return csr_matrix((data, (rows, cols)), shape=(2 * m, 4 * N))

    constraints.append({'type': 'ineq', 'fun': ineq_accel, 'jac': accel_jac})

    # ------------------------------------------------------------------ #
    # Inequality: curvature bounds
    # ------------------------------------------------------------------ #
    def ineq_curvature(z):
        x, y, theta, _ = _unpack(z, N)
        ds = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
        ds = np.maximum(ds, 0.01)
        dtheta = _wrap_pi(theta[1:] - theta[:-1])
        kappa = np.abs(dtheta / ds)
        return vehicle.kappa_max - kappa

    def curv_jac(z):
        x, y, theta, _ = _unpack(z, N)
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        ds_raw = np.sqrt(dx ** 2 + dy ** 2)
        clamped = ds_raw < 0.01
        ds = np.maximum(ds_raw, 0.01)
        dth = _wrap_pi(theta[1:] - theta[:-1])
        abs_dth = np.abs(dth)
        sign_dth = np.sign(dth)

        # kappa = |dtheta| / ds
        # d(kappa)/d(theta_i)     = -sign(dtheta) / ds
        # d(kappa)/d(theta_{i+1}) = +sign(dtheta) / ds
        # d(kappa)/d(x_i)         = |dtheta| * dx / ds^3   (zero when clamped)
        # d(g)/d(z) = -d(kappa)/d(z)
        ds3 = ds ** 3
        ii = np.arange(N - 1)

        rows = np.repeat(ii, 6)
        cols = np.column_stack([
            4*ii, 4*ii+1, 4*ii+2, 4*(ii+1), 4*(ii+1)+1, 4*(ii+1)+2
        ]).ravel()
        data = np.column_stack([
            np.where(clamped, 0.0, -abs_dth * dx / ds3),
            np.where(clamped, 0.0, -abs_dth * dy / ds3),
             sign_dth / ds,
            np.where(clamped, 0.0,  abs_dth * dx / ds3),
            np.where(clamped, 0.0,  abs_dth * dy / ds3),
            -sign_dth / ds,
        ]).ravel()
        return csr_matrix((data, (rows, cols)), shape=(N - 1, 4 * N))

    constraints.append({'type': 'ineq', 'fun': ineq_curvature, 'jac': curv_jac})

    # ------------------------------------------------------------------ #
    # Inequality: lateral acceleration bounds
    # ------------------------------------------------------------------ #
    def ineq_lat_accel(z):
        x, y, theta, v = _unpack(z, N)
        ds = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
        ds = np.maximum(ds, 0.01)
        dtheta = _wrap_pi(theta[1:] - theta[:-1])
        kappa = np.abs(dtheta / ds)
        v_avg = 0.5 * (v[:-1] + v[1:])
        a_lat = v_avg ** 2 * kappa
        return vehicle.a_lat_max - a_lat

    def lat_jac(z):
        x, y, theta, v = _unpack(z, N)
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        ds_raw = np.sqrt(dx ** 2 + dy ** 2)
        clamped = ds_raw < 0.01
        ds = np.maximum(ds_raw, 0.01)
        dth = _wrap_pi(theta[1:] - theta[:-1])
        abs_dth = np.abs(dth)
        sign_dth = np.sign(dth)
        kappa = abs_dth / ds
        v_avg = 0.5 * (v[:-1] + v[1:])
        v_avg2 = v_avg ** 2

        # a_lat = v_avg^2 * kappa
        # d(a_lat)/d(v_i)     = v_avg * kappa  (from d(v_avg^2)/d(v_i) = v_avg)
        # d(a_lat)/d(theta_i) = v_avg^2 * (-sign(dtheta)/ds)
        # d(a_lat)/d(x_i)     = v_avg^2 * |dtheta| * dx / ds^3
        # d(g)/d(z) = -d(a_lat)/d(z)
        ds3 = ds ** 3
        ii = np.arange(N - 1)

        rows = np.repeat(ii, 8)
        cols = np.column_stack([
            4*ii, 4*ii+1, 4*ii+2, 4*ii+3,
            4*(ii+1), 4*(ii+1)+1, 4*(ii+1)+2, 4*(ii+1)+3,
        ]).ravel()
        data = np.column_stack([
            np.where(clamped, 0.0, -v_avg2 * abs_dth * dx / ds3),
            np.where(clamped, 0.0, -v_avg2 * abs_dth * dy / ds3),
             v_avg2 * sign_dth / ds,
            -v_avg * kappa,
            np.where(clamped, 0.0,  v_avg2 * abs_dth * dx / ds3),
            np.where(clamped, 0.0,  v_avg2 * abs_dth * dy / ds3),
            -v_avg2 * sign_dth / ds,
            -v_avg * kappa,
        ]).ravel()
        return csr_matrix((data, (rows, cols)), shape=(N - 1, 4 * N))

    constraints.append({'type': 'ineq', 'fun': ineq_lat_accel, 'jac': lat_jac})

    # ------------------------------------------------------------------ #
    # Inequality: obstacle clearance  (hot path - block-diagonal Jacobian)
    # ------------------------------------------------------------------ #
    def ineq_obstacle(z):
        x, y, theta, _ = _unpack(z, N)
        cos_t = np.cos(theta)   # (N,)
        sin_t = np.sin(theta)

        # World coords of all 5N footprint points (vectorized, consistent with Jacobian)
        px = cos_t[:, None] * lx_obb[None, :] - sin_t[:, None] * ly_obb[None, :] + x[:, None]
        py = sin_t[:, None] * lx_obb[None, :] + cos_t[:, None] * ly_obb[None, :] + y[:, None]

        costs = landscape.evaluate(px.ravel(), py.ravel()).reshape(N, 5)
        return problem.obstacle_cost_threshold - costs.max(axis=1)

    def obstacle_jac(z):
        x, y, theta, _ = _unpack(z, N)
        cos_t = np.cos(theta)   # (N,)
        sin_t = np.sin(theta)

        # World coords of all 5N footprint points
        px = cos_t[:, None] * lx_obb[None, :] - sin_t[:, None] * ly_obb[None, :] + x[:, None]
        py = sin_t[:, None] * lx_obb[None, :] + cos_t[:, None] * ly_obb[None, :] + y[:, None]

        # Field cost and gradient at all 5N points (vectorized)
        costs = landscape.evaluate(px.ravel(), py.ravel()).reshape(N, 5)
        gx_all, gy_all = _field_gradient(landscape, px.ravel(), py.ravel())
        gx_all = gx_all.reshape(N, 5)
        gy_all = gy_all.reshape(N, 5)

        # Subgradient at argmax footprint point
        k_star = np.argmax(costs, axis=1)         # (N,)
        ii = np.arange(N)
        gx_max = gx_all[ii, k_star]               # (N,)
        gy_max = gy_all[ii, k_star]

        # d(max_cost)/d(theta_i) via OBB rotation derivative at k_star
        lx_k = lx_obb[k_star]
        ly_k = ly_obb[k_star]
        dp_x_dt = -sin_t * lx_k - cos_t * ly_k   # (N,)
        dp_y_dt =  cos_t * lx_k - sin_t * ly_k
        dmax_dtheta = gx_max * dp_x_dt + gy_max * dp_y_dt  # (N,)

        # g_i = tau - max_cost_i  ->  d(g_i)/d(z) = -d(max_cost_i)/d(z)
        ii = np.arange(N)
        rows = np.repeat(ii, 3)
        cols = np.column_stack([4*ii, 4*ii+1, 4*ii+2]).ravel()
        data = np.column_stack([-gx_max, -gy_max, -dmax_dtheta]).ravel()
        return csr_matrix((data, (rows, cols)), shape=(N, 4 * N))

    constraints.append({'type': 'ineq', 'fun': ineq_obstacle, 'jac': obstacle_jac})

    return constraints


def _compute_epsilon(problem: PlanningProblem) -> float:
    """Compute forward progress slack epsilon."""
    if problem.progress_epsilon != 0.0:
        return problem.progress_epsilon
    straight_line_dist = np.sqrt(
        (problem.x_goal - problem.x_init) ** 2 + (problem.y_goal - problem.y_init) ** 2
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

    dist_goal = np.sqrt((x[-1] - problem.x_goal) ** 2 + (y[-1] - problem.y_goal) ** 2)
    if dist_goal > problem.goal_radius + tol:
        return False

    if np.any(v < vehicle.v_min - tol) or np.any(v > vehicle.v_max + tol):
        return False

    d = np.sqrt((x - problem.x_goal) ** 2 + (y - problem.y_goal) ** 2)
    if np.any(d[:-1] + epsilon + tol < d[1:]):
        return False

    ds = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
    ds = np.maximum(ds, 0.01)
    dtheta = _wrap_pi(theta[1:] - theta[:-1])
    kappa = np.abs(dtheta / ds)

    if np.any(kappa > vehicle.kappa_max + tol):
        return False

    v_avg = 0.5 * (v[:-1] + v[1:])
    a_lat = v_avg ** 2 * kappa
    if np.any(a_lat > vehicle.a_lat_max + tol):
        return False

    dv = v[1:] - v[:-1]
    a_long = v_avg * dv / ds
    if np.any(np.abs(a_long) > vehicle.a_max + tol):
        return False

    for i in range(N):
        _, max_c = landscape.evaluate_footprint(x[i], y[i], theta[i], hl, hw)
        if max_c > problem.obstacle_cost_threshold + tol:
            return False

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

    ds_arr = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
    total_length = float(ds_arr.sum())

    integrated_cost = 0.0
    peak_cost = 0.0
    for i in range(N - 1):
        mean_c, max_c = landscape.evaluate_footprint(x[i], y[i], theta[i], hl, hw)
        integrated_cost += mean_c * ds_arr[i]
        peak_cost = max(peak_cost, max_c)

    _, last_max = landscape.evaluate_footprint(x[-1], y[-1], theta[-1], hl, hw)
    peak_cost = max(peak_cost, last_max)

    v_avg = 0.5 * (v[:-1] + v[1:])
    v_avg = np.maximum(v_avg, 0.01)
    total_time = float(np.sum(ds_arr / v_avg))

    dist_to_goal = float(np.sqrt((x[-1] - problem.x_goal) ** 2 + (y[-1] - problem.y_goal) ** 2))

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


def _to_nonlinear_constraints(cons: list[dict], n_vars: int) -> list:
    """
    Convert SLSQP-style constraint dicts to NonlinearConstraint objects with
    explicit zero Hessians. This prevents trust-constr from falling back to
    quasi-Newton Hessian approximations for each constraint's Lagrangian term,
    which triggers delta_grad==0 warnings when exact Jacobians are supplied.
    """
    _zero_hess = lambda x, v: csr_matrix((n_vars, n_vars))
    result = []
    for c in cons:
        lb, ub = (0.0, 0.0) if c['type'] == 'eq' else (0.0, np.inf)
        result.append(NonlinearConstraint(
            c['fun'], lb, ub, jac=c['jac'], hess=_zero_hess,
        ))
    return result


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
        maxiter: Max solver iterations per seed

    Returns:
        Sorted list of candidate dicts.
    """
    vehicle = problem.vehicle
    N = problem.n_waypoints
    epsilon = _compute_epsilon(problem)
    cfg = landscape.config
    margin = max(vehicle.half_length, vehicle.half_width)

    # Box constraints as bounds: initial state pinned, speed and domain bounded.
    # This removes 124/240 constraint rows from the KKT system.
    bounds = []
    for i in range(N):
        if i == 0:
            bounds += [
                (problem.x_init,     problem.x_init),
                (problem.y_init,     problem.y_init),
                (problem.theta_init, problem.theta_init),
                (problem.v_init,     problem.v_init),
            ]
        else:
            bounds += [
                (cfg.x_min + margin, cfg.x_max - margin),
                (cfg.y_min + margin, cfg.y_max - margin),
                (None, None),
                (vehicle.v_min,      vehicle.v_max),
            ]

    # Diagonal objective Hessian from the quadratic smoothness and speed terms.
    # Provides curvature for theta and v variables, enabling superlinear convergence.
    def _obj_hess(z, *_):
        d = np.zeros(4 * N)
        d[2::4] = 4.0 * W_SMOOTH    # interior theta nodes
        d[2]          = 2.0 * W_SMOOTH  # first endpoint
        d[4*(N-1)+2]  = 2.0 * W_SMOOTH  # last endpoint
        d[3::4] = 4.0 * W_SPEED     # interior v nodes
        d[3]          = 2.0 * W_SPEED
        d[4*(N-1)+3]  = 2.0 * W_SPEED
        return diags(d)

    n_vars = 4 * N
    constraints = _to_nonlinear_constraints(
        _make_constraints(N, problem, landscape, epsilon), n_vars
    )

    candidates = []

    for seed_idx, seed_path in enumerate(seed_paths):
        z0 = _build_initial_z(seed_path, N, problem.v_init, vehicle.v_max)

        result = minimize(
            _objective_and_grad,
            z0,
            args=(N, landscape, vehicle),
            method='trust-constr',
            jac=True,          # _objective_and_grad returns (f, grad)
            hess=_obj_hess,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': maxiter,
                'gtol': 1e-5,
                'verbose': 0,
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
