"""
graph_search.py — A* seed generator with penalty-based diversity mechanism.
"""

from __future__ import annotations

import heapq
import numpy as np
from typing import Optional

from .cost_landscape import CostLandscape


def astar_on_grid(
    cost_grid: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    obstacle_threshold: float = 100.0,
) -> Optional[np.ndarray]:
    """
    Standard 8-connected A* on 2D cost grid.

    Args:
        cost_grid: (rows, cols) array of cost values
        start_rc: (row, col) start cell
        goal_rc: (row, col) goal cell
        x_coords: 1D array of x-coordinates for each column
        y_coords: 1D array of y-coordinates for each row
        obstacle_threshold: cells with cost > threshold are blocked

    Returns:
        (N, 2) xy path array, or None if no path found.
    """
    rows, cols = cost_grid.shape
    start_r, start_c = start_rc
    goal_r, goal_c = goal_rc

    if cost_grid[start_r, start_c] > obstacle_threshold:
        # Start is in obstacle — find nearest free cell
        start_r, start_c = _find_nearest_free(cost_grid, start_r, start_c, obstacle_threshold)

    if cost_grid[goal_r, goal_c] > obstacle_threshold:
        goal_r, goal_c = _find_nearest_free(cost_grid, goal_r, goal_c, obstacle_threshold)

    # Euclidean heuristic
    def heuristic(r, c):
        dr = r - goal_r
        dc = c - goal_c
        return np.sqrt(dr * dr + dc * dc)

    # 8-connected neighbors with diagonal cost weighting
    directions = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
        (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)),
    ]

    open_heap = []
    g_score = np.full((rows, cols), np.inf)
    g_score[start_r, start_c] = 0.0
    came_from = {}

    h0 = heuristic(start_r, start_c)
    heapq.heappush(open_heap, (h0, 0.0, start_r, start_c))

    closed = np.zeros((rows, cols), dtype=bool)

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)
        if closed[r, c]:
            continue
        closed[r, c] = True

        if r == goal_r and c == goal_c:
            return _reconstruct_path(came_from, r, c, x_coords, y_coords)

        for dr, dc, step_dist in directions:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if closed[nr, nc]:
                continue
            if cost_grid[nr, nc] > obstacle_threshold:
                continue

            # Edge cost: distance × average cell cost
            avg_cost = 0.5 * (cost_grid[r, c] + cost_grid[nr, nc])
            # Use actual world-space distance for edge weight
            dx = x_coords[nc] - x_coords[c]
            dy = y_coords[nr] - y_coords[r]
            dist = np.sqrt(dx * dx + dy * dy)
            edge_cost = dist * max(avg_cost, 0.01)

            tentative_g = g + edge_cost
            if tentative_g < g_score[nr, nc]:
                g_score[nr, nc] = tentative_g
                came_from[(nr, nc)] = (r, c)
                f_new = tentative_g + heuristic(nr, nc)
                heapq.heappush(open_heap, (f_new, tentative_g, nr, nc))

    return None  # No path found


def _find_nearest_free(
    cost_grid: np.ndarray,
    r: int,
    c: int,
    threshold: float,
) -> tuple[int, int]:
    """BFS to find nearest free cell from (r, c)."""
    rows, cols = cost_grid.shape
    from collections import deque
    q = deque([(r, c)])
    visited = {(r, c)}
    while q:
        cr, cc = q.popleft()
        if cost_grid[cr, cc] <= threshold:
            return cr, cc
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return r, c  # fallback


def _reconstruct_path(
    came_from: dict,
    r: int,
    c: int,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """Trace back parent pointers to get path as (N, 2) xy array."""
    path_rc = []
    cur = (r, c)
    while cur in came_from:
        path_rc.append(cur)
        cur = came_from[cur]
    path_rc.append(cur)
    path_rc.reverse()

    path_xy = np.array([[x_coords[c], y_coords[r]] for r, c in path_rc])
    return path_xy


def resample_path(path: np.ndarray, n_waypoints: int) -> np.ndarray:
    """
    Resample arbitrary-length path to fixed N equally-spaced waypoints
    by arc length via linear interpolation.

    Args:
        path: (M, 2) array of xy points
        n_waypoints: target number of waypoints

    Returns:
        (n_waypoints, 2) array
    """
    if len(path) < 2:
        # Degenerate path — return copies of the single point
        return np.tile(path[0], (n_waypoints, 1))

    # Compute cumulative arc length
    diffs = np.diff(path, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len < 1e-10:
        return np.tile(path[0], (n_waypoints, 1))

    # Target parameter values
    target = np.linspace(0.0, total_len, n_waypoints)

    # Interpolate
    resampled = np.zeros((n_waypoints, 2))
    resampled[0] = path[0]
    resampled[-1] = path[-1]

    j = 0
    for i in range(1, n_waypoints - 1):
        s = target[i]
        while j < len(cum_len) - 2 and cum_len[j + 1] < s:
            j += 1
        seg_start = cum_len[j]
        seg_end = cum_len[j + 1]
        if abs(seg_end - seg_start) < 1e-12:
            t = 0.0
        else:
            t = (s - seg_start) / (seg_end - seg_start)
        resampled[i] = path[j] + t * (path[j + 1] - path[j])

    return resampled


def generate_diverse_seeds(
    landscape: CostLandscape,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    n_seeds: int = 3,
    grid_resolution: int = 80,
    penalty_sigma: float = 8.0,
    penalty_amplitude: float = 5.0,
) -> list[np.ndarray]:
    """
    Generate K topologically diverse seed paths via penalty-based repeated A*.

    After finding path k, add a Gaussian penalty along that path to push
    subsequent searches into different corridors.

    Returns:
        List of (n_waypoints, 2) path arrays (raw A* paths, not resampled).
    """
    cfg = landscape.config

    # Build coordinate arrays
    x_coords = np.linspace(cfg.x_min, cfg.x_max, grid_resolution)
    y_coords = np.linspace(cfg.y_min, cfg.y_max, grid_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    x_flat = X.ravel()
    y_flat = Y.ravel()

    # Base cost grid (rows=y, cols=x)
    base_cost = landscape.evaluate(x_flat, y_flat).reshape(grid_resolution, grid_resolution)

    # Map start/goal to grid cells
    start_c = int(np.clip(
        np.searchsorted(x_coords, start_xy[0]), 0, grid_resolution - 1
    ))
    start_r = int(np.clip(
        np.searchsorted(y_coords, start_xy[1]), 0, grid_resolution - 1
    ))
    goal_c = int(np.clip(
        np.searchsorted(x_coords, goal_xy[0]), 0, grid_resolution - 1
    ))
    goal_r = int(np.clip(
        np.searchsorted(y_coords, goal_xy[1]), 0, grid_resolution - 1
    ))

    penalty_grid = np.zeros((grid_resolution, grid_resolution))
    seeds = []

    for k in range(n_seeds):
        current_cost = base_cost + penalty_grid

        path = astar_on_grid(
            current_cost,
            (start_r, start_c),
            (goal_r, goal_c),
            x_coords,
            y_coords,
            obstacle_threshold=landscape.config.obstacle_amplitude * 0.01,
        )

        if path is None:
            # If A* fails, skip this seed
            continue

        seeds.append(path)

        # Add Gaussian penalty along found path
        for pt in path:
            dx = x_flat - pt[0]
            dy = y_flat - pt[1]
            dist_sq = dx * dx + dy * dy
            penalty_grid += (
                penalty_amplitude * np.exp(-dist_sq / (2.0 * penalty_sigma ** 2))
            ).reshape(grid_resolution, grid_resolution)

    return seeds
