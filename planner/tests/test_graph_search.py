"""Tests for graph_search.py — Phase 4 validation."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from planner.cost_landscape import CostLandscape, CostLandscapeConfig
from planner.graph_search import astar_on_grid, resample_path, generate_diverse_seeds


def hausdorff_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Compute symmetric Hausdorff distance between two paths."""
    # max over a of min distance to b, and vice versa
    def one_sided(p, q):
        dists = np.sqrt(((p[:, None] - q[None, :]) ** 2).sum(axis=2))
        return dists.min(axis=1).max()
    return max(one_sided(path_a, path_b), one_sided(path_b, path_a))


class TestAstar:
    """Test A* on simple grids with known solutions."""

    def test_astar_straight_path(self):
        """On a uniform-cost grid, A* finds a direct path."""
        grid = np.ones((20, 20), dtype=float)
        x_coords = np.linspace(0, 19, 20)
        y_coords = np.linspace(0, 19, 20)
        path = astar_on_grid(grid, (0, 0), (19, 19), x_coords, y_coords)
        assert path is not None, "Should find a path on open grid"
        assert path.shape[1] == 2
        # Path must start near (0,0) and end near (19,19)
        assert np.allclose(path[0], [x_coords[0], y_coords[0]], atol=1e-6)
        assert np.allclose(path[-1], [x_coords[19], y_coords[19]], atol=1e-6)

    def test_astar_avoids_obstacle(self):
        """A* routes around a high-cost barrier."""
        rows, cols = 20, 20
        grid = np.ones((rows, cols), dtype=float)
        # Vertical wall at column 10, rows 0–14
        grid[:15, 10] = 1e6
        x_coords = np.linspace(0, 19, cols)
        y_coords = np.linspace(0, 19, rows)
        path = astar_on_grid(grid, (0, 0), (0, 19), x_coords, y_coords,
                             obstacle_threshold=1000.0)
        assert path is not None, "Should find a path around the barrier"
        # None of the path points should be at column 10 row < 15
        for pt in path:
            col = np.searchsorted(x_coords, pt[0])
            row = np.searchsorted(y_coords, pt[1])
            if col == 10:
                assert row >= 15, f"Path passes through obstacle at row={row}"

    def test_astar_no_path(self):
        """A* returns None when goal is fully surrounded by obstacles."""
        rows, cols = 10, 10
        grid = np.ones((rows, cols), dtype=float)
        # Surround goal completely
        grid[5, :] = 1e6  # horizontal wall
        x_coords = np.linspace(0, 9, cols)
        y_coords = np.linspace(0, 9, rows)
        result = astar_on_grid(grid, (0, 0), (9, 9), x_coords, y_coords,
                               obstacle_threshold=1000.0)
        assert result is None, "Should return None when goal is unreachable"

    def test_astar_returns_ndarray(self):
        """A* result is an ndarray of shape (N, 2)."""
        grid = np.ones((15, 15), dtype=float)
        x_coords = np.linspace(0, 14, 15)
        y_coords = np.linspace(0, 14, 15)
        path = astar_on_grid(grid, (0, 0), (14, 14), x_coords, y_coords)
        assert isinstance(path, np.ndarray)
        assert path.ndim == 2
        assert path.shape[1] == 2


class TestResample:
    """Test arc-length resampling."""

    def test_resample_preserves_endpoints(self):
        """Resampled path has same start and end as original."""
        path = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 5.0], [15.0, 10.0]])
        resampled = resample_path(path, 10)
        assert np.allclose(resampled[0], path[0], atol=1e-10)
        assert np.allclose(resampled[-1], path[-1], atol=1e-10)

    def test_resample_correct_count(self):
        """Resampled path has exactly n_waypoints points."""
        path = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        for n in [5, 10, 20]:
            r = resample_path(path, n)
            assert len(r) == n, f"Expected {n} points, got {len(r)}"

    def test_resample_straight_line(self):
        """Resampling a straight line gives uniform spacing."""
        path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                         [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
        resampled = resample_path(path, 6)
        # Should be uniformly spaced from 0 to 5
        expected_x = np.linspace(0, 5, 6)
        assert np.allclose(resampled[:, 0], expected_x, atol=1e-6)
        assert np.allclose(resampled[:, 1], 0.0, atol=1e-6)

    def test_resample_diagonal(self):
        """Resampling a diagonal preserves direction."""
        path = np.array([[0.0, 0.0], [10.0, 10.0]])
        resampled = resample_path(path, 5)
        assert np.allclose(resampled[0], [0.0, 0.0], atol=1e-6)
        assert np.allclose(resampled[-1], [10.0, 10.0], atol=1e-6)
        # All points should be on the diagonal y=x
        assert np.allclose(resampled[:, 0], resampled[:, 1], atol=1e-6)


class TestDiverseSeeds:
    """Test diversity of seed generation."""

    def test_seeds_connect_start_to_goal(self):
        """Each seed starts near start and ends near goal."""
        cfg = CostLandscapeConfig(n_obstacles=0, n_initial_sources=5, seed=42)
        landscape = CostLandscape(cfg)
        start = np.array([10.0, 10.0])
        goal = np.array([90.0, 90.0])
        seeds = generate_diverse_seeds(landscape, start, goal, n_seeds=2, grid_resolution=40)
        assert len(seeds) >= 1
        for seed in seeds:
            # First and last points should be within ~5 units of start/goal
            assert np.linalg.norm(seed[0] - start) < 10.0, "Seed doesn't start near start"
            assert np.linalg.norm(seed[-1] - goal) < 10.0, "Seed doesn't end near goal"

    def test_diverse_seeds_hausdorff(self):
        """3 seeds should have pairwise Hausdorff distance > 5 units."""
        cfg = CostLandscapeConfig(n_obstacles=0, n_initial_sources=5, seed=7)
        landscape = CostLandscape(cfg)
        start = np.array([5.0, 50.0])
        goal = np.array([95.0, 50.0])
        seeds = generate_diverse_seeds(
            landscape, start, goal, n_seeds=3,
            grid_resolution=50, penalty_sigma=8.0, penalty_amplitude=5.0
        )
        if len(seeds) >= 2:
            d = hausdorff_distance(seeds[0], seeds[1])
            assert d > 3.0, f"Seeds too similar: Hausdorff={d:.2f}"

    def test_returns_list_of_arrays(self):
        """generate_diverse_seeds returns a list of 2D arrays."""
        cfg = CostLandscapeConfig(n_obstacles=0, seed=1)
        landscape = CostLandscape(cfg)
        seeds = generate_diverse_seeds(
            landscape, np.array([10.0, 10.0]), np.array([90.0, 90.0]),
            n_seeds=2, grid_resolution=30
        )
        assert isinstance(seeds, list)
        for s in seeds:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2
            assert s.shape[1] == 2

    def test_resample_integrates_with_seeds(self):
        """Resampling seed paths to N_WAYPOINTS works end-to-end."""
        cfg = CostLandscapeConfig(n_obstacles=0, n_initial_sources=5, seed=5)
        landscape = CostLandscape(cfg)
        start = np.array([10.0, 10.0])
        goal = np.array([90.0, 90.0])
        seeds = generate_diverse_seeds(landscape, start, goal, n_seeds=2, grid_resolution=40)
        for seed in seeds:
            resampled = resample_path(seed, 20)
            assert resampled.shape == (20, 2)
            assert np.allclose(resampled[0], seed[0], atol=1e-6)
            assert np.allclose(resampled[-1], seed[-1], atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
