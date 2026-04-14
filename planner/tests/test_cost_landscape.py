"""Tests for cost_landscape.py — Phase 1 validation."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from planner.cost_landscape import CostLandscape, CostLandscapeConfig, RBFSource


class TestRBFEvaluation:
    """Test RBF evaluation at known points."""

    def test_single_source_at_center(self):
        """A source evaluated at its own center returns amplitude + background."""
        cfg = CostLandscapeConfig(
            n_initial_sources=0, n_obstacles=0, seed=0
        )
        landscape = CostLandscape(cfg)
        # Manually add one source at (50, 50)
        landscape._sources = [RBFSource(
            cx=50.0, cy=50.0, sigma_x=5.0, sigma_y=5.0,
            amplitude=2.0, theta=0.0
        )]
        cost = landscape.evaluate(np.array([50.0]), np.array([50.0]))
        expected = cfg.background_cost + 2.0  # amplitude at center, exp(-0) = 1
        assert abs(cost[0] - expected) < 1e-10

    def test_rbf_decay(self):
        """Cost decays with distance from source center."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        landscape._sources = [RBFSource(
            cx=50.0, cy=50.0, sigma_x=5.0, sigma_y=5.0,
            amplitude=3.0, theta=0.0
        )]
        c_center = landscape.evaluate(np.array([50.0]), np.array([50.0]))[0]
        c_far = landscape.evaluate(np.array([70.0]), np.array([50.0]))[0]
        assert c_center > c_far, "Cost at center should exceed cost farther away"

    def test_rbf_rotation(self):
        """Rotated source has cost along rotated axis."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        # Elongated source pointing along y-axis (theta=pi/2)
        landscape._sources = [RBFSource(
            cx=50.0, cy=50.0, sigma_x=10.0, sigma_y=2.0,
            amplitude=5.0, theta=np.pi / 2
        )]
        # Point along local x-axis (i.e., world y after pi/2 rotation) — should be high
        c_along_local_x = landscape.evaluate(np.array([50.0]), np.array([55.0]))[0]
        # Point along local y-axis (world x) — should be lower (tight sigma_y=2)
        c_along_local_y = landscape.evaluate(np.array([55.0]), np.array([50.0]))[0]
        assert c_along_local_x > c_along_local_y

    def test_background_cost_minimum(self):
        """Far from all sources, cost ≈ background."""
        cfg = CostLandscapeConfig(
            n_initial_sources=0, n_obstacles=0,
            background_cost=0.1, seed=0
        )
        landscape = CostLandscape(cfg)
        landscape._sources = [RBFSource(
            cx=50.0, cy=50.0, sigma_x=2.0, sigma_y=2.0,
            amplitude=5.0, theta=0.0
        )]
        # Query far away — cost should be very close to background
        c = landscape.evaluate(np.array([0.0]), np.array([0.0]))[0]
        assert abs(c - 0.1) < 1e-5

    def test_vectorized_evaluation(self):
        """Evaluate on a 2D grid returns correct shape."""
        cfg = CostLandscapeConfig(seed=42)
        landscape = CostLandscape(cfg)
        X, Y, C = landscape.to_grid(resolution=50)
        assert X.shape == (50, 50)
        assert Y.shape == (50, 50)
        assert C.shape == (50, 50)
        assert np.all(C >= 0), "Cost should be non-negative"

    def test_rbf_analytical_value(self):
        """Verify RBF formula analytically for known inputs."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        # sigma_x = 10, sigma_y = 10, theta = 0
        # At (60, 50): dx=10, dy=0 → u=10, v=0 → exp(-0.5*(10/10)^2) = exp(-0.5)
        landscape._sources = [RBFSource(
            cx=50.0, cy=50.0, sigma_x=10.0, sigma_y=10.0,
            amplitude=4.0, theta=0.0
        )]
        c = landscape.evaluate(np.array([60.0]), np.array([50.0]))[0]
        expected = 0.1 + 4.0 * np.exp(-0.5)
        assert abs(c - expected) < 1e-10


class TestFootprintEvaluation:
    """Test OBB footprint evaluation."""

    def test_footprint_at_origin_theta_zero(self):
        """At theta=0, footprint samples align with axes."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        # Uniform cost field (background only)
        mean_c, max_c = landscape.evaluate_footprint(50.0, 50.0, 0.0, 2.5, 1.25)
        assert abs(mean_c - 0.1) < 1e-10
        assert abs(max_c - 0.1) < 1e-10

    def test_footprint_symmetry(self):
        """Footprint at theta=pi/2 gives same values as theta=0 for isotropic source."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        landscape._sources = [RBFSource(
            cx=50.0, cy=50.0, sigma_x=5.0, sigma_y=5.0,
            amplitude=3.0, theta=0.0  # isotropic source
        )]
        mean_0, max_0 = landscape.evaluate_footprint(50.0, 50.0, 0.0, 2.5, 1.25)
        mean_90, max_90 = landscape.evaluate_footprint(50.0, 50.0, np.pi / 2, 2.5, 1.25)
        assert abs(mean_0 - mean_90) < 1e-6
        assert abs(max_0 - max_90) < 1e-6

    def test_footprint_detects_obstacle(self):
        """is_obstacle returns True when footprint overlaps high-cost region."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        landscape._obstacles = [RBFSource(
            cx=50.0, cy=50.0, sigma_x=3.0, sigma_y=3.0,
            amplitude=1e4, theta=0.0
        )]
        assert landscape.is_obstacle(50.0, 50.0, 0.0, 2.5, 1.25, threshold=100.0)

    def test_footprint_clear_of_obstacle(self):
        """is_obstacle returns False when footprint is far from obstacle."""
        cfg = CostLandscapeConfig(n_initial_sources=0, n_obstacles=0, seed=0)
        landscape = CostLandscape(cfg)
        landscape._obstacles = [RBFSource(
            cx=80.0, cy=80.0, sigma_x=3.0, sigma_y=3.0,
            amplitude=1e4, theta=0.0
        )]
        assert not landscape.is_obstacle(10.0, 10.0, 0.0, 2.5, 1.25, threshold=100.0)


class TestUpdateDynamics:
    """Test OU process update dynamics."""

    def test_update_produces_correlated_field(self):
        """L2 delta between consecutive fields is non-zero but bounded."""
        cfg = CostLandscapeConfig(seed=42)
        landscape = CostLandscape(cfg)

        # Sample on low-res grid
        xs = np.linspace(0, 100, 50)
        ys = np.linspace(0, 100, 50)
        X, Y = np.meshgrid(xs, ys)
        x_flat, y_flat = X.ravel(), Y.ravel()

        before = np.clip(landscape.evaluate(x_flat, y_flat), 0, 50)
        landscape.update()
        after = np.clip(landscape.evaluate(x_flat, y_flat), 0, 50)

        delta_norm = float(np.linalg.norm(after - before))
        assert delta_norm > 0, "Update should change cost field"
        assert delta_norm < 5000, "Cost field should not change catastrophically"

    def test_source_count_stays_in_bounds(self):
        """After 20 updates, source count stays in [min_sources, max_sources]."""
        cfg = CostLandscapeConfig(seed=7)
        landscape = CostLandscape(cfg)
        for _ in range(20):
            landscape.update()
            n = landscape.n_sources()
            assert n >= cfg.min_sources, f"Too few sources: {n}"
            assert n <= cfg.max_sources, f"Too many sources: {n}"

    def test_amplitude_stays_positive(self):
        """After updates, all amplitudes remain positive."""
        cfg = CostLandscapeConfig(seed=99)
        landscape = CostLandscape(cfg)
        for _ in range(10):
            landscape.update()
        for src in landscape._sources:
            assert src.amplitude > 0, f"Amplitude went non-positive: {src.amplitude}"

    def test_positions_stay_in_domain(self):
        """After updates, all source positions remain in domain."""
        cfg = CostLandscapeConfig(seed=13)
        landscape = CostLandscape(cfg)
        for _ in range(15):
            landscape.update()
        for src in landscape._sources:
            assert cfg.x_min <= src.cx <= cfg.x_max
            assert cfg.y_min <= src.cy <= cfg.y_max

    def test_obstacle_centers_accessible(self):
        """get_obstacle_centers returns correct shape."""
        cfg = CostLandscapeConfig(n_obstacles=4, seed=1)
        landscape = CostLandscape(cfg)
        centers = landscape.get_obstacle_centers()
        assert centers.shape == (4, 2)

    def test_obstacle_radii_accessible(self):
        """get_obstacle_radii returns correct shape."""
        cfg = CostLandscapeConfig(n_obstacles=4, seed=2)
        landscape = CostLandscape(cfg)
        radii = landscape.get_obstacle_radii()
        assert radii.shape == (4,)
        assert np.all(radii > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
