"""
cost_landscape.py — 2D continuous scalar cost field composed of RBF sources
with OU-process dynamics for spatiotemporally correlated evolution.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class RBFSource:
    cx: float           # center x
    cy: float           # center y
    sigma_x: float      # width along local x-axis
    sigma_y: float      # width along local y-axis
    amplitude: float    # peak cost contribution (positive)
    theta: float        # rotation angle for elongated features
    age: int = 0        # update count since birth


@dataclass
class CostLandscapeConfig:
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0
    background_cost: float = 0.1

    # Initial generation
    n_initial_sources: int = 15
    amplitude_range: tuple = (0.5, 5.0)
    sigma_range: tuple = (3.0, 15.0)
    elongation_prob: float = 0.3       # fraction of sources that are ridges

    # Hard obstacles (very high amplitude, tight sigma)
    n_obstacles: int = 4
    obstacle_amplitude: float = 1e4
    obstacle_sigma: float = 3.0

    # OU process parameters
    ou_theta_amplitude: float = 0.3    # mean reversion rate
    ou_sigma_amplitude: float = 0.4    # volatility
    ou_theta_position: float = 0.1
    ou_sigma_position: float = 1.5
    ou_sigma_width: float = 0.3

    # Birth/death
    birth_rate: float = 0.3            # Poisson rate per update
    death_rate: float = 0.15           # per-source probability per update
    min_sources: int = 8
    max_sources: int = 25

    seed: Optional[int] = None


# Precomputed 5-point local OBB template (center + 4 corners) in local frame
# Shape: (5, 2) — to be scaled by half_length, half_width at call time
_OBB_TEMPLATE_UNIT = np.array([
    [0.0,  0.0],   # center
    [1.0,  1.0],   # front-left
    [1.0, -1.0],   # front-right
    [-1.0, 1.0],   # rear-left
    [-1.0, -1.0],  # rear-right
], dtype=np.float64)


class CostLandscape:
    """2D continuous scalar cost field composed of RBF sources with OU dynamics."""

    def __init__(self, config: Optional[CostLandscapeConfig] = None):
        if config is None:
            config = CostLandscapeConfig()
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self._sources: list[RBFSource] = []
        self._obstacles: list[RBFSource] = []

        self._generate_initial_sources()
        self._generate_obstacles()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of cost field at query points (x, y)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        cost = np.full_like(x, self.config.background_cost, dtype=np.float64)

        for src in self._sources + self._obstacles:
            cost += self._eval_source(src, x, y)

        return cost

    def evaluate_footprint(
        self,
        cx: float,
        cy: float,
        theta: float,
        half_length: float,
        half_width: float,
    ) -> Tuple[float, float]:
        """
        Evaluate cost over 5 OBB sample points (center + 4 corners).
        Returns (mean_cost, max_cost).
        This is the hot path — precomputed template, single vectorized call.
        """
        # Scale unit template by half extents
        local_pts = _OBB_TEMPLATE_UNIT * np.array([half_length, half_width])  # (5,2)

        # Rotate into world frame
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]], dtype=np.float64)
        world_pts = local_pts @ R.T  # (5,2)

        px = world_pts[:, 0] + cx
        py = world_pts[:, 1] + cy

        costs = self.evaluate(px, py)
        return float(costs.mean()), float(costs.max())

    def is_obstacle(
        self,
        cx: float,
        cy: float,
        theta: float,
        hl: float,
        hw: float,
        threshold: float = 100.0,
    ) -> bool:
        """True if max footprint cost exceeds threshold."""
        _, max_cost = self.evaluate_footprint(cx, cy, theta, hl, hw)
        return max_cost > threshold

    def update(self) -> None:
        """One OU step: evolve all source params, birth/death."""
        cfg = self.config
        rng = self.rng

        # Evolve regular sources
        surviving = []
        for src in self._sources:
            # Death check
            death_prob = cfg.death_rate * (1.0 + 0.1 * src.age)
            if (len(surviving) + len(self._sources) - len(surviving) - 1 > cfg.min_sources
                    and rng.random() < death_prob):
                continue  # source dies

            # Amplitude OU
            mean_amp = (cfg.amplitude_range[0] + cfg.amplitude_range[1]) / 2.0
            src.amplitude += (cfg.ou_theta_amplitude * (mean_amp - src.amplitude)
                              + cfg.ou_sigma_amplitude * rng.standard_normal())
            src.amplitude = max(0.1, src.amplitude)

            # Position OU (weak mean-reversion to domain center)
            domain_cx = (cfg.x_min + cfg.x_max) / 2.0
            domain_cy = (cfg.y_min + cfg.y_max) / 2.0
            src.cx += (cfg.ou_theta_position * (domain_cx - src.cx) * 0.01
                       + cfg.ou_sigma_position * rng.standard_normal())
            src.cy += (cfg.ou_theta_position * (domain_cy - src.cy) * 0.01
                       + cfg.ou_sigma_position * rng.standard_normal())
            src.cx = float(np.clip(src.cx, cfg.x_min, cfg.x_max))
            src.cy = float(np.clip(src.cy, cfg.y_min, cfg.y_max))

            # Width OU
            src.sigma_x += cfg.ou_sigma_width * rng.standard_normal()
            src.sigma_y += cfg.ou_sigma_width * rng.standard_normal()
            src.sigma_x = float(np.clip(src.sigma_x, 1.0, 25.0))
            src.sigma_y = float(np.clip(src.sigma_y, 1.0, 25.0))

            # Rotation slow drift
            src.theta += 0.05 * rng.standard_normal()

            src.age += 1
            surviving.append(src)

        # Birth: new sources
        n_new = rng.poisson(cfg.birth_rate)
        if len(surviving) + n_new > cfg.max_sources:
            n_new = max(0, cfg.max_sources - len(surviving))

        for _ in range(n_new):
            surviving.append(self._make_source(amplitude_scale=0.3))

        # Enforce min sources
        while len(surviving) < cfg.min_sources:
            surviving.append(self._make_source())

        self._sources = surviving

        # Evolve obstacles (slow drift, no death)
        for obs in self._obstacles:
            obs.cx += 0.3 * rng.standard_normal()
            obs.cy += 0.3 * rng.standard_normal()
            obs.cx = float(np.clip(obs.cx, cfg.x_min + 5.0, cfg.x_max - 5.0))
            obs.cy = float(np.clip(obs.cy, cfg.y_min + 5.0, cfg.y_max - 5.0))

    def to_grid(self, resolution: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rasterize to meshgrid (X, Y, C) for visualization/A*."""
        cfg = self.config
        xs = np.linspace(cfg.x_min, cfg.x_max, resolution)
        ys = np.linspace(cfg.y_min, cfg.y_max, resolution)
        X, Y = np.meshgrid(xs, ys)
        C = self.evaluate(X.ravel(), Y.ravel()).reshape(X.shape)
        return X, Y, C

    def get_obstacle_centers(self) -> np.ndarray:
        """Return obstacle centers as (N, 2) array."""
        if not self._obstacles:
            return np.empty((0, 2))
        return np.array([[o.cx, o.cy] for o in self._obstacles])

    def get_obstacle_radii(self) -> np.ndarray:
        """Return 3σ effective radii for obstacles."""
        if not self._obstacles:
            return np.empty(0)
        return np.array([3.0 * max(o.sigma_x, o.sigma_y) for o in self._obstacles])

    def n_sources(self) -> int:
        """Number of regular (non-obstacle) sources."""
        return len(self._sources)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_source(self, src: RBFSource, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate a single RBF source at query points (vectorized)."""
        dx = x - src.cx
        dy = y - src.cy
        cos_t = np.cos(src.theta)
        sin_t = np.sin(src.theta)
        u = cos_t * dx + sin_t * dy
        v = -sin_t * dx + cos_t * dy
        exponent = 0.5 * ((u / src.sigma_x) ** 2 + (v / src.sigma_y) ** 2)
        return src.amplitude * np.exp(-exponent)

    def _make_source(self, amplitude_scale: float = 1.0) -> RBFSource:
        """Create a new random RBF source."""
        cfg = self.config
        rng = self.rng

        amp = rng.uniform(*cfg.amplitude_range) * amplitude_scale
        sigma_base = rng.uniform(*cfg.sigma_range)

        if rng.random() < cfg.elongation_prob:
            # Elongated ridge
            sigma_x = sigma_base
            sigma_y = sigma_base * rng.uniform(0.2, 0.5)
        else:
            sigma_x = sigma_base
            sigma_y = sigma_base * rng.uniform(0.7, 1.3)

        return RBFSource(
            cx=rng.uniform(cfg.x_min, cfg.x_max),
            cy=rng.uniform(cfg.y_min, cfg.y_max),
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            amplitude=amp,
            theta=rng.uniform(0, np.pi),
            age=0,
        )

    def _generate_initial_sources(self) -> None:
        """Generate the initial set of regular RBF sources."""
        for _ in range(self.config.n_initial_sources):
            self._sources.append(self._make_source())

    def _generate_obstacles(self) -> None:
        """Generate hard obstacle sources."""
        cfg = self.config
        rng = self.rng

        for _ in range(cfg.n_obstacles):
            # Place obstacles away from start/goal corners
            cx = rng.uniform(cfg.x_min + 15, cfg.x_max - 15)
            cy = rng.uniform(cfg.y_min + 15, cfg.y_max - 15)
            self._obstacles.append(RBFSource(
                cx=cx,
                cy=cy,
                sigma_x=cfg.obstacle_sigma,
                sigma_y=cfg.obstacle_sigma,
                amplitude=cfg.obstacle_amplitude,
                theta=0.0,
                age=0,
            ))
