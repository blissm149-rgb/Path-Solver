"""
event_emulator.py — Trigger system that drives cost landscape evolution.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from .cost_landscape import CostLandscape


@dataclass
class TriggerEvent:
    event_id: int
    timestamp: float          # cumulative Poisson time
    n_sources_before: int
    n_sources_after: int
    cost_delta_norm: float    # L2 norm of delta on low-res sample grid
    description: str          # human-readable summary


class EventEmulator:
    """Drive cost landscape evolution on Poisson-clock trigger events."""

    def __init__(
        self,
        landscape: CostLandscape,
        mean_interval: float = 5.0,
        seed: Optional[int] = None,
    ):
        self.landscape = landscape
        self.mean_interval = mean_interval
        self.rng = np.random.default_rng(seed)

        self._event_counter = 0
        self._cumulative_time = 0.0

        # Precompute sampling grid for delta norm (50×50, clipped to [0, 50])
        cfg = landscape.config
        xs = np.linspace(cfg.x_min, cfg.x_max, 50)
        ys = np.linspace(cfg.y_min, cfg.y_max, 50)
        Xg, Yg = np.meshgrid(xs, ys)
        self._sample_x = Xg.ravel()
        self._sample_y = Yg.ravel()

    def trigger(self) -> TriggerEvent:
        """
        Apply one landscape.update(), compute cost delta norm,
        advance Poisson clock, return event record.
        """
        n_before = self.landscape.n_sources()

        # Snapshot cost before
        before = np.clip(
            self.landscape.evaluate(self._sample_x, self._sample_y),
            0.0, 50.0
        )

        # Apply update
        self.landscape.update()

        # Snapshot cost after
        after = np.clip(
            self.landscape.evaluate(self._sample_x, self._sample_y),
            0.0, 50.0
        )

        delta_norm = float(np.linalg.norm(after - before))
        n_after = self.landscape.n_sources()

        # Advance Poisson clock
        dt = self.rng.exponential(self.mean_interval)
        self._cumulative_time += dt

        self._event_counter += 1
        event = TriggerEvent(
            event_id=self._event_counter,
            timestamp=self._cumulative_time,
            n_sources_before=n_before,
            n_sources_after=n_after,
            cost_delta_norm=delta_norm,
            description=(
                f"Event {self._event_counter}: t={self._cumulative_time:.2f}s, "
                f"sources {n_before}→{n_after}, "
                f"Δcost_norm={delta_norm:.3f}"
            ),
        )
        return event

    def run_sequence(
        self,
        n_events: int,
        callback: Optional[Callable[[TriggerEvent, CostLandscape], None]] = None,
    ) -> list[TriggerEvent]:
        """Run N triggers, calling callback(event, landscape) after each."""
        events = []
        for _ in range(n_events):
            event = self.trigger()
            events.append(event)
            if callback is not None:
                callback(event, self.landscape)
        return events
