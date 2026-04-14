"""Tests for event_emulator.py — Phase 3 validation."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from planner.cost_landscape import CostLandscape, CostLandscapeConfig
from planner.event_emulator import EventEmulator, TriggerEvent


class TestEventEmulator:
    """Test trigger system and event records."""

    def setup_method(self):
        cfg = CostLandscapeConfig(seed=42)
        self.landscape = CostLandscape(cfg)
        self.emulator = EventEmulator(self.landscape, mean_interval=5.0, seed=100)

    def test_trigger_returns_event(self):
        """trigger() returns a TriggerEvent with correct types."""
        event = self.emulator.trigger()
        assert isinstance(event, TriggerEvent)
        assert isinstance(event.event_id, int)
        assert isinstance(event.timestamp, float)
        assert isinstance(event.cost_delta_norm, float)
        assert isinstance(event.description, str)

    def test_event_id_increments(self):
        """event_id increments by 1 each trigger."""
        e1 = self.emulator.trigger()
        e2 = self.emulator.trigger()
        e3 = self.emulator.trigger()
        assert e1.event_id == 1
        assert e2.event_id == 2
        assert e3.event_id == 3

    def test_monotonic_timestamps(self):
        """Timestamps are strictly increasing across 10 triggers."""
        events = self.emulator.run_sequence(10)
        timestamps = [e.timestamp for e in events]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], (
                f"Timestamp not increasing: t[{i-1}]={timestamps[i-1]}, t[{i}]={timestamps[i]}"
            )

    def test_source_counts_in_bounds(self):
        """Source counts before/after each trigger stay in config bounds."""
        cfg = self.landscape.config
        events = self.emulator.run_sequence(10)
        for e in events:
            assert e.n_sources_before >= cfg.min_sources
            assert e.n_sources_before <= cfg.max_sources
            assert e.n_sources_after >= cfg.min_sources
            assert e.n_sources_after <= cfg.max_sources

    def test_cost_delta_norm_nonnegative(self):
        """cost_delta_norm is non-negative."""
        events = self.emulator.run_sequence(5)
        for e in events:
            assert e.cost_delta_norm >= 0.0

    def test_callback_is_called(self):
        """Callback is invoked once per trigger with correct args."""
        call_log = []

        def my_callback(event, landscape):
            call_log.append((event.event_id, id(landscape)))

        self.emulator.run_sequence(4, callback=my_callback)
        assert len(call_log) == 4
        ids = [c[0] for c in call_log]
        assert ids == [1, 2, 3, 4]

    def test_run_sequence_returns_all_events(self):
        """run_sequence returns a list of exactly N events."""
        events = self.emulator.run_sequence(7)
        assert len(events) == 7
        assert all(isinstance(e, TriggerEvent) for e in events)

    def test_description_is_nonempty(self):
        """Event description is a non-empty string."""
        event = self.emulator.trigger()
        assert len(event.description) > 0

    def test_initial_timestamp_positive(self):
        """First trigger timestamp is positive (Poisson inter-arrival > 0)."""
        event = self.emulator.trigger()
        assert event.timestamp > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
