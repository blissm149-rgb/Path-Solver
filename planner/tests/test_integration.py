"""Integration test for main.py — Phase 6 validation."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from planner.main import run


class TestIntegration:
    """Run full loop and verify outputs exist and candidates are non-empty."""

    def test_run_single_event(self, tmp_path):
        """Run full pipeline for 1 trigger event."""
        output_dir = str(tmp_path / 'output')
        all_candidates = run(
            n_events=1,
            seed_landscape=42,
            seed_emulator=123,
            output_dir=output_dir,
        )

        # Should have 2 entries: initial + 1 event
        assert len(all_candidates) == 2, f"Expected 2 events, got {len(all_candidates)}"

        # Initial event should have candidates
        initial = all_candidates[0]
        assert len(initial) > 0, "Initial event produced no candidates"

        # Each candidate should have required fields
        required = {'label', 'x', 'y', 'theta', 'v', 'integrated_cost',
                    'peak_cost', 'total_length', 'total_time', 'feasible',
                    'dist_to_goal', 'seed_index'}
        for cand in initial:
            assert required.issubset(set(cand.keys()))

        # PNG outputs should exist
        assert os.path.exists(os.path.join(output_dir, 'event_00_initial.png')), \
            "Initial PNG not created"
        assert os.path.exists(os.path.join(output_dir, 'event_01.png')), \
            "Event 01 PNG not created"

    def test_png_files_non_empty(self, tmp_path):
        """Generated PNGs should be non-empty files."""
        output_dir = str(tmp_path / 'output')
        run(n_events=1, seed_landscape=0, seed_emulator=0, output_dir=output_dir)

        for fname in ['event_00_initial.png', 'event_01.png']:
            fpath = os.path.join(output_dir, fname)
            assert os.path.exists(fpath), f"{fname} not found"
            assert os.path.getsize(fpath) > 10_000, f"{fname} is suspiciously small"

    def test_candidates_labeled_alphabetically(self, tmp_path):
        """Candidates are labeled Plan A, Plan B, ..."""
        output_dir = str(tmp_path / 'output')
        all_candidates = run(n_events=1, seed_landscape=5, seed_emulator=5,
                             output_dir=output_dir)
        for event_candidates in all_candidates:
            for i, cand in enumerate(event_candidates):
                expected = f"Plan {chr(ord('A') + i)}"
                assert cand['label'] == expected, \
                    f"Expected '{expected}', got '{cand['label']}'"

    def test_at_least_one_feasible_plan_initial(self, tmp_path):
        """At least one plan for the initial event should be feasible."""
        output_dir = str(tmp_path / 'output')
        all_candidates = run(n_events=1, seed_landscape=42, seed_emulator=42,
                             output_dir=output_dir)
        # Initial event (index 0) on the unmodified landscape should yield ≥1 feasible plan.
        # Subsequent trigger events may have all infeasible plans when the OU-evolved
        # landscape forces large detours that violate the forward-progress constraint —
        # this is the intended "no monotonic-progress path exists" signal.
        initial_candidates = all_candidates[0]
        assert len(initial_candidates) > 0, "Initial event produced no candidates"
        feasible = [c for c in initial_candidates if c['feasible']]
        assert len(feasible) >= 1, "No feasible plan found for initial event"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
