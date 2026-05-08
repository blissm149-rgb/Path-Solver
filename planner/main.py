"""
main.py — Orchestrator: wire everything together, run trigger loop, produce visualizations.
"""

from __future__ import annotations

import os
import sys
import warnings
from math import atan2

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np

from .cost_landscape import CostLandscape, CostLandscapeConfig
from .event_emulator import EventEmulator, TriggerEvent
from .vehicle_model import VehicleModel, PlanningProblem
from .graph_search import generate_diverse_seeds, resample_path
from .optimizer import generate_candidate_plans

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOMAIN = (0.0, 100.0, 0.0, 100.0)
VEHICLE = VehicleModel(
    length=5.0, width=2.5, v_max=15.0, v_min=0.5,
    a_max=3.0, a_lat_max=2.0, kappa_max=0.15,
)
START = np.array([10.0, 10.0])
GOAL = np.array([90.0, 90.0])
GOAL_RADIUS = 5.0
INITIAL_HEADING = atan2(GOAL[1] - START[1], GOAL[0] - START[0])  # ~π/4
INITIAL_SPEED = 5.0
N_WAYPOINTS = 20
N_DIVERSE_SEEDS = 3
N_EVENTS = 3

# Dark theme colors
DARK_BG = '#1a1a2e'
TEXT_COLOR = 'white'
PATH_COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


# ---------------------------------------------------------------------------
# OBB drawing helper
# ---------------------------------------------------------------------------

def _draw_obb(ax, cx, cy, theta, hl, hw, color, alpha=0.3, lw=1.0):
    """Draw an oriented bounding box on the given axes."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    corners_local = np.array([
        [ hl,  hw], [ hl, -hw], [-hl, -hw], [-hl,  hw], [ hl,  hw],
    ])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners_world = corners_local @ R.T
    corners_world[:, 0] += cx
    corners_world[:, 1] += cy
    ax.plot(corners_world[:, 0], corners_world[:, 1],
            color=color, alpha=alpha, lw=lw)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def render_event(
    landscape: CostLandscape,
    candidates: list[dict],
    event_label: str,
    output_path: str,
) -> None:
    """Render a two-panel figure for this planning event and save as PNG."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 8), facecolor=DARK_BG)

    # Grid layout: left 2/3 for cost map, right 1/3 for info
    gs = fig.add_gridspec(
        2, 3,
        left=0.04, right=0.97, top=0.93, bottom=0.07,
        wspace=0.35, hspace=0.45,
    )
    ax_map = fig.add_subplot(gs[:, :2])
    ax_table = fig.add_subplot(gs[0, 2])
    ax_speed = fig.add_subplot(gs[1, 2])

    # --- Cost field heatmap ---
    X, Y, C = landscape.to_grid(resolution=200)
    C_display = np.log1p(np.clip(C, 0, None))
    im = ax_map.pcolormesh(X, Y, C_display, cmap='inferno', shading='auto')
    plt.colorbar(im, ax=ax_map, label='ln(1 + Cost)', shrink=0.7)

    # Obstacle contours
    ax_map.contour(X, Y, C, levels=[100.0], colors=['red'],
                   linestyles='dashed', linewidths=1.5, alpha=0.8)

    # Start marker
    ax_map.scatter(*START, marker='s', s=120, color='#00ff88', zorder=10,
                   label='Start', edgecolors='white', linewidths=1)

    # Goal disk
    goal_circle = Circle(GOAL, GOAL_RADIUS, fill=False, edgecolor='#00ff88',
                         linestyle='--', linewidth=2)
    ax_map.add_patch(goal_circle)
    ax_map.scatter(*GOAL, marker='*', s=150, color='#00ff88', zorder=10, label='Goal')

    # --- Plot candidate paths ---
    hl, hw = VEHICLE.half_length, VEHICLE.half_width
    legend_handles = []

    for i, cand in enumerate(candidates):
        color = PATH_COLORS[i % len(PATH_COLORS)]
        x, y, theta = cand['x'], cand['y'], cand['theta']
        n = len(x)
        label_str = cand['label']
        if not cand['feasible']:
            label_str += ' [infeasible]'
        ls = '-' if cand['feasible'] else '--'
        line, = ax_map.plot(x, y, color=color, lw=2, linestyle=ls, alpha=0.9, zorder=5)
        legend_handles.append(mpatches.Patch(color=color, label=label_str))

        # OBB footprints at every ~6th waypoint
        for j in range(0, n, 6):
            _draw_obb(ax_map, x[j], y[j], theta[j], hl, hw, color, alpha=0.4, lw=0.8)

        # Direction arrow at midpoint
        mid = n // 2
        if mid + 1 < n:
            dx = x[mid + 1] - x[mid]
            dy = y[mid + 1] - y[mid]
            ax_map.annotate('', xy=(x[mid] + dx * 2, y[mid] + dy * 2),
                            xytext=(x[mid], y[mid]),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=2, mutation_scale=15))

    ax_map.set_xlim(DOMAIN[0], DOMAIN[1])
    ax_map.set_ylim(DOMAIN[2], DOMAIN[3])
    ax_map.set_aspect('equal')
    ax_map.set_title(f'{event_label}', color=TEXT_COLOR, fontsize=13)
    ax_map.set_xlabel('X [m]', color=TEXT_COLOR)
    ax_map.set_ylabel('Y [m]', color=TEXT_COLOR)
    ax_map.legend(handles=legend_handles, loc='lower right',
                  facecolor='#16213e', edgecolor='white', labelcolor='white',
                  fontsize=8)
    ax_map.tick_params(colors=TEXT_COLOR)
    for spine in ax_map.spines.values():
        spine.set_edgecolor(TEXT_COLOR)

    # --- Right panel: comparison table ---
    ax_table.set_facecolor(DARK_BG)
    ax_table.axis('off')

    if candidates:
        col_labels = ['Plan', 'Int.Cost', 'PkCost', 'Len[m]', 'Time[s]', 'OK?']
        table_data = []
        for cand in candidates:
            table_data.append([
                cand['label'],
                f"{cand['integrated_cost']:.1f}",
                f"{cand['peak_cost']:.1f}",
                f"{cand['total_length']:.1f}",
                f"{cand['total_time']:.1f}",
                '✓' if cand['feasible'] else '✗',
            ])
        tbl = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_facecolor('#16213e')
            cell.set_edgecolor('#444466')
            cell.set_text_props(color=TEXT_COLOR)
        ax_table.set_title('Plan Comparison', color=TEXT_COLOR, fontsize=10, pad=8)

    # --- Right panel: speed profile ---
    ax_speed.set_facecolor(DARK_BG)

    for i, cand in enumerate(candidates):
        color = PATH_COLORS[i % len(PATH_COLORS)]
        x, y, v = cand['x'], cand['y'], cand['v']
        ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        arc = np.concatenate([[0.0], np.cumsum(ds)])
        ls = '-' if cand['feasible'] else '--'
        ax_speed.plot(arc, v, color=color, lw=1.5, linestyle=ls, label=cand['label'])

    ax_speed.set_xlabel('Arc length [m]', color=TEXT_COLOR, fontsize=8)
    ax_speed.set_ylabel('Speed [m/s]', color=TEXT_COLOR, fontsize=8)
    ax_speed.set_title('Speed Profiles', color=TEXT_COLOR, fontsize=10)
    ax_speed.tick_params(colors=TEXT_COLOR)
    ax_speed.legend(facecolor='#16213e', edgecolor='white', labelcolor='white', fontsize=7)
    for spine in ax_speed.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    ax_speed.set_facecolor(DARK_BG)

    fig.patch.set_facecolor(DARK_BG)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary(event_label: str, candidates: list[dict]) -> None:
    """Print plan comparison table to stdout."""
    print(f"\n{'='*70}")
    print(f"  {event_label}")
    print(f"{'='*70}")
    if not candidates:
        print("  No feasible paths found.")
        return
    header = f"  {'Label':<10} {'IntCost':>8} {'PkCost':>8} {'Len[m]':>8} {'Time[s]':>8} {'Feasible':>9}"
    print(header)
    print(f"  {'-'*60}")
    for c in candidates:
        status = 'YES' if c['feasible'] else 'NO'
        print(
            f"  {c['label']:<10} {c['integrated_cost']:>8.2f} {c['peak_cost']:>8.1f} "
            f"{c['total_length']:>8.1f} {c['total_time']:>8.1f} {status:>9}"
        )
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    n_events: int = N_EVENTS,
    seed_landscape: int = 42,
    seed_emulator: int = 123,
    output_dir: str = OUTPUT_DIR,
) -> list[list[dict]]:
    """
    Run the full trigger loop.

    Returns:
        List of candidate lists (one per event, including initial).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    cfg = CostLandscapeConfig(seed=seed_landscape)
    landscape = CostLandscape(cfg)
    emulator = EventEmulator(landscape, mean_interval=8.0, seed=seed_emulator)

    problem = PlanningProblem(
        vehicle=VEHICLE,
        x_init=float(START[0]), y_init=float(START[1]),
        theta_init=INITIAL_HEADING, v_init=INITIAL_SPEED,
        x_goal=float(GOAL[0]), y_goal=float(GOAL[1]),
        goal_radius=GOAL_RADIUS,
        n_waypoints=N_WAYPOINTS,
    )

    all_candidates = []

    def process_event(event_label: str, output_filename: str) -> list[dict]:
        """Run seed generation, optimization, visualization for one event."""
        print(f"\n[{event_label}] Generating diverse seeds...")
        raw_seeds = generate_diverse_seeds(
            landscape, START, GOAL,
            n_seeds=N_DIVERSE_SEEDS,
            grid_resolution=80,
        )

        if not raw_seeds:
            print(f"  WARNING: No A* seeds found. All corridors may be blocked.")
            return []

        seeds = [resample_path(s, N_WAYPOINTS) for s in raw_seeds]
        print(f"  Found {len(seeds)} seed(s). Optimizing...")

        candidates = generate_candidate_plans(landscape, problem, seeds, maxiter=100)

        out_path = os.path.join(output_dir, output_filename)
        render_event(landscape, candidates, event_label, out_path)
        print(f"  Saved visualization: {output_filename}")

        print_summary(event_label, candidates)
        return candidates

    # Initial event (before any emulator triggers)
    candidates0 = process_event("Event 00 — Initial", "event_00_initial.png")
    all_candidates.append(candidates0)

    # Emulator-driven events
    def on_trigger(event: TriggerEvent, _landscape: CostLandscape) -> None:
        print(f"\n  >> Trigger: {event.description}")

    events = emulator.run_sequence(n_events, callback=on_trigger)

    for idx, event in enumerate(events, start=1):
        event_label = f"Event {idx:02d} — t={event.timestamp:.1f}s"
        output_filename = f"event_{idx:02d}.png"
        candidates = process_event(event_label, output_filename)
        all_candidates.append(candidates)

    print(f"\nDone. Outputs in: {output_dir}")
    return all_candidates


if __name__ == '__main__':
    run()
