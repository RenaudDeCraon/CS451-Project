#!/usr/bin/env python3
"""
test_medella.py — Test runner for MedellaANLAgent in ANL 2025 tournaments.

Runs MedellaANLAgent against built-in competitors (Boulware2025, Linear2025,
Conceder2025, Random2025) and prints a scoreboard showing how it ranks.

Usage:
    python test_medella.py              # quick test  (1 scenario, 1 rep)
    python test_medella.py medium       # medium test (3 scenarios, 2 reps)
    python test_medella.py full         # full test   (5 scenarios, 3 reps)

Author: Umut Murat (umutmurat275@gmail.com)
Course: CS451 Project
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure our myagent package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anl2025 import (
    anl2025_tournament,
    Boulware2025,
    Linear2025,
    Conceder2025,
    Random2025,
)
from anl2025.tournament import Tournament, TournamentResults, scenario_maker
from anl2025.common import RunParams

from myagent.myagent import MedellaANLAgent


def run_tournament(mode: str = "quick") -> TournamentResults:
    """
    Run a tournament with MedellaANLAgent against built-in competitors.

    Args:
        mode: One of "quick", "medium", or "full".

    Returns:
        TournamentResults with scores for all agents.
    """
    # ── Configure based on mode ──
    if mode == "quick":
        n_scenarios = 1       # number of different negotiation scenarios
        n_reps = 1            # repetitions per scenario-rotation
        nedges = 3            # number of edge agents per scenario
        nsteps = 50           # negotiation rounds per thread
        n_jobs = None         # run serially (simpler for debugging)
    elif mode == "medium":
        n_scenarios = 3
        n_reps = 2
        nedges = 4
        nsteps = 100
        n_jobs = None
    elif mode == "full":
        n_scenarios = 5
        n_reps = 3
        nedges = 5
        nsteps = 100
        n_jobs = None
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'quick', 'medium', or 'full'.")

    # ── Define competitors ──
    # MedellaANLAgent competes against the 4 built-in ANL 2025 baselines.
    competitors = (
        MedellaANLAgent,
        Boulware2025,
        Linear2025,
        Conceder2025,
        Random2025,
    )

    print("=" * 70)
    print(f"  ANL 2025 TOURNAMENT — {mode.upper()} MODE")
    print(f"  Competitors: {[c.__name__ for c in competitors]}")
    print(f"  Scenarios: {n_scenarios} | Repetitions: {n_reps}")
    print(f"  Edges: {nedges} | Steps: {nsteps}")
    print("=" * 70)
    print()

    # ── Generate scenarios ──
    # We use the scenario_maker which randomly picks from different scenario
    # types (dinners, job_hunt, target_quantity, random). This gives a diverse
    # set of challenges to test our agent on.
    scenarios = tuple(
        scenario_maker(nedges=nedges, nvalues=7, nissues=3)
        for _ in range(n_scenarios)
    )

    # ── Run the tournament ──
    results = anl2025_tournament(
        scenarios=scenarios,
        competitors=competitors,
        n_repetitions=n_reps,
        n_steps=nsteps,
        n_jobs=n_jobs,
        verbose=False,
        method="sequential",  # ANL 2025 uses sequential (finish one thread before next)
    )

    # ── Print the scoreboard ──
    print_scoreboard(results)

    return results


def print_scoreboard(results: TournamentResults) -> None:
    """
    Print a formatted scoreboard from tournament results.

    Shows each agent's accumulated score, weighted average, center/edge
    breakdown, and how many times they played in each role.
    """
    print()
    print("=" * 70)
    print("  SCOREBOARD")
    print("=" * 70)

    # Sort agents by weighted_average (the fairest metric since it normalizes
    # for how many times each agent played as center vs edge)
    sorted_agents = sorted(
        results.weighted_average.keys(),
        key=lambda a: results.weighted_average[a],
        reverse=True,
    )

    # Header
    print(f"{'Rank':<6}{'Agent':<30}{'Weighted Avg':<15}{'Total Score':<15}"
          f"{'Center Avg':<15}{'Edge Avg':<15}")
    print("-" * 96)

    for rank, agent in enumerate(sorted_agents, 1):
        # Shorten the agent name for display (remove module path prefix)
        short_name = agent.split(".")[-1]

        weighted_avg = results.weighted_average.get(agent, 0.0)
        total_score = results.final_scores.get(agent, 0.0)

        # Calculate center and edge averages
        center_count = results.center_count.get(agent, 0)
        edge_count = results.edge_count.get(agent, 0)
        center_total = results.final_scoresC.get(agent, 0.0)
        edge_total = results.final_scoresE.get(agent, 0.0)

        center_avg = center_total / center_count if center_count > 0 else 0.0
        edge_avg = edge_total / edge_count if edge_count > 0 else 0.0

        # Highlight our agent with an arrow
        marker = " <--" if "Medella" in short_name else ""

        print(f"  {rank:<4}{short_name:<30}{weighted_avg:<15.4f}{total_score:<15.4f}"
              f"{center_avg:<15.4f}{edge_avg:<15.4f}{marker}")

    # Summary stats
    print()
    print(f"  Threads succeeded: {results.n_threads_succeeded}")
    print(f"  Threads timed out: {results.n_threads_timedout}")
    print(f"  Threads failed:    {results.n_threads_failed}")
    print("=" * 70)


if __name__ == "__main__":
    # Parse command-line argument for mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "quick"

    try:
        results = run_tournament(mode)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
