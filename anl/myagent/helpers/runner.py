"""
runner.py — Helper module for running MedellaANLAgent tests programmatically.

Provides functions to run quick, medium, and full test sessions against
built-in ANL 2025 competitors.

Usage:
    from myagent.helpers.runner import run_quick_test, run_medium_test

    results = run_quick_test()
    print(f"Center utility: {results.center_utility}")

Author: Umut Murat (umutmurat275@gmail.com)
Course: CS451 Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anl2025 import (
    run_generated_session,
    Boulware2025,
    Linear2025,
    Conceder2025,
    Random2025,
)
from anl2025.runner import SessionResults
from anl2025.scenarios import load_example_scenario

# Import our agent — use a relative import if running as part of the package,
# or an absolute one if running standalone.
try:
    from myagent.myagent import MedellaANLAgent
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from myagent.myagent import MedellaANLAgent


# ── Default competitors that our agent will face as edges ──
DEFAULT_EDGE_TYPES = [Boulware2025, Linear2025, Conceder2025, Random2025]


def run_quick_test(
    nedges: int = 3,
    nsteps: int = 50,
    verbose: bool = True,
) -> SessionResults:
    """
    Run a single quick generated session with MedellaANLAgent as center.

    This is useful for rapid iteration — runs in a few seconds.

    Args:
        nedges: Number of edge agents to negotiate with.
        nsteps: Number of negotiation rounds per thread.
        verbose: Print progress information.

    Returns:
        SessionResults with center_utility, edge_utilities, agreements, etc.
    """
    if verbose:
        print("=" * 60)
        print("QUICK TEST — Single generated session")
        print(f"  Center: MedellaANLAgent")
        print(f"  Edges:  {nedges} agents from {[t.__name__ for t in DEFAULT_EDGE_TYPES]}")
        print(f"  Steps:  {nsteps}")
        print("=" * 60)

    results = run_generated_session(
        center_type=MedellaANLAgent,
        edge_types=DEFAULT_EDGE_TYPES,
        nedges=nedges,
        nsteps=nsteps,
        verbose=verbose,
        output=None,  # don't save logs for quick test
    )

    if verbose:
        _print_session_results(results)

    return results


def run_medium_test(
    nedges: int = 5,
    nsteps: int = 100,
    n_sessions: int = 3,
    verbose: bool = True,
) -> list[SessionResults]:
    """
    Run several generated sessions to get a broader picture of performance.

    Args:
        nedges: Number of edge agents per session.
        nsteps: Number of negotiation rounds per thread.
        n_sessions: How many sessions to run.
        verbose: Print progress information.

    Returns:
        List of SessionResults from all sessions.
    """
    if verbose:
        print("=" * 60)
        print("MEDIUM TEST — Multiple generated sessions")
        print(f"  Center: MedellaANLAgent")
        print(f"  Sessions: {n_sessions}")
        print(f"  Edges per session: {nedges}")
        print(f"  Steps: {nsteps}")
        print("=" * 60)

    all_results = []
    for i in range(n_sessions):
        if verbose:
            print(f"\n--- Session {i + 1}/{n_sessions} ---")

        results = run_generated_session(
            center_type=MedellaANLAgent,
            edge_types=DEFAULT_EDGE_TYPES,
            nedges=nedges,
            nsteps=nsteps,
            verbose=False,
            output=None,
        )
        all_results.append(results)

        if verbose:
            _print_session_results(results)

    if verbose:
        # Print aggregate stats
        avg_center = sum(r.center_utility for r in all_results) / len(all_results)
        print(f"\n{'=' * 60}")
        print(f"AGGREGATE: Average center utility = {avg_center:.4f}")
        print(f"{'=' * 60}")

    return all_results


def run_full_test(
    nedges: int = 5,
    nsteps: int = 100,
    n_sessions: int = 10,
    verbose: bool = True,
) -> list[SessionResults]:
    """
    Run a comprehensive test with many sessions across generated scenarios.

    This takes longer but gives a statistically more reliable picture.

    Args:
        nedges: Number of edge agents per session.
        nsteps: Number of negotiation rounds per thread.
        n_sessions: How many sessions to run.
        verbose: Print progress information.

    Returns:
        List of SessionResults from all sessions.
    """
    if verbose:
        print("=" * 60)
        print("FULL TEST — Comprehensive evaluation")
        print(f"  Center: MedellaANLAgent")
        print(f"  Sessions: {n_sessions}")
        print(f"  Edges per session: {nedges}")
        print(f"  Steps: {nsteps}")
        print("=" * 60)

    all_results = []
    for i in range(n_sessions):
        if verbose:
            print(f"\n--- Session {i + 1}/{n_sessions} ---")

        results = run_generated_session(
            center_type=MedellaANLAgent,
            edge_types=DEFAULT_EDGE_TYPES,
            nedges=nedges,
            nsteps=nsteps,
            verbose=False,
            output=None,
        )
        all_results.append(results)

        if verbose:
            _print_session_results(results)

    if verbose:
        # Print comprehensive aggregate stats
        center_utils = [r.center_utility for r in all_results]
        avg_center = sum(center_utils) / len(center_utils)
        min_center = min(center_utils)
        max_center = max(center_utils)
        n_deals = sum(r.n_succeeded for r in all_results)
        n_total = sum(len(r.agreements) for r in all_results)

        print(f"\n{'=' * 60}")
        print(f"FULL TEST RESULTS")
        print(f"  Sessions run:     {len(all_results)}")
        print(f"  Avg center util:  {avg_center:.4f}")
        print(f"  Min center util:  {min_center:.4f}")
        print(f"  Max center util:  {max_center:.4f}")
        print(f"  Deals made:       {n_deals}/{n_total} ({100*n_deals/max(1,n_total):.1f}%)")
        print(f"{'=' * 60}")

    return all_results


def _print_session_results(results: SessionResults) -> None:
    """Pretty-print the results of a single session."""
    print(f"  Center utility: {results.center_utility:.4f}")
    print(f"  Edge utilities: {[f'{u:.4f}' for u in results.edge_utilities]}")
    n_agreed = sum(1 for a in results.agreements if a is not None)
    print(f"  Deals made:     {n_agreed}/{len(results.agreements)}")
    print(f"  Time:           {results.total_time:.2f}s")
