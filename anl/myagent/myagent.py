"""
MedellaANLAgent — A competitive agent for the ANL 2025 (Automated Negotiation League).

This agent plays as the CENTER in a sequential multi-deal negotiation. It negotiates
with N edge agents one at a time, and its final utility is evaluated over the
COMBINATION of all deals (via the CenterUFun). This means what we accept in thread 1
directly affects what's optimal in thread 2, 3, ..., N.

Strategy Overview:
    1. Aspiration-based concession — start tough, concede toward the deadline using
       exponential decay. The exponent controls how "boulware" vs "conceder" we are.
    2. SideUFun-based offer selection — we use the inverse utility function (presorted)
       to efficiently find offers at our current aspiration level.
    3. Adaptive acceptance — accept if the opponent's offer meets our decaying threshold,
       AND check that it's above our reservation value.
    4. Cross-thread coordination — after each thread finalizes, we call
       set_expected_outcome() so that future threads' SideUFuns account for what was
       already agreed. This is the KEY challenge in ANL 2025.
    5. Opponent modeling — we track the opponent's concession pattern to estimate
       whether they'll concede further or if we should settle now.

Author: Umut Murat (umutmurat275@gmail.com)
Course: CS451 Project
"""

from __future__ import annotations

import math
from collections import defaultdict
from random import choice

from negmas import (
    InverseUFun,
    PresortingInverseUtilityFunction,
    ResponseType,
)
from negmas.outcomes import Outcome
from negmas.sao.controllers import SAOState

# ANL 2025 imports
from anl2025 import ANL2025Negotiator
from anl2025.ufun import CenterUFun, SideUFun


class MedellaANLAgent(ANL2025Negotiator):
    """
    A competitive ANL 2025 center agent using aspiration-based concession,
    cross-thread coordination, and lightweight opponent modeling.

    Key design decisions:
        - We use a "boulware" aspiration curve (exponent > 1) so we hold out for
          good deals early and only concede near the deadline.
        - We rebuild the inverse utility function after each thread finalizes,
          because the SideUFun changes when expected outcomes are updated.
        - We track opponent offers to detect their concession trend and adjust
          our acceptance threshold accordingly.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Configurable parameters (class-level defaults)
    # ──────────────────────────────────────────────────────────────────────

    # Aspiration exponent: >1 = boulware (holds out), <1 = conceder (gives in fast)
    # 3.0 is a moderate boulware — tough but not so aggressive that we miss deals.
    # The key insight: in ANL 2025, the CenterUFun evaluates ALL deals together,
    # so making deals (even slightly discounted) is usually better than no deal.
    ASPIRATION_EXPONENT: float = 3.0

    # When searching for offers at a given utility level, we try increasingly
    # wide bands. These deltas define how much we widen each attempt.
    SEARCH_DELTAS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)

    # How much weight to give the opponent model when adjusting our threshold.
    # 0.0 = ignore opponent model, 1.0 = fully trust it.
    OPPONENT_MODEL_WEIGHT: float = 0.15

    # Number of recent opponent offers to consider for trend estimation.
    OPPONENT_WINDOW_SIZE: int = 5

    # Adaptive exponent range for cross-thread strategy:
    # On later threads (when we already have some deals locked in), we can
    # afford to be more flexible because the marginal value of each additional
    # deal may be lower. We interpolate the exponent between these bounds
    # based on how well previous threads went.
    EARLY_THREAD_EXPONENT: float = 4.0   # more boulware on first threads
    LATE_THREAD_EXPONENT: float = 2.0    # more conceding on later threads

    def __init__(self, *args, **kwargs):
        # We set update_side_ufuns_on_end=True so that after each thread ends,
        # the base class automatically calls set_expected_outcome with the
        # agreement (or None if no deal). This is CRITICAL for cross-thread
        # coordination — it tells future SideUFuns what was already locked in.
        super().__init__(
            *args,
            update_side_ufuns_on_end=True,
            update_side_ufuns_after_offering=False,
            update_side_ufuns_after_receiving_offers=False,
            **kwargs,
        )

        # ── Per-thread state (populated in thread_init) ──

        # Maps negotiator_id -> PresortingInverseUtilityFunction
        # The inverter lets us efficiently find outcomes at a target utility level
        self._inverters: dict[str, InverseUFun] = {}

        # Maps negotiator_id -> (max_utility, min_utility) for the SideUFun
        self._utility_ranges: dict[str, tuple[float, float]] = {}

        # Maps negotiator_id -> list of best outcomes (at or near max utility)
        self._best_outcomes: dict[str, list[Outcome]] = {}

        # ── Opponent modeling state ──

        # Maps negotiator_id -> list of (step, utility_of_their_offer)
        # We track the utility (from OUR perspective) of what they offer us
        self._opponent_history: dict[str, list[tuple[int, float]]] = defaultdict(list)

        # ── Cross-thread tracking ──

        # How many threads have been completed so far
        self._threads_completed: int = 0

        # Total number of threads (set in init())
        self._total_threads: int = 0

        # Track agreements from completed threads for logging/debugging
        self._agreements: list[Outcome | None] = []

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle methods
    # ──────────────────────────────────────────────────────────────────────

    def init(self):
        """
        Called once after ALL negotiation threads (mechanisms) are set up.

        At this point we can access:
            - self.ufun: the CenterUFun that evaluates the FULL tuple of deals
            - self.negotiators: dict mapping negotiator_id -> (negotiator, context)
              where context has 'ufun' (SideUFun), 'index', and 'center' (bool)
        """
        # Count how many threads we'll negotiate
        self._total_threads = len(self.negotiators)

    def thread_init(self, negotiator_id: str, state: SAOState) -> None:
        """
        Called when a specific negotiation thread STARTS.

        This is where we set up the utility inverter for this thread's SideUFun.
        We rebuild it fresh each time because the SideUFun changes as previous
        threads' outcomes get locked in via set_expected_outcome.

        Args:
            negotiator_id: The ID linking us to this negotiation thread.
            state: The initial state of this negotiation thread.
        """
        # Build the inverter for this thread (handles all the setup)
        self._build_inverter(negotiator_id)

        # Clear opponent history for this new thread
        self._opponent_history[negotiator_id] = []

    def thread_finalize(self, negotiator_id: str, state: SAOState) -> None:
        """
        Called when a negotiation thread ENDS (deal reached or deadline passed).

        This is where we:
            1. Record what happened (agreement or None)
            2. Clean up the inverter (it's no longer valid after expected outcomes change)
            3. Update our thread counter

        Note: The base class already calls set_expected_outcome() for us because
        we set update_side_ufuns_on_end=True. So future SideUFuns will automatically
        account for whatever was agreed (or not) in this thread.

        Args:
            negotiator_id: The ID of the thread that just ended.
            state: The final state of the negotiation.
        """
        # Record the agreement (or None if no deal was made)
        self._agreements.append(state.agreement)
        self._threads_completed += 1

        # Invalidate ALL cached inverters because the expected outcomes changed.
        # Any remaining threads need fresh inverters built from the updated SideUFuns.
        self._inverters.clear()
        self._utility_ranges.clear()
        self._best_outcomes.clear()

    # ──────────────────────────────────────────────────────────────────────
    # Core negotiation: propose
    # ──────────────────────────────────────────────────────────────────────

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """
        Called when it's our turn to make an offer on a specific thread.

        Strategy:
            1. Calculate our current aspiration level (high early, low near deadline)
            2. Use the inverse utility function to find an outcome at that level
            3. If we can't find one exactly, widen the search band incrementally

        Args:
            negotiator_id: Which thread we're proposing on.
            state: Current negotiation state (step, relative_time, etc.)
            dest: The ID of the edge agent we're negotiating with.

        Returns:
            An Outcome (tuple of issue values) to propose, or None to end negotiation.
        """
        # Make sure we have an inverter for this thread (may need rebuilding
        # if a previous thread just finalized and changed the expected outcomes)
        self._ensure_inverter(negotiator_id)

        # Get the SideUFun and check if there are ANY rational outcomes
        _, cntxt = self.negotiators[negotiator_id]
        side_ufun: SideUFun = cntxt["ufun"]
        mx, mn = self._utility_ranges[negotiator_id]

        # If even the best outcome is worse than disagreement, end negotiation.
        # This can happen when previous deals make ANY outcome on this thread bad.
        reserved = float(side_ufun(None))
        if mx < reserved:
            return None

        # Calculate our target utility level using aspiration-based concession.
        # This returns a normalized value in [0, 1].
        nmi = self.negotiators[negotiator_id].negotiator.nmi
        target_level = self._calc_aspiration_level(nmi, state)

        # Search for an outcome at or near our target level.
        # We try increasingly wide bands to ensure we find SOMETHING.
        # The inverter works with normalized utilities (0 = worst, 1 = best).
        inverter = self._inverters[negotiator_id]
        outcome = None
        for delta in self.SEARCH_DELTAS:
            upper = min(1.0, target_level + delta)
            outcome = inverter.one_in((target_level, upper), normalized=True)
            if outcome is not None:
                break

        # If we still couldn't find anything (very rare), fall back to our
        # pre-computed best outcomes
        if outcome is None:
            outcome = choice(self._best_outcomes[negotiator_id])

        return outcome

    # ──────────────────────────────────────────────────────────────────────
    # Core negotiation: respond
    # ──────────────────────────────────────────────────────────────────────

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        """
        Called when the opponent makes an offer and we need to accept or reject.

        Strategy:
            1. Calculate our aspiration-based threshold (decays over time)
            2. Convert it to a raw utility value using the thread's utility range
            3. Adjust threshold using opponent modeling (if they're conceding fast,
               we can afford to wait; if they're stuck, maybe settle)
            4. Accept if the offer's SideUFun utility >= adjusted threshold

        Args:
            negotiator_id: Which thread we're responding on.
            state: Current state including state.current_offer.
            source: The ID of the edge agent who made the offer.

        Returns:
            ResponseType.ACCEPT_OFFER, REJECT_OFFER, or END_NEGOTIATION
        """
        # Make sure we have an inverter (for utility range info)
        self._ensure_inverter(negotiator_id)

        # Get the SideUFun for this thread
        _, cntxt = self.negotiators[negotiator_id]
        side_ufun: SideUFun = cntxt["ufun"]
        mx, mn = self._utility_ranges[negotiator_id]

        # If there are no rational outcomes at all, end the negotiation.
        # This means even the BEST deal on this thread is worse than no deal.
        reserved = float(side_ufun(None))
        if mx < reserved:
            return ResponseType.END_NEGOTIATION

        # Evaluate the opponent's offer using our SideUFun.
        # The SideUFun automatically accounts for expected outcomes on other threads.
        offer = state.current_offer
        offer_utility = float(side_ufun(offer))

        # Record this offer for opponent modeling
        self._opponent_history[negotiator_id].append((state.step, offer_utility))

        # Calculate our aspiration-based threshold as a raw utility value.
        # normalized_level is in [0, 1], we scale it to [mn, mx].
        nmi = self.negotiators[negotiator_id].negotiator.nmi
        normalized_level = self._calc_aspiration_level(nmi, state)

        # Scale: threshold = mn + level * (mx - mn)
        # At level=1.0 (start of negotiation), threshold = mx (demand the best)
        # At level=0.0 (deadline), threshold = mn (accept anything rational)
        threshold = mn + normalized_level * (mx - mn)

        # Adjust threshold using opponent modeling
        threshold = self._adjust_threshold_with_opponent_model(
            negotiator_id, threshold, mn, mx
        )

        # Accept if the offer meets our (possibly adjusted) threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    # ──────────────────────────────────────────────────────────────────────
    # Aspiration curve
    # ──────────────────────────────────────────────────────────────────────

    def _get_adaptive_exponent(self) -> float:
        """
        Compute the aspiration exponent based on our thread position.

        Early threads (when we have no deals yet) use a higher exponent (more
        boulware) because we need to secure good initial deals. Later threads
        use a lower exponent (more conceding) because:
            - We already have some deals locked in
            - The marginal value of additional deals may diminish
            - It's better to make a mediocre deal than no deal at all

        Returns:
            The exponent to use for the aspiration curve.
        """
        if self._total_threads <= 1:
            return self.ASPIRATION_EXPONENT

        # How far through the sequence of threads are we? (0.0 = first, 1.0 = last)
        thread_progress = self._threads_completed / max(1, self._total_threads - 1)

        # Interpolate between early (tough) and late (flexible) exponents
        exponent = (
            self.EARLY_THREAD_EXPONENT
            + thread_progress * (self.LATE_THREAD_EXPONENT - self.EARLY_THREAD_EXPONENT)
        )

        return exponent

    def _calc_aspiration_level(self, nmi, state: SAOState) -> float:
        """
        Calculate the current aspiration level using exponential time-based decay.

        The aspiration starts at 1.0 (demand the best) and decays toward 0.0
        (accept anything rational) as we approach the deadline.

        The formula is:
            level = 1.0 - t^(1/e)
        where:
            t = relative time (0.0 at start, 1.0 at deadline)
            e = aspiration exponent (adaptive based on thread position)

        The exponent adapts across threads:
            - Thread 1 (first): e=4.0 (boulware, hold out for good deals)
            - Thread N (last):  e=2.0 (more conceding, secure the deal)

        Args:
            nmi: Negotiator-Mechanism Interface (has n_steps, etc.)
            state: Current negotiation state.

        Returns:
            Normalized aspiration level in [0.0, 1.0].
        """
        # At the very first step, demand the best
        if state.step == 0:
            return 1.0

        # At the very last step, accept anything rational
        if nmi.n_steps is not None and state.step >= nmi.n_steps - 1:
            return 0.0

        # Use relative_time which goes from 0.0 to 1.0
        t = state.relative_time

        # Get the adaptive exponent based on which thread we're on
        exponent = self._get_adaptive_exponent()

        # Aspiration formula: level = 1 - t^(1/exponent)
        level = 1.0 - math.pow(t, 1.0 / exponent)

        return level

    # ──────────────────────────────────────────────────────────────────────
    # Opponent modeling
    # ──────────────────────────────────────────────────────────────────────

    def _estimate_opponent_trend(self, negotiator_id: str) -> float:
        """
        Estimate how fast the opponent is conceding by looking at recent offers.

        We fit a simple linear trend to the last OPPONENT_WINDOW_SIZE offers
        (measured in OUR utility). A positive trend means the opponent is
        conceding (their offers are getting better for us). A negative trend
        means they're hardening (offers getting worse for us).

        Returns:
            A value roughly in [-1, 1] indicating the opponent's concession rate.
            Positive = they're conceding toward us.
            Negative = they're hardening or being erratic.
            0.0 = not enough data or flat.
        """
        history = self._opponent_history.get(negotiator_id, [])

        # Need at least 2 data points to estimate a trend
        if len(history) < 2:
            return 0.0

        # Take the last N offers
        recent = history[-self.OPPONENT_WINDOW_SIZE:]

        # Simple linear regression: slope of utility vs step
        n = len(recent)
        steps = [h[0] for h in recent]
        utils = [h[1] for h in recent]

        # Calculate slope using least squares formula
        mean_step = sum(steps) / n
        mean_util = sum(utils) / n

        numerator = sum(
            (s - mean_step) * (u - mean_util) for s, u in zip(steps, utils)
        )
        denominator = sum((s - mean_step) ** 2 for s in steps)

        # Avoid division by zero (happens if all steps are the same)
        if abs(denominator) < 1e-10:
            return 0.0

        slope = numerator / denominator

        # Normalize the slope to roughly [-1, 1] range.
        # A slope of 0.01 per step is already significant concession.
        normalized_slope = max(-1.0, min(1.0, slope * 50.0))

        return normalized_slope

    def _adjust_threshold_with_opponent_model(
        self,
        negotiator_id: str,
        base_threshold: float,
        mn: float,
        mx: float,
    ) -> float:
        """
        Adjust our acceptance threshold based on opponent behavior.

        If the opponent is conceding fast (positive trend), we can afford to be
        MORE demanding (raise our threshold slightly) because they'll likely
        give us even better offers soon.

        If the opponent is stuck or hardening (negative trend), we should be
        MORE willing to accept (lower our threshold slightly) because waiting
        won't help.

        Args:
            negotiator_id: Which thread.
            base_threshold: Our aspiration-based threshold before adjustment.
            mn: Minimum rational utility on this thread.
            mx: Maximum utility on this thread.

        Returns:
            The adjusted threshold (clamped to [mn, mx]).
        """
        trend = self._estimate_opponent_trend(negotiator_id)

        # Scale the adjustment by the utility range and our weight parameter.
        # Positive trend (opponent conceding) -> increase threshold (be greedier)
        # Negative trend (opponent stuck) -> decrease threshold (be more flexible)
        adjustment = trend * self.OPPONENT_MODEL_WEIGHT * (mx - mn)

        adjusted = base_threshold + adjustment

        # Clamp to [mn, mx] — never go below reservation or above maximum
        adjusted = max(mn, min(mx, adjusted))

        return adjusted

    # ──────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_inverter(self, negotiator_id: str) -> None:
        """
        Build a fresh PresortingInverseUtilityFunction for a thread.

        The PresortingInverseUtilityFunction pre-sorts all possible outcomes by
        their utility value. This lets us efficiently find an outcome at any
        target utility level in O(log n) time instead of searching through all
        outcomes each time.

        Args:
            negotiator_id: The thread to build the inverter for.
        """
        # Get the SideUFun for this thread from the context dict
        _, cntxt = self.negotiators[negotiator_id]
        side_ufun: SideUFun = cntxt["ufun"]

        # Build and initialize the presorted inverter
        inverter = PresortingInverseUtilityFunction(side_ufun, rational_only=True)
        inverter.init()

        # Get the utility range for this thread
        mx = inverter.max()  # best possible utility from this SideUFun
        mn = inverter.min()  # worst possible utility

        # The reservation value is what we get if NO deal is made on this thread.
        # We should never accept an offer worse than this.
        reserved = float(side_ufun(None))
        mn = max(mn, reserved)

        # Store everything
        self._inverters[negotiator_id] = inverter
        self._utility_ranges[negotiator_id] = (mx, mn)

        # Pre-compute the best outcomes (within a tiny margin of the max).
        # We'll fall back to these if our search at a specific level fails.
        best_margin = 1e-8
        best_outcomes = inverter.some(
            (max(0.0, mn, reserved, mx - best_margin), mx),
            normalized=True,
        )
        if not best_outcomes:
            best_outcomes = [inverter.best()]
        self._best_outcomes[negotiator_id] = best_outcomes

    def _ensure_inverter(self, negotiator_id: str) -> None:
        """
        Make sure we have a valid inverter for this thread.

        The inverter may have been invalidated when a previous thread finalized
        (because set_expected_outcome changes the SideUFun). If so, we rebuild it.

        Args:
            negotiator_id: The thread to ensure an inverter for.
        """
        if negotiator_id not in self._inverters:
            self._build_inverter(negotiator_id)
