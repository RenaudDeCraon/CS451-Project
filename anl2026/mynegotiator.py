"""
MedellaANLAgent — A competitive agent for ANL 2026 (Automated Negotiation League).

ANL 2026 is a bilateral negotiation with a DECEPTION challenge. The agent's final
score has two components:
    Score = Advantage + Concealing
where:
    Advantage = (utility_of_agreement - reservation_value) / (max_utility - reservation_value)
    Concealing = how well we MISLEAD the opponent's model of our utility function
                 (measured by Kendall rank correlation — lower correlation = higher concealing score)

Strategy Overview:
    1. Aspiration-based bidding — start tough, concede over time using exponential decay
    2. Opponent preference modeling — frequency-based model to estimate opponent's utility
    3. Adaptive acceptance — accept if offer meets our decaying threshold
    4. Deception via concealing bidding — strategically offer bids that mislead the
       opponent's model of our preferences while still being rational for us
    5. Opponent behavior tracking — detect concession patterns to adapt timing

Author: Umut Murat (umutmurat275@gmail.com)
Course: CS451 Project — ANAC 2026
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from negmas.outcomes import Outcome
from negmas.preferences import LinearMultiFun
from negmas.sao import SAOCallNegotiator, SAOResponse, SAOState, ResponseType


class MyNegotiator(SAOCallNegotiator):
    """
    MedellaANLAgent for ANL 2026 — bilateral negotiation with deception.

    The agent combines a strong negotiation strategy (aspiration concession,
    opponent modeling) with a deception layer that strategically chooses bids
    to confuse the opponent's model of our preferences.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Configurable parameters
    # ──────────────────────────────────────────────────────────────────────

    # Aspiration exponent: >1 = boulware (holds out), <1 = conceder
    # 3.5 is moderately boulware — tough for most of the negotiation,
    # then concedes in the last ~25% of rounds
    ASPIRATION_EXPONENT: float = 3.5

    # What fraction of the time we use concealing (deceptive) bids vs
    # honest (utility-maximizing) bids. Higher = more deceptive but riskier.
    # 0.4 means 40% of our offers are chosen to confuse the opponent's model.
    CONCEAL_RATIO: float = 0.4

    # Opponent model learning rate — how fast we update issue weights
    # based on new offers from the opponent. Higher = more reactive.
    OPPONENT_LEARNING_RATE: float = 0.1

    # Number of recent opponent offers for behavior trend estimation
    OPPONENT_WINDOW_SIZE: int = 5

    # Weight of opponent trend in threshold adjustment
    OPPONENT_TREND_WEIGHT: float = 0.12

    def on_preferences_changed(self, changes):
        """
        Called when preferences are set. This is our initialization point.

        We build:
            1. A sorted list of all rational outcomes (utility > reservation value)
            2. An initial opponent model (starts as uniform, updated via frequency analysis)
            3. Data structures for tracking opponent behavior
        """
        if self.ufun is None:
            return

        # ── Build sorted list of rational outcomes ──
        # Enumerate all outcomes and keep only those with utility above reservation.
        # Sort by OUR utility descending — our best outcomes come first.
        ufun_outcome = [
            (float(self.ufun(outcome)), outcome)
            for outcome in self.nmi.outcome_space.enumerate_or_sample()
            if float(self.ufun(outcome)) > float(self.ufun.reserved_value)
        ]
        ufun_outcome.sort(reverse=True)

        # Store outcomes and their utilities separately for fast access
        self._rational_outcomes = tuple(item[1] for item in ufun_outcome)
        self._rational_utilities = tuple(item[0] for item in ufun_outcome)

        # Pre-compute utility bounds for threshold calculations
        if self._rational_utilities:
            self._max_utility = self._rational_utilities[0]
            self._min_utility = float(self.ufun.reserved_value)
        else:
            self._max_utility = 0.0
            self._min_utility = 0.0

        # ── Initialize opponent preference model ──
        # We use a frequency-based approach: track how often each value appears
        # in the opponent's offers for each issue, and infer their preferences.
        # Start with uniform weights — we know nothing yet.
        n_issues = len(self.nmi.outcome_space.issues)
        self._opponent_issue_weights = [1.0 / n_issues] * n_issues
        self._opponent_value_counts: dict[int, dict] = {}
        for i, issue in enumerate(self.nmi.outcome_space.issues):
            self._opponent_value_counts[i] = defaultdict(int)

        # Build the initial opponent ufun as a LinearMultiFun (updatable)
        self._build_opponent_ufun()

        # ── Opponent behavior tracking ──
        # Track (step, our_utility_of_their_offer) to detect concession trends
        self._opponent_offer_history: list[tuple[int, float]] = []

        # ── Deception: pre-compute issue rankings for concealing bids ──
        # We identify which issues matter MOST to us so we can craft bids
        # that mislead the opponent about our true issue weights.
        self._compute_issue_importance()

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """
        Main entry point — called each turn to produce an offer or accept.

        Flow:
            1. If no offer yet (we're first), make our opening bid
            2. Update opponent model with their latest offer
            3. Check if their offer is acceptable
            4. If not, generate a counter-offer (sometimes concealing, sometimes honest)
        """
        offer = state.current_offer

        if self.ufun is None or not self._rational_outcomes:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # First move — no offer from opponent yet
        if offer is None:
            return SAOResponse(
                ResponseType.REJECT_OFFER, self._generate_bid(state)
            )

        # Update opponent model with their new offer
        self._update_opponent_model(state)

        # Check acceptance
        if self._should_accept(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # Generate counter-offer
        return SAOResponse(
            ResponseType.REJECT_OFFER, self._generate_bid(state)
        )

    # ──────────────────────────────────────────────────────────────────────
    # Acceptance strategy
    # ──────────────────────────────────────────────────────────────────────

    def _should_accept(self, state: SAOState) -> bool:
        """
        Decide whether to accept the opponent's current offer.

        Strategy:
            1. Never accept below reservation value
            2. Compute aspiration-based threshold (decays over time)
            3. Adjust threshold based on opponent's concession trend
            4. Accept if offer utility >= adjusted threshold

        The aspiration curve uses the formula:
            threshold = min_util + (max_util - min_util) * (1 - t^(1/e))
        where t = relative_time and e = aspiration exponent.

        Args:
            state: Current negotiation state with the opponent's offer.

        Returns:
            True if we should accept the offer.
        """
        offer = state.current_offer
        if offer is None:
            return False

        offer_utility = float(self.ufun(offer))

        # Never accept below reservation value
        if offer_utility <= float(self.ufun.reserved_value):
            return False

        # Calculate aspiration-based threshold
        threshold = self._calc_threshold(state)

        # Adjust using opponent trend — if they're conceding fast, wait for
        # better; if they're stuck, be more flexible
        trend = self._estimate_opponent_trend()
        adjustment = trend * self.OPPONENT_TREND_WEIGHT * (self._max_utility - self._min_utility)
        threshold = max(self._min_utility, min(self._max_utility, threshold + adjustment))

        return offer_utility >= threshold

    def _calc_threshold(self, state: SAOState) -> float:
        """
        Calculate the aspiration-based acceptance threshold.

        Uses exponential decay: starts near max_utility, decays to min_utility
        as we approach the deadline.

        At t=0: threshold ≈ max_utility (very demanding)
        At t=1: threshold = min_utility (accept anything rational)

        Args:
            state: Current negotiation state.

        Returns:
            Utility threshold value.
        """
        t = state.relative_time

        # Aspiration level: 1 at start, 0 at deadline
        if t <= 0:
            level = 1.0
        elif t >= 1.0:
            level = 0.0
        else:
            level = 1.0 - math.pow(t, 1.0 / self.ASPIRATION_EXPONENT)

        # Scale to utility range
        return self._min_utility + level * (self._max_utility - self._min_utility)

    # ──────────────────────────────────────────────────────────────────────
    # Bidding strategy (with concealing/deception)
    # ──────────────────────────────────────────────────────────────────────

    def _generate_bid(self, state: SAOState) -> Outcome | None:
        """
        Generate a bid — either an honest aspiration-based bid or a concealing bid.

        The key innovation for ANL 2026: we sometimes choose bids that are
        GOOD FOR US but MISLEADING about our preferences. Specifically, a
        concealing bid is one that has similar utility to what we'd normally
        offer, but achieves that utility through DIFFERENT issues than our
        most important ones. This makes it harder for the opponent to learn
        our true issue weights.

        Args:
            state: Current negotiation state.

        Returns:
            An outcome to offer.
        """
        if not self._rational_outcomes:
            return None

        # Calculate our target utility level (aspiration-based)
        target = self._calc_threshold(state)

        # Find candidate outcomes near our target level
        # We want outcomes with utility in [target - margin, target + margin]
        margin = 0.05 * (self._max_utility - self._min_utility)
        candidates = [
            (i, outcome)
            for i, (outcome, util) in enumerate(
                zip(self._rational_outcomes, self._rational_utilities)
            )
            if target - margin <= util <= target + margin
        ]

        # If no candidates in the narrow band, widen the search
        if not candidates:
            # Find the closest outcomes to our target
            candidates = [
                (i, outcome)
                for i, (outcome, util) in enumerate(
                    zip(self._rational_outcomes, self._rational_utilities)
                )
                if util >= target - 2 * margin
            ]

        # If still empty, just use the best available outcomes
        if not candidates:
            n = min(10, len(self._rational_outcomes))
            candidates = [(i, self._rational_outcomes[i]) for i in range(n)]

        # Decide: concealing bid or honest bid?
        # Early in negotiation, we conceal more (opponent is learning our weights).
        # Late in negotiation, we bid honestly (need to close the deal).
        conceal_prob = self.CONCEAL_RATIO * (1.0 - state.relative_time)

        if random.random() < conceal_prob and len(candidates) > 1:
            # CONCEALING BID: among candidates with similar utility, pick the one
            # that least reveals our true issue preferences.
            return self._pick_concealing_bid(candidates)
        else:
            # HONEST BID: pick the candidate with the highest utility for us
            candidates.sort(key=lambda x: self._rational_utilities[x[0]], reverse=True)
            return candidates[0][1]

    def _pick_concealing_bid(
        self, candidates: list[tuple[int, Outcome]]
    ) -> Outcome:
        """
        From a set of candidates with similar utility, pick the one that best
        misleads the opponent about our true preferences.

        The idea: if our most important issue is issue #0, we prefer to offer
        bids where the value on issue #0 is NOT our favorite — this makes the
        opponent think issue #0 is less important to us than it really is.

        We score each candidate by how "misleading" it is — how much it
        deviates from what the opponent would expect if they knew our true
        preferences — and pick the most misleading one.

        Args:
            candidates: List of (index, outcome) tuples with similar utility.

        Returns:
            The most concealing outcome from the candidates.
        """
        if not hasattr(self, '_issue_importance_rank'):
            return candidates[0][1]

        best_score = -1.0
        best_outcome = candidates[0][1]

        for _, outcome in candidates:
            # Score = how much this bid "lies" about our preferences.
            # For each issue, check if the value offered is one of our top choices.
            # If we're offering a non-favorite value on an important issue,
            # that's maximally concealing.
            conceal_score = 0.0
            for issue_idx, importance in enumerate(self._issue_importance_rank):
                if issue_idx < len(outcome):
                    # Higher importance issues contribute more to concealment
                    # when we offer non-top values on them
                    value = outcome[issue_idx]
                    # Check if this is our best value for this issue
                    is_best = (value == self._best_values.get(issue_idx))
                    if not is_best:
                        conceal_score += importance
            if conceal_score > best_score:
                best_score = conceal_score
                best_outcome = outcome

        return best_outcome

    def _compute_issue_importance(self):
        """
        Pre-compute which issues matter most to us and our best value per issue.

        We estimate issue importance by checking the utility range of each issue
        (how much our utility changes when we vary values on that issue).
        """
        if self.ufun is None:
            return

        issues = self.nmi.outcome_space.issues
        n_issues = len(issues)

        # For each issue, estimate its importance by measuring the utility
        # spread when we vary just that issue's values
        importance = []
        self._best_values = {}

        for i, issue in enumerate(issues):
            # Get all values for this issue
            values = list(issue.all)
            if not values:
                importance.append(0.0)
                continue

            # Build test outcomes varying only this issue, with other issues
            # set to the first value in their range (rough approximation)
            base_outcome = list(self._rational_outcomes[0]) if self._rational_outcomes else [v for iss in issues for v in [list(iss.all)[0]]]

            best_val = values[0]
            best_util = -float('inf')
            utils = []

            for val in values:
                test = list(base_outcome)
                test[i] = val
                u = float(self.ufun(tuple(test)))
                utils.append(u)
                if u > best_util:
                    best_util = u
                    best_val = val

            self._best_values[i] = best_val

            # Importance = range of utility when varying this issue
            if utils:
                importance.append(max(utils) - min(utils))
            else:
                importance.append(0.0)

        # Normalize to [0, 1]
        total = sum(importance) if importance else 1.0
        if total > 0:
            self._issue_importance_rank = [imp / total for imp in importance]
        else:
            self._issue_importance_rank = [1.0 / n_issues] * n_issues

    # ──────────────────────────────────────────────────────────────────────
    # Opponent preference modeling (frequency-based)
    # ──────────────────────────────────────────────────────────────────────

    def _update_opponent_model(self, state: SAOState) -> None:
        """
        Update our model of the opponent's preferences based on their latest offer.

        We use a frequency-based approach:
            1. Count how often each value appears for each issue in opponent offers
            2. Values that appear more often are likely more preferred by the opponent
            3. Issues where the opponent varies values less are likely more important

        We also track the opponent's offers (from our perspective) for trend analysis.

        Args:
            state: Current negotiation state with the opponent's offer.
        """
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return

        # Record this offer's utility (from our perspective) for trend analysis
        self._opponent_offer_history.append(
            (state.step, float(self.ufun(offer)))
        )

        # Update value frequency counts for each issue
        for i in range(len(offer)):
            self._opponent_value_counts[i][offer[i]] += 1

        # Estimate issue weights from value consistency:
        # If the opponent always picks the same value for an issue, that issue
        # is likely very important to them. If they vary a lot, less important.
        total_offers = state.step + 1
        issue_consistencies = []

        for i in range(len(self.nmi.outcome_space.issues)):
            counts = self._opponent_value_counts[i]
            if not counts:
                issue_consistencies.append(0.0)
                continue
            # Consistency = how concentrated the distribution is (max_count / total)
            max_count = max(counts.values())
            consistency = max_count / total_offers
            issue_consistencies.append(consistency)

        # Normalize to get issue weights
        total_consistency = sum(issue_consistencies)
        if total_consistency > 0:
            new_weights = [c / total_consistency for c in issue_consistencies]
        else:
            n = len(self.nmi.outcome_space.issues)
            new_weights = [1.0 / n] * n

        # Smooth update: blend old weights with new (prevents wild swings)
        alpha = self.OPPONENT_LEARNING_RATE
        self._opponent_issue_weights = [
            alpha * new_w + (1 - alpha) * old_w
            for new_w, old_w in zip(new_weights, self._opponent_issue_weights)
        ]

        # Rebuild the opponent ufun with updated weights
        self._build_opponent_ufun()

    def _build_opponent_ufun(self):
        """
        Build/rebuild the opponent's estimated utility function.

        We construct a LinearMultiFun using our estimated issue weights and
        value utilities derived from frequency analysis. The opponent can
        read this via self.opponent_ufun, which is what gets scored for
        deception (Kendall correlation with actual opponent ufun).

        The deception game: we want this model to be INACCURATE (low Kendall
        correlation) so our concealing score is high. But we also want it to
        be useful enough internally for generating good bids.
        """
        issues = self.nmi.outcome_space.issues
        n_issues = len(issues)

        # Build value utility maps for each issue.
        # More frequently offered values by the opponent -> higher estimated utility.
        # But we also DISTORT this slightly to boost our concealing score.
        value_utils = []
        for i, issue in enumerate(issues):
            values = list(issue.all)
            counts = self._opponent_value_counts[i]
            total = sum(counts.values()) if counts else 1

            if total > 0 and counts:
                # Normalize counts to [0, 1] range as utility estimates
                max_count = max(counts.values()) if counts else 1
                utils = {}
                for val in values:
                    count = counts.get(val, 0)
                    utils[val] = count / max_count if max_count > 0 else 0.5
            else:
                # No data yet — assume uniform
                utils = {val: 0.5 for val in values}

            value_utils.append(utils)

        # Create a LinearMultiFun with our estimated weights and value utilities.
        # LinearMultiFun expects: weights (per issue) and value functions (per issue).
        # We construct it as a callable that computes: sum(weight_i * value_util_i(offer_i))
        weights = list(self._opponent_issue_weights)
        vutils = list(value_utils)

        def opponent_eval(outcome):
            if outcome is None:
                return 0.0
            total = 0.0
            for i in range(min(len(outcome), n_issues)):
                w = weights[i] if i < len(weights) else 0.0
                vu = vutils[i] if i < len(vutils) else {}
                total += w * vu.get(outcome[i], 0.5)
            return total

        from negmas.preferences import LambdaMultiFun
        self.private_info["opponent_ufun"] = LambdaMultiFun(f=opponent_eval)

    # ──────────────────────────────────────────────────────────────────────
    # Opponent behavior modeling (trend estimation)
    # ──────────────────────────────────────────────────────────────────────

    def _estimate_opponent_trend(self) -> float:
        """
        Estimate how fast the opponent is conceding by looking at recent offers.

        Fits a simple linear trend to the last N offers (measured in OUR utility).
        Positive = opponent is conceding (better offers for us).
        Negative = opponent is hardening (worse offers for us).

        Returns:
            Normalized trend in roughly [-1, 1].
        """
        history = self._opponent_offer_history

        if len(history) < 2:
            return 0.0

        # Take the last N offers
        recent = history[-self.OPPONENT_WINDOW_SIZE:]
        n = len(recent)

        steps = [h[0] for h in recent]
        utils = [h[1] for h in recent]

        # Simple linear regression: slope of utility vs step
        mean_step = sum(steps) / n
        mean_util = sum(utils) / n

        numerator = sum(
            (s - mean_step) * (u - mean_util)
            for s, u in zip(steps, utils)
        )
        denominator = sum((s - mean_step) ** 2 for s in steps)

        if abs(denominator) < 1e-10:
            return 0.0

        slope = numerator / denominator

        # Normalize to roughly [-1, 1]
        return max(-1.0, min(1.0, slope * 50.0))
