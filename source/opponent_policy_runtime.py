from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


def _resource_path(name: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root / name


def _stable_uniform(*parts: Any) -> float:
    payload = repr(parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (value + 0.5) / (2**64)


def _logit(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, float(p)))
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class OpponentAction:
    action: str
    index: int
    card_id: int
    score: float
    immediate_win: bool


class OpponentPolicyRuntime:
    """Population opponent policy used by the online counterfactual search.

    The model is the already-trained pairwise conditional-logit ranker from
    models/opponent_policy.json.  A single deterministic common-random-number
    draw is used per hidden-hand particle, so alternative root actions see the
    same opponent-policy randomness.  Exact opponent lethal is never sampled
    away: if the hidden hand contains a legal immediate win, the opponent is
    treated as taking a lethal action.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.weights = [float(value) for value in payload["weights"]]
        validation = payload.get("validation") or {}
        actual = float(validation.get("actual_discard_rate") or 0.0)
        predicted = float(validation.get("predicted_discard_rate_particle") or 0.0)
        self.discard_logit_offset = (
            _logit(actual) - _logit(predicted)
            if 0.0 < actual < 1.0 and 0.0 < predicted < 1.0
            else 0.0
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "OpponentPolicyRuntime | None":
        try:
            payload = json.loads((path or _resource_path("opponent_policy.json")).read_text(encoding="utf-8-sig"))
            if len(payload.get("weights") or []) != 31:
                return None
            return cls(payload)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def score(self, features: list[float], *, action: str) -> float:
        score = sum(left * right for left, right in zip(features, self.weights))
        if action == "drop":
            score += self.discard_logit_offset
        return score

    @staticmethod
    def choose(actions: list[OpponentAction], uniform: float) -> OpponentAction | None:
        if not actions:
            return None
        lethal = [item for item in actions if item.immediate_win and item.action == "turn"]
        if lethal:
            return max(lethal, key=lambda item: item.score)
        top = max(item.score for item in actions)
        weights = [math.exp(max(-30.0, min(30.0, item.score - top))) for item in actions]
        total = sum(weights)
        if total <= 0.0:
            return max(actions, key=lambda item: item.score)
        needle = min(1.0 - 1e-15, max(0.0, uniform)) * total
        for item, weight in zip(actions, weights):
            needle -= weight
            if needle <= 0.0:
                return item
        return actions[-1]


def _choice_key(choice: Any) -> tuple[str, int]:
    return str(choice.action), int(choice.slot)


def _bounded_probability(value: Any, default: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(default if value is None else value)))
    except (TypeError, ValueError):
        return default


def _selector_risk_weight(*, tower: int, top_pwin: float, uncertain: bool) -> float:
    """Extra survival penalty applied only by the final 3.9.3 selector.

    The engine's P(win) already contains CVaR and a small survival term.  Live
    telemetry showed that term is too weak when P(win) is saturated near zero
    or our tower is critical, so this is intentionally state-dependent rather
    than a global replacement for the learned value model.
    """

    if tower <= 10:
        weight = 0.75
    elif top_pwin <= 0.10:
        weight = 0.60
    elif top_pwin <= 0.25:
        weight = 0.40
    else:
        weight = 0.20
    if uncertain:
        weight += 0.15
    return min(0.90, weight)


def _select_survival_choice(
    choices: list[Any],
    *,
    tower: int,
    unsafe_keys: set[tuple[str, int]] | None = None,
    policy_available: bool = False,
    uncertain: bool = False,
) -> tuple[Any | None, str, float, float]:
    """Select the production action after Monte Carlo has finished.

    Safety invariants:
    * an exact immediate win always wins;
    * a move that immediately loses the game is never selected while any
      non-losing legal action exists;
    * a move is removed when another move cuts next-turn loss risk by >=15pp
      while giving up <=10pp P(win) (clear survival dominance);
    * remaining choices use P(win) minus an explicit state-dependent loss-risk
      penalty; learned policy only breaks near-ties in saturated/uncertain
      regions and can never override a material risk gap.
    """

    if not choices:
        return None, "no_choices", 0.0, 0.0

    unsafe = unsafe_keys or set()
    exact_wins = [
        item for item in choices
        if bool(getattr(item, "immediate_terminal_win", False)) and _choice_key(item) not in unsafe
    ]
    if exact_wins:
        selected = max(
            exact_wins,
            key=lambda item: (
                _bounded_probability(getattr(item, "p_win", None)),
                float(getattr(item, "policy_score", 0.0) or 0.0),
            ),
        )
        return selected, "exact_terminal_win", 1.0, 0.0

    safe = [item for item in choices if _choice_key(item) not in unsafe]
    if not safe:
        safe = list(choices)

    def pwin(item: Any) -> float:
        return _bounded_probability(getattr(item, "p_win", None))

    def risk(item: Any) -> float:
        return _bounded_probability(getattr(item, "p_lose_next_turn", None))

    top_pwin = max(pwin(item) for item in safe)

    # Remove actions that are plainly dominated on survival.  This is the
    # pattern found in the 22-34 live series: e.g. 5.17% P(win)/36.8% loss
    # versus 0.03% P(win)/0.5% loss, and 3.76%/58.1% versus 0.12%/0%.
    nondominated: list[Any] = []
    for candidate in safe:
        candidate_pwin = pwin(candidate)
        candidate_risk = risk(candidate)
        dominated = any(
            candidate_risk - risk(other) >= 0.15
            and candidate_pwin - pwin(other) <= 0.10
            for other in safe
            if other is not candidate
        )
        if not dominated:
            nondominated.append(candidate)
    pool = nondominated or safe

    risk_weight = _selector_risk_weight(tower=tower, top_pwin=top_pwin, uncertain=uncertain)

    def utility(item: Any) -> float:
        return pwin(item) - risk_weight * risk(item)

    best_utility = max(utility(item) for item in pool)
    utility_best = [item for item in pool if best_utility - utility(item) <= 1e-12]
    selected = max(
        utility_best,
        key=lambda item: (pwin(item), -risk(item), float(getattr(item, "policy_score", 0.0) or 0.0)),
    )
    reason = "survival_adjusted_pwin"

    # Learned policy becomes an actual tie-break only where the value surface
    # is strongly saturated or the Monte Carlo decision is explicitly
    # uncertain.  It may move at most 0.5pp in P(win)/selector utility and 2pp
    # in loss risk, so policy cannot re-introduce a materially dangerous move.
    saturated = top_pwin <= 0.02 or top_pwin >= 0.98
    if policy_available and (saturated or uncertain):
        selected_risk = risk(selected)
        selected_pwin = pwin(selected)
        near = [
            item for item in pool
            if best_utility - utility(item) <= 0.005
            and abs(pwin(item) - selected_pwin) <= 0.005
            and risk(item) <= selected_risk + 0.02
        ]
        policy_near = [
            item for item in near
            if getattr(item, "policy_score", None) is not None
            and float(getattr(item, "policy_score")) > -999_999.0
        ]
        if policy_near:
            policy_selected = max(
                policy_near,
                key=lambda item: (
                    float(getattr(item, "policy_score")),
                    utility(item),
                    pwin(item),
                    -risk(item),
                ),
            )
            if policy_selected is not selected:
                selected = policy_selected
                reason = "policy_tiebreak_in_saturated_or_uncertain_zone"

    return selected, reason, utility(selected), risk_weight


_PATCHED = False
_ORIGINAL_OPPONENT_REPLY = None


def apply_runtime_patch() -> bool:
    """Install the 3.9.3 opponent-policy and survival-selector runtime.

    Returns True only when the learned opponent model is available and all
    runtime overrides are installed.  The application entry point fails closed
    when this returns False, so an older selector cannot launch silently.
    """

    global _PATCHED, _ORIGINAL_OPPONENT_REPLY
    if _PATCHED:
        return True

    from card_game import ProbabilisticCardStrategy

    model = OpponentPolicyRuntime.load()
    if model is None:
        return False

    original = ProbabilisticCardStrategy._opponent_reply
    _ORIGINAL_OPPONENT_REPLY = original

    def _policy_actions(
        self: Any,
        me: Any,
        enemy: Any,
        opponent_hand: tuple[int, ...],
        state: Any,
        *,
        label: Any,
    ) -> tuple[list[OpponentAction], Any]:
        mirrored = self._state_with(state, me, enemy, opponent_actor=True)
        actions: list[OpponentAction] = []
        if self.policy_runtime is None:
            return actions, mirrored
        for index, card_id in enumerate(opponent_hand):
            card = self.catalog.cards.get(card_id)
            if card is None:
                continue
            for action in ("drop", "turn"):
                if action == "turn" and not self._affordable(card, enemy):
                    continue
                features = self.policy_runtime._action_base(self, mirrored, action, card)
                score = model.score(features, action=action)
                immediate_win = bool(action == "turn" and len(features) > 2 and features[2] >= 0.5)
                actions.append(OpponentAction(action, index, card_id, score, immediate_win))
        return actions, mirrored

    def _sample_action(
        self: Any,
        me: Any,
        enemy: Any,
        opponent_hand: tuple[int, ...],
        state: Any,
        *,
        label: Any,
    ) -> tuple[OpponentAction | None, Any]:
        actions, mirrored = _policy_actions(self, me, enemy, opponent_hand, state, label=label)
        if not actions:
            return None, mirrored
        uniform = _stable_uniform(
            "opponent-policy",
            state.game_id,
            state.turn,
            tuple(opponent_hand),
            label,
        )
        return model.choose(actions, uniform), mirrored

    def _opponent_reply_empirical(
        self: Any,
        me: Any,
        enemy: Any,
        opponent_hand: tuple[int, ...],
        our_remaining_hand: tuple[int, ...],
        state: Any,
        unseen_pool: list[int],
        reply_cache: dict[tuple[Any, ...], tuple[float, bool, Any, Any, float]],
        extra_reply_cache: dict[tuple[Any, ...], dict[int, tuple[float, bool, Any, Any, float]]],
        next_win_cache: dict[tuple[Any, ...], bool],
        depth: int = 2,
        deadline: float | None = None,
    ) -> tuple[float, bool, bool, float, int | None]:
        # If the learned state model is unavailable, use the original method
        # with the untouched pre-income state.  The original method applies
        # opponent income itself; passing an already-incremented state here
        # would double-count income.
        if self.policy_runtime is None:
            return original(
                self,
                me,
                enemy,
                opponent_hand,
                our_remaining_hand,
                state,
                unseen_pool,
                reply_cache,
                extra_reply_cache,
                next_win_cache,
                depth,
                deadline,
            )

        # Keep the proven terminal guard from the original search.
        enemy = self._income(enemy)
        mirrored = self._state_with(state, me, enemy, opponent_actor=True)
        if self._won(enemy, me, mirrored):
            return 0.0, True, True, 0.0, None

        selected, mirrored = _sample_action(
            self,
            me,
            enemy,
            opponent_hand,
            state,
            label=("root", depth),
        )
        if selected is None:
            probability = self._state_probability(self._income(me), enemy, state, our_remaining_hand)
            return (
                probability,
                False,
                False,
                self._win_on_next_action_probability(me, enemy, our_remaining_hand, state, unseen_pool),
                None,
            )

        # A discard passes play back to us.  Its learned probability comes from
        # the population model, including the aggregate discard-rate offset.
        if selected.action == "drop":
            probability = self._state_probability(self._income(me), enemy, state, our_remaining_hand)
            our_win_next = self._win_on_next_action_probability(me, enemy, our_remaining_hand, state, unseen_pool)
            return probability, False, False, our_win_next, selected.card_id

        card = self.catalog[selected.card_id]
        reply_enemy, reply_me = self.simulate(card, mirrored)
        immediate_loss = self._won(reply_enemy, reply_me, mirrored)
        remaining_enemy = (
            opponent_hand[: selected.index] + opponent_hand[selected.index + 1 :]
        )
        if immediate_loss:
            return 0.0, True, True, 0.0, selected.card_id

        # Extra-turn cards get one additional population-policy decision.  This
        # removes the old minimax second-card choice while retaining the exact
        # lethal override and the existing bounded search depth.
        if selected.card_id in self.EXTRA_TURN_CARDS and depth > 1 and remaining_enemy:
            second, second_state = _sample_action(
                self,
                reply_me,
                reply_enemy,
                remaining_enemy,
                state,
                label=("extra", depth, selected.card_id),
            )
            if second is not None and second.action == "turn":
                second_card = self.catalog[second.card_id]
                second_enemy, second_me = self.simulate(second_card, second_state)
                second_loss = self._won(second_enemy, second_me, second_state)
                if second_loss:
                    return 0.0, True, True, 0.0, selected.card_id
                reply_enemy, reply_me = second_enemy, second_me
                remaining_enemy = remaining_enemy[: second.index] + remaining_enemy[second.index + 1 :]

        # Important 3.9.2 change: evaluate the actual post-reply state with the
        # calibrated learned state-value model.  The old code used the manual
        # board heuristic here and added a pre-reply hand-value correction.
        probability = self._state_probability(
            self._income(reply_me),
            reply_enemy,
            state,
            our_remaining_hand,
        )
        our_win_next = self._win_on_next_action_probability(
            reply_me,
            reply_enemy,
            our_remaining_hand,
            state,
            unseen_pool,
        )

        # Preserve the conservative two-opponent-action diagnostic.  It does
        # not choose the current reply and therefore no longer injects minimax
        # bias into root P(win).
        opponent_win_later = False
        for remaining_id in remaining_enemy:
            cache_key = (selected.card_id, remaining_id, reply_me, reply_enemy)
            wins = next_win_cache.get(cache_key)
            if wins is None:
                attacker = self._income(reply_enemy)
                remaining_card = self.catalog.cards.get(remaining_id)
                if remaining_card and self._affordable(remaining_card, attacker):
                    next_state = self._state_with(state, reply_me, attacker, opponent_actor=True)
                    projected_attacker, projected_defender = self.simulate(remaining_card, next_state)
                    wins = self._won(projected_attacker, projected_defender, next_state)
                else:
                    wins = False
                next_win_cache[cache_key] = wins
            if wins:
                opponent_win_later = True
                break

        return probability, False, opponent_win_later, our_win_next, selected.card_id

    original_aggregate_samples = ProbabilisticCardStrategy._aggregate_samples
    original_rank_choices = ProbabilisticCardStrategy.rank_choices
    original_metadata = ProbabilisticCardStrategy.metadata

    def _aggregate_samples_393(
        self: Any,
        base: Any,
        samples: dict[str, Any],
        state: Any,
        top_threats: tuple[str, ...],
    ) -> Any:
        choice = original_aggregate_samples(self, base, samples, state, top_threats)
        if base.action != "turn":
            return choice
        projected_me, projected_enemy = self.simulate(base.card, state)
        if not self._lost(projected_me, projected_enemy, state):
            return choice
        # Clamp before adaptive ranking/stopping.  The stock aggregator applies
        # unlock/identity/symmetric bonuses even to terminal_value=0, which can
        # resurrect an exact self-loss as a small positive P(win).
        return replace(
            choice,
            score=0.0,
            p_win=0.0,
            p_win_next_action=0.0,
            p_lose_next_turn=1.0,
            p_win_within_2_own_actions=0.0,
            p_opponent_win_within_2_actions=1.0,
            expected_reply_value=0.0,
            policy_score=-1_000_000.0,
            reasons=("3.9.3 safety clamp: ход немедленно проигрывает партию",),
            final_rank_reason="запрещён: ход немедленно проигрывает партию",
        )

    def _rank_choices_survival(self: Any, state: Any) -> list[Any]:
        choices = original_rank_choices(self, state)
        if not choices:
            return choices

        unsafe_keys: set[tuple[str, int]] = set()
        repaired: list[Any] = []
        for choice in choices:
            self_terminal_loss = False
            if choice.action == "turn":
                projected_me, projected_enemy = self.simulate(choice.card, state)
                self_terminal_loss = self._lost(projected_me, projected_enemy, state)
            if self_terminal_loss:
                unsafe_keys.add(_choice_key(choice))
                # 3.9.1/3.9.2 could resurrect a terminal-zero P(win) with a
                # later unlock/identity bonus.  Clamp exact self-loss after all
                # engine adjustments so it can never appear as 0.84% again.
                choice = replace(
                    choice,
                    score=0.0,
                    p_win=0.0,
                    p_lose_next_turn=1.0,
                    expected_reply_value=0.0,
                    policy_score=-1_000_000.0,
                    final_rank_reason="запрещён: ход немедленно проигрывает партию",
                )
            repaired.append(choice)

        uncertain = bool(getattr(repaired[0], "decision_uncertain", False))
        selected, selector_reason, selector_utility, risk_weight = _select_survival_choice(
            repaired,
            tower=int(state.me.tower),
            unsafe_keys=unsafe_keys,
            policy_available=self.policy_runtime is not None,
            uncertain=uncertain,
        )
        if selected is None:
            return repaired

        selected_key = _choice_key(selected)
        selected_pwin = _bounded_probability(getattr(selected, "p_win", None))
        selected_risk = _bounded_probability(getattr(selected, "p_lose_next_turn", None))
        reason_text = {
            "exact_terminal_win": "3.9.3 selector: точная немедленная победа",
            "survival_adjusted_pwin": (
                f"3.9.3 selector: survival-adjusted P(win), "
                f"risk weight {risk_weight:.2f}"
            ),
            "policy_tiebreak_in_saturated_or_uncertain_zone": (
                "3.9.3 selector: learned policy tie-break внутри безопасной "
                "насыщенной/неопределённой зоны"
            ),
        }.get(selector_reason, f"3.9.3 selector: {selector_reason}")

        result: list[Any] = []
        for choice in repaired:
            if _choice_key(choice) == selected_key:
                result.append(
                    replace(
                        choice,
                        reasons=(reason_text,) + tuple(choice.reasons),
                        final_rank_reason=(
                            f"{reason_text}; utility={selector_utility:.4f}; "
                            f"P(win)={selected_pwin:.2%}; P(lose next)={selected_risk:.2%}"
                        ),
                    )
                )
            else:
                result.append(choice)
        result.sort(key=lambda item: _choice_key(item) == selected_key, reverse=True)

        if isinstance(getattr(self, "last_sampling", None), dict):
            self.last_sampling.update(
                {
                    "selected_key": selected_key,
                    "selector_version": "3.9.3-survival-guard",
                    "selector_reason": selector_reason,
                    "selector_utility": selector_utility,
                    "selector_risk_weight": risk_weight,
                    "self_terminal_vetoes": sorted(unsafe_keys),
                    "learned_policy_role": (
                        "safe tie-break in saturated/uncertain zone; cannot override material risk gap"
                    ),
                }
            )
        return result

    def _metadata_393(self: Any) -> dict[str, Any]:
        metadata = dict(original_metadata(self))
        metadata.update(
            {
                "version": "3.9.3-survival-selector",
                "oos_series_id": "3.9.3-clean-oos-2026-08-21",
                "runtime_patch_active": True,
                "selector_version": "3.9.3-survival-guard",
                "selection_objective": [
                    "exact_immediate_terminal_win",
                    "hard_veto_immediate_self_loss",
                    "survival_dominance_filter",
                    "state_dependent_Pwin_minus_loss_risk",
                    "safe_policy_tiebreak_when_saturated_or_uncertain",
                    "deterministic_stable_order",
                ],
                "learned_policy_role": (
                    "safe tie-break in saturated/uncertain zone; never overrides material survival gap"
                ),
            }
        )
        return metadata

    ProbabilisticCardStrategy._opponent_reply = _opponent_reply_empirical
    ProbabilisticCardStrategy._aggregate_samples = _aggregate_samples_393
    ProbabilisticCardStrategy.rank_choices = _rank_choices_survival
    ProbabilisticCardStrategy.metadata = _metadata_393
    ProbabilisticCardStrategy.STRATEGY_VERSION = "3.9.3-survival-selector"
    ProbabilisticCardStrategy.OOS_SERIES_ID = "3.9.3-clean-oos-2026-08-21"
    ProbabilisticCardStrategy.RUNTIME_PATCH_ACTIVE = True
    _PATCHED = True
    return True
