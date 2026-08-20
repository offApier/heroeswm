from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
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


_PATCHED = False
_ORIGINAL_OPPONENT_REPLY = None


def apply_runtime_patch() -> bool:
    """Patch 3.9.1 online reply selection without touching the large engine.

    Returns True when the learned opponent policy is available and installed.
    If the model file is unavailable the original 3.9.1 implementation remains
    untouched, which keeps source and frozen builds fail-safe.
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
        # Keep the proven terminal guard from the original search.
        enemy = self._income(enemy)
        mirrored = self._state_with(state, me, enemy, opponent_actor=True)
        if self._won(enemy, me, mirrored):
            return 0.0, True, True, 0.0, None
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

    ProbabilisticCardStrategy._opponent_reply = _opponent_reply_empirical
    ProbabilisticCardStrategy.STRATEGY_VERSION = "3.9.2-opponent-policy-runtime"
    ProbabilisticCardStrategy.OOS_SERIES_ID = "3.9.2-clean-oos-2026-08-20"
    _PATCHED = True
    return True
