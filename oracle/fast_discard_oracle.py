from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BeliefParticle:
    opponent_hand: tuple[int, ...]
    latent_last_seen: tuple[tuple[int, int], ...] = tuple()


@dataclass(frozen=True)
class SearchAction:
    action: str
    slot: int
    card_id: int


@dataclass
class SearchDecision:
    action: str
    slot: int
    card_id: int
    p_win: float
    outcomes: list[float]
    replacements: list[int]
    replacement_distribution: dict[str, float]
    tail_survival: float
    paired_se: float | None = None
    ci_diff: tuple[float, float] | None = None


def _stable_uniform(seed: str, *parts: Any) -> float:
    payload = repr((seed, *parts)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (value + 0.5) / (2**64)


class PersistentParticleFilter:
    """Sequential hidden-hand filter used only by the offline audit.

    Compatible particles retain the five unplayed opponent cards.  Particles
    incompatible with a revealed opponent action receive zero weight and are
    rejuvenated from the conditional distribution containing that card.  A
    single replacement is then drawn from that particle's cyclic deck state.
    """

    def __init__(self, strategy: Any, state: Any, count: int, *, seed: str) -> None:
        self.strategy = strategy
        self.catalog = strategy.catalog
        self.seed = seed
        self.action_index = int(state.turn)
        self.public_last_seen: dict[int, int] = {}
        initial = strategy.belief.particles(state, max(1, count))
        self.particles = [BeliefParticle(tuple(hand)) for hand in initial]
        self.conditioned_actions = 0
        self.rejuvenated_particles = 0
        self.unknown_transitions = 0
        self._reconcile_with_our_hand(tuple(state.hand), self.action_index, "initial")

    @staticmethod
    def _last_seen(particle: BeliefParticle, public: dict[int, int]) -> dict[int, int]:
        merged = dict(public)
        merged.update(dict(particle.latent_last_seen))
        return merged

    def _weighted_pool(
        self,
        particle: BeliefParticle,
        action_index: int,
        excluded: set[int],
        *,
        last_seen_override: dict[int, int] | None = None,
    ) -> list[tuple[int, float]]:
        last_seen = last_seen_override or self._last_seen(particle, self.public_last_seen)
        return [
            (card_id, weight)
            for card_id in self.catalog.cards
            if card_id not in excluded
            and (weight := self.strategy.belief._return_weight_at(last_seen.get(card_id), action_index)) > 0.0
        ]

    def _choice(
        self,
        pool: list[tuple[int, float]],
        particle_index: int,
        label: Any,
    ) -> int | None:
        total = sum(weight for _card_id, weight in pool)
        if total <= 0.0:
            return None
        needle = _stable_uniform(self.seed, particle_index, label) * total
        for card_id, weight in pool:
            needle -= weight
            if needle <= 0.0:
                return card_id
        return pool[-1][0]

    def _sample_without_replacement(
        self,
        particle: BeliefParticle,
        count: int,
        action_index: int,
        excluded: set[int],
        particle_index: int,
        label: Any,
    ) -> tuple[int, ...]:
        selected: list[int] = []
        for draw_index in range(count):
            pool = self._weighted_pool(particle, action_index, excluded | set(selected))
            card_id = self._choice(pool, particle_index, (label, draw_index))
            if card_id is None:
                break
            selected.append(card_id)
        return tuple(selected)

    def _reconcile_with_our_hand(self, our_hand: tuple[int, ...], action_index: int, label: Any) -> None:
        ours = set(our_hand)
        reconciled: list[BeliefParticle] = []
        for particle_index, particle in enumerate(self.particles):
            kept = tuple(card_id for card_id in particle.opponent_hand if card_id not in ours)
            missing = max(0, 6 - len(kept))
            if missing:
                fill = self._sample_without_replacement(
                    particle,
                    missing,
                    action_index,
                    ours | set(kept),
                    particle_index,
                    (label, "reconcile"),
                )
                kept += fill
            reconciled.append(BeliefParticle(kept[:6], particle.latent_last_seen))
        self.particles = reconciled

    def _latent_gap(self, until: int, our_hand: tuple[int, ...]) -> None:
        while self.action_index + 1 < until:
            self.action_index += 1
            self.unknown_transitions += 1
            updated: list[BeliefParticle] = []
            for particle_index, particle in enumerate(self.particles):
                last_seen = self._last_seen(particle, self.public_last_seen)
                pool = self._weighted_pool(
                    particle,
                    self.action_index,
                    set(our_hand) | set(particle.opponent_hand),
                    last_seen_override=last_seen,
                )
                moved = self._choice(pool, particle_index, ("latent", self.action_index))
                latent = dict(particle.latent_last_seen)
                if moved is not None:
                    latent[moved] = self.action_index
                updated.append(BeliefParticle(particle.opponent_hand, tuple(sorted(latent.items()))))
            self.particles = updated

    def advance(
        self,
        before: Any,
        after: Any,
        *,
        actor: str,
        action: str,
        card_id: int,
        action_index: int,
    ) -> None:
        self._latent_gap(action_index, tuple(before.hand))
        if actor == "opponent":
            self.conditioned_actions += 1
            compatible: list[BeliefParticle] = []
            for particle in self.particles:
                hand = list(particle.opponent_hand)
                if card_id in hand:
                    hand.remove(card_id)
                    compatible.append(BeliefParticle(tuple(hand), particle.latent_last_seen))
            # Sequential importance resampling: incompatible particles have
            # zero weight.  Resample the actually compatible five-card states,
            # then give each clone its own replacement draw.  This preserves
            # the five-card posterior instead of rebuilding it independently.
            if not compatible:
                template = self.particles[0]
                hand = self._sample_without_replacement(
                    template,
                    5,
                    action_index - 1,
                    set(before.hand) | {card_id},
                    0,
                    ("conditional-fallback", action_index, card_id),
                )
                compatible = [BeliefParticle(hand, template.latent_last_seen)]
            self.rejuvenated_particles += max(0, len(self.particles) - len(compatible))
            conditioned: list[BeliefParticle] = []
            for particle_index in range(len(self.particles)):
                source_index = min(
                    len(compatible) - 1,
                    int(_stable_uniform(self.seed, particle_index, "resample", action_index) * len(compatible)),
                )
                particle = compatible[source_index]
                hand = list(particle.opponent_hand)
                last_seen = self._last_seen(particle, self.public_last_seen)
                last_seen[card_id] = action_index
                interim = BeliefParticle(tuple(hand), particle.latent_last_seen)
                pool = self._weighted_pool(
                    interim,
                    action_index,
                    set(after.hand) | set(hand),
                    last_seen_override=last_seen,
                )
                replacement = self._choice(pool, particle_index, ("opponent-draw", action_index))
                if replacement is not None:
                    hand.append(replacement)
                conditioned.append(BeliefParticle(tuple(hand[:6]), particle.latent_last_seen))
            self.particles = conditioned
        self.public_last_seen[card_id] = action_index
        self.action_index = max(self.action_index, action_index)
        self._reconcile_with_our_hand(tuple(after.hand), self.action_index, ("after", action_index))

    def snapshot(self, state: Any) -> tuple[list[BeliefParticle], dict[int, int]]:
        if int(state.turn) > self.action_index:
            self._latent_gap(int(state.turn) + 1, tuple(state.hand))
        self._reconcile_with_our_hand(tuple(state.hand), int(state.turn), ("snapshot", state.turn))
        return list(self.particles), dict(self.public_last_seen)


class FastDiscardOracle:
    """High-particle counterfactual evaluator with PLAY/DISCARD parity.

    Both root actions use the same sequence:
      action -> particle-conditioned replacement -> opponent best reply ->
      our best next action -> calibrated positional leaf.

    The root draw uses common random numbers.  No action-type bonus or fixed
    discard-retention penalty is present.
    """

    FORCED_EXTRA_DISCARD = {100, 101}

    def __init__(
        self,
        module: Any,
        strategy: Any,
        state: Any,
        particles: list[BeliefParticle],
        public_last_seen: dict[int, int],
        *,
        seed: str,
        allowed_actions: set[tuple[str, int]] | None = None,
    ) -> None:
        self.module = module
        self.strategy = strategy
        self.state = state
        self.particles = particles
        self.public_last_seen = public_last_seen
        self.seed = seed
        self.catalog = strategy.catalog
        self.extra_cards = set(strategy.EXTRA_TURN_CARDS)
        self.allowed_actions = allowed_actions
        self._next_cache: dict[tuple[Any, ...], float] = {}
        self._reply_cache: dict[tuple[Any, ...], float] = {}
        self._category_cache: dict[tuple[int, Any, Any], str] = {}

    def _actions(self, hand: tuple[int, ...], player: Any, *, must_discard: bool = False) -> list[SearchAction]:
        actions: list[SearchAction] = []
        for slot, card_id in enumerate(hand):
            card = self.catalog.cards.get(card_id)
            if card is None:
                continue
            actions.append(SearchAction("drop", slot, card_id))
            if not must_discard and self.strategy._affordable(card, player):
                actions.append(SearchAction("turn", slot, card_id))
        return actions

    def _simulate(self, me: Any, enemy: Any, actor: str, card_id: int) -> tuple[Any, Any]:
        card = self.catalog[card_id]
        if actor == "us":
            return self.strategy.simulate(card, self.strategy._state_with(self.state, me, enemy))
        next_enemy, next_me = self.strategy.simulate(
            card,
            self.strategy._state_with(self.state, me, enemy, opponent_actor=True),
        )
        return next_me, next_enemy

    def _draw(
        self,
        our_hand: tuple[int, ...],
        opponent_hand: tuple[int, ...],
        last_seen: dict[int, int],
        action_index: int,
        particle_index: int,
        label: Any,
    ) -> int | None:
        excluded = set(our_hand) | set(opponent_hand)
        pool = [
            (card_id, weight)
            for card_id in self.catalog.cards
            if card_id not in excluded
            and (weight := self.strategy.belief._return_weight_at(last_seen.get(card_id), action_index)) > 0.0
        ]
        total = sum(weight for _card_id, weight in pool)
        if total <= 0.0:
            return None
        needle = _stable_uniform(self.seed, particle_index, label) * total
        for card_id, weight in pool:
            needle -= weight
            if needle <= 0.0:
                return card_id
        return pool[-1][0]

    def _transition(
        self,
        me: Any,
        enemy: Any,
        our_hand: tuple[int, ...],
        opponent_hand: tuple[int, ...],
        last_seen: dict[int, int],
        action_index: int,
        actor: str,
        action: SearchAction,
        particle_index: int,
        label: Any,
    ) -> tuple[Any, Any, tuple[int, ...], tuple[int, ...], dict[int, int], int | None, bool]:
        target = list(our_hand if actor == "us" else opponent_hand)
        if action.slot >= len(target) or target[action.slot] != action.card_id:
            raise ValueError("action/slot does not match hand")
        if action.action == "turn":
            me, enemy = self._simulate(me, enemy, actor, action.card_id)
        target.pop(action.slot)
        next_index = action_index + 1
        next_seen = dict(last_seen)
        next_seen[action.card_id] = next_index
        interim_ours = tuple(target) if actor == "us" else our_hand
        interim_opponent = opponent_hand if actor == "us" else tuple(target)
        replacement = self._draw(
            interim_ours,
            interim_opponent,
            next_seen,
            next_index,
            particle_index,
            label,
        )
        if replacement is not None:
            target.insert(min(action.slot, len(target)), replacement)
        if actor == "us":
            our_hand = tuple(target)
        else:
            opponent_hand = tuple(target)
        extra = action.action == "turn" and action.card_id in self.extra_cards
        return me, enemy, our_hand, opponent_hand, next_seen, replacement, extra

    def _terminal(self, me: Any, enemy: Any) -> float | None:
        if self.strategy._won(me, enemy, self.state):
            return 1.0
        if self.strategy._lost(me, enemy, self.state):
            return 0.0
        return None

    def _best_our_next(self, me: Any, enemy: Any, our_hand: tuple[int, ...]) -> float:
        me = self.strategy._income(me)
        terminal = self._terminal(me, enemy)
        if terminal is not None:
            return terminal
        key = (me, enemy, our_hand)
        cached = self._next_cache.get(key)
        if cached is not None:
            return cached
        best = self.strategy._position_probability(me, enemy, self.state)
        for action in self._actions(our_hand, me):
            if action.action != "turn":
                continue
            next_me, next_enemy = self._simulate(me, enemy, "us", action.card_id)
            terminal = self._terminal(next_me, next_enemy)
            value = terminal if terminal is not None else self.strategy._position_probability(
                next_me, next_enemy, self.state
            )
            best = max(best, value)
        self._next_cache[key] = best
        return best

    def _opponent_reply(
        self,
        me: Any,
        enemy: Any,
        our_hand: tuple[int, ...],
        opponent_hand: tuple[int, ...],
        last_seen: dict[int, int],
        action_index: int,
        particle_index: int,
        path: Any,
    ) -> float:
        enemy = self.strategy._income(enemy)
        if self.strategy._won(enemy, me, self.strategy._state_with(self.state, me, enemy, opponent_actor=True)):
            return 0.0
        values: list[float] = []
        # All opponent discards have the same visible state at the requested
        # two-ply horizon.  The replacement is still part of the persistent
        # transition, but cannot affect our immediately following action.
        if opponent_hand:
            values.append(self._best_our_next(me, enemy, our_hand))
        for reply in self._actions(opponent_hand, enemy):
            if reply.action == "drop":
                continue
            # Opponent discards have identical visible value at this horizon,
            # but still execute the real one-slot replacement transition.
            cache_key = (me, enemy, our_hand, reply.action, reply.card_id)
            cached = self._reply_cache.get(cache_key)
            if cached is not None and reply.action == "turn" and reply.card_id not in self.extra_cards:
                values.append(cached)
                continue
            (
                next_me,
                next_enemy,
                next_ours,
                next_opponent,
                next_seen,
                _replacement,
                extra,
            ) = self._transition(
                me,
                enemy,
                our_hand,
                opponent_hand,
                last_seen,
                action_index,
                "opponent",
                reply,
                particle_index,
                (path, "opponent", reply.action, reply.slot),
            )
            terminal = self._terminal(next_me, next_enemy)
            if terminal is not None:
                value = terminal
            elif extra:
                value = self._extra_value(
                    next_me,
                    next_enemy,
                    next_ours,
                    next_opponent,
                    next_seen,
                    action_index + 1,
                    "opponent",
                    particle_index,
                    (path, "opponent-extra", reply.slot),
                    2,
                    reply.card_id in self.FORCED_EXTRA_DISCARD,
                )
            else:
                value = self._best_our_next(next_me, next_enemy, next_ours)
            if reply.action == "turn" and reply.card_id not in self.extra_cards:
                self._reply_cache[cache_key] = value
            values.append(value)
        return min(values) if values else self._best_our_next(me, enemy, our_hand)

    def _extra_value(
        self,
        me: Any,
        enemy: Any,
        our_hand: tuple[int, ...],
        opponent_hand: tuple[int, ...],
        last_seen: dict[int, int],
        action_index: int,
        actor: str,
        particle_index: int,
        path: Any,
        depth: int,
        forced_discard: bool = False,
    ) -> float:
        if depth <= 0:
            return self._opponent_reply(
                me, enemy, our_hand, opponent_hand, last_seen, action_index, particle_index, path
            ) if actor == "us" else self._best_our_next(me, enemy, our_hand)
        player = me if actor == "us" else enemy
        hand = our_hand if actor == "us" else opponent_hand
        actions = self._actions(hand, player, must_discard=forced_discard)
        if forced_discard:
            actions = [action for action in actions if action.action == "drop"]
        values: list[float] = []
        for action in actions:
            transitioned = self._transition(
                me,
                enemy,
                our_hand,
                opponent_hand,
                last_seen,
                action_index,
                actor,
                action,
                particle_index,
                (path, actor, action.action, action.slot),
            )
            next_me, next_enemy, next_ours, next_opponent, next_seen, _replacement, extra = transitioned
            terminal = self._terminal(next_me, next_enemy)
            if terminal is not None:
                value = terminal
            elif forced_discard:
                value = self._extra_value(
                    next_me,
                    next_enemy,
                    next_ours,
                    next_opponent,
                    next_seen,
                    action_index + 1,
                    actor,
                    particle_index,
                    (path, "forced-done"),
                    depth - 1,
                )
            elif extra:
                value = self._extra_value(
                    next_me,
                    next_enemy,
                    next_ours,
                    next_opponent,
                    next_seen,
                    action_index + 1,
                    actor,
                    particle_index,
                    (path, "extra"),
                    depth - 1,
                    action.card_id in self.FORCED_EXTRA_DISCARD,
                )
            elif actor == "us":
                value = self._opponent_reply(
                    next_me,
                    next_enemy,
                    next_ours,
                    next_opponent,
                    next_seen,
                    action_index + 1,
                    particle_index,
                    (path, "pass"),
                )
            else:
                value = self._best_our_next(next_me, next_enemy, next_ours)
            values.append(value)
        fallback = self.strategy._position_probability(me, enemy, self.state)
        return (max(values) if actor == "us" else min(values)) if values else fallback

    def _replacement_category(self, card_id: int, me: Any, enemy: Any) -> str:
        cache_key = (card_id, me, enemy)
        cached = self._category_cache.get(cache_key)
        if cached is not None:
            return cached
        card = self.catalog[card_id]
        future_me = self.strategy._income(me)
        if self.strategy._affordable(card, future_me):
            next_me, next_enemy = self._simulate(future_me, enemy, "us", card_id)
            if self.strategy._won(next_me, next_enemy, self.state):
                result = "immediate_win"
                self._category_cache[cache_key] = result
                return result
            before = self.strategy._position_probability(future_me, enemy, self.state)
            after = self.strategy._position_probability(next_me, next_enemy, self.state)
            if after - before >= 0.10:
                result = "strong_finisher_or_defense"
                self._category_cache[cache_key] = result
                return result
            if after - before >= 0.025:
                result = "useful_playable"
                self._category_cache[cache_key] = result
                return result
            result = "neutral"
            self._category_cache[cache_key] = result
            return result
        eta = self.strategy._turns_until_affordable(card, future_me)
        result = "neutral" if eta is not None and eta <= 2 else "bad_or_dead"
        self._category_cache[cache_key] = result
        return result

    def evaluate(self) -> list[SearchDecision]:
        root_actions = self._actions(tuple(self.state.hand), self.state.me, must_discard=self.state.must_discard)
        if self.allowed_actions is not None:
            root_actions = [
                action for action in root_actions if (action.action, action.slot) in self.allowed_actions
            ]
        decisions: list[SearchDecision] = []
        for root in root_actions:
            outcomes: list[float] = []
            replacements: list[int] = []
            categories: Counter[str] = Counter()
            for particle_index, particle in enumerate(self.particles):
                last_seen = dict(self.public_last_seen)
                last_seen.update(dict(particle.latent_last_seen))
                transitioned = self._transition(
                    self.state.me,
                    self.state.opponent,
                    tuple(self.state.hand),
                    particle.opponent_hand,
                    last_seen,
                    int(self.state.turn),
                    "us",
                    root,
                    particle_index,
                    ("root-draw",),
                )
                me, enemy, our_hand, opponent_hand, next_seen, replacement, extra = transitioned
                if replacement is not None:
                    replacements.append(replacement)
                    categories[self._replacement_category(replacement, me, enemy)] += 1
                terminal = self._terminal(me, enemy)
                if terminal is not None:
                    value = terminal
                elif extra:
                    value = self._extra_value(
                        me,
                        enemy,
                        our_hand,
                        opponent_hand,
                        next_seen,
                        int(self.state.turn) + 1,
                        "us",
                        particle_index,
                        ("root-extra", root.slot),
                        2,
                        root.card_id in self.FORCED_EXTRA_DISCARD,
                    )
                else:
                    value = self._opponent_reply(
                        me,
                        enemy,
                        our_hand,
                        opponent_hand,
                        next_seen,
                        int(self.state.turn) + 1,
                        particle_index,
                        ("root-reply", root.slot),
                    )
                outcomes.append(max(0.0, min(1.0, value)))
            expected = statistics.fmean(outcomes)
            tail_count = max(1, math.ceil(len(outcomes) * 0.10))
            tail_survival = statistics.fmean(sorted(outcomes)[:tail_count])
            denominator = max(1, len(replacements))
            decisions.append(
                SearchDecision(
                    root.action,
                    root.slot,
                    root.card_id,
                    expected,
                    outcomes,
                    replacements,
                    {name: count / denominator for name, count in sorted(categories.items())},
                    tail_survival,
                )
            )
        # No PLAY-over-DISCARD type preference.  Exact ties use tail survival,
        # then preserve the earlier slot as stable deterministic ordering.
        decisions.sort(key=lambda item: (item.p_win, item.tail_survival, -item.slot, item.card_id), reverse=True)
        if len(decisions) >= 2:
            best, runner = decisions[:2]
            differences = [left - right for left, right in zip(best.outcomes, runner.outcomes)]
            if len(differences) >= 2:
                best.paired_se = statistics.pstdev(differences) / math.sqrt(len(differences))
                margin = best.p_win - runner.p_win
                best.ci_diff = (
                    margin - 1.96 * best.paired_se,
                    margin + 1.96 * best.paired_se,
                )
        return decisions
