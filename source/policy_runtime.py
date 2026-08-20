from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any


def _resource_path(name: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root / name


class PolicyRuntime:
    """Small stdlib-only inference runtime for the offline 3.9 models."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.horizon_model = payload["horizon"]
        self.value_model = payload["state_value"]
        self.action_model = payload["action_policy"]

    @classmethod
    def load(cls, path: Path | None = None) -> "PolicyRuntime | None":
        try:
            payload = json.loads((path or _resource_path("policy_models.json")).read_text(encoding="utf-8-sig"))
            return cls(payload)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    @staticmethod
    def _cost(card: Any) -> tuple[str, int]:
        if card.ore:
            return "ore", card.ore
        if card.mana:
            return "mana", card.mana
        return "army", card.army

    @classmethod
    def _eta(cls, card: Any, me: Any) -> int:
        resource, cost = cls._cost(card)
        current = getattr(me, resource)
        if current >= cost:
            return 0
        producer = {"ore": "mine", "mana": "monastery", "army": "barracks"}[resource]
        income = getattr(me, producer)
        return 99 if income <= 0 else math.ceil((cost - current) / income)

    def _base_features(self, strategy: Any, state: Any, me: Any, enemy: Any, hand: list[int]) -> list[float]:
        tower_o, tower_e = me.tower, enemy.tower
        td_o, td_e = max(0, 50 - tower_o), max(0, 50 - tower_e)
        resources_o = [me.ore, me.mana, me.army]
        resources_e = [enemy.ore, enemy.mana, enemy.army]
        producers_o = [me.mine, me.monastery, me.barracks]
        producers_e = [enemy.mine, enemy.monastery, enemy.barracks]
        cards = [strategy.catalog.cards[card_id] for card_id in hand if card_id in strategy.catalog.cards]
        etas = [min(12, self._eta(card, me)) for card in cards] or [12]
        types = Counter(self._cost(card)[0] for card in cards)
        playable = sum(strategy._affordable(card, me) for card in cards)
        first_actor = getattr(state, "first_actor", "unknown")
        values = {
            "turn": min(160, state.turn), "our_tower": tower_o, "enemy_tower": tower_e,
            "our_wall": me.wall, "enemy_wall": enemy.wall,
            "tower_distance_ours": td_o, "tower_distance_enemy": td_e,
            "destroy_distance_ours": tower_o, "destroy_distance_enemy": tower_e,
            "tower_distance_min": min(td_o, td_e, tower_o, tower_e),
            "tower_race_diff": td_e - td_o,
            "tower_distance_ours_sq": td_o * td_o / 50,
            "tower_distance_enemy_sq": td_e * td_e / 50,
            "our_ore": me.ore, "our_mana": me.mana, "our_army": me.army,
            "enemy_ore": enemy.ore, "enemy_mana": enemy.mana, "enemy_army": enemy.army,
            "resource_sum_ours": sum(resources_o), "resource_sum_enemy": sum(resources_e),
            "resource_min_ours": min(resources_o), "resource_min_enemy": min(resources_e),
            "resource_diff": sum(resources_o) - sum(resources_e),
            "our_mine": me.mine, "our_monastery": me.monastery, "our_barracks": me.barracks,
            "enemy_mine": enemy.mine, "enemy_monastery": enemy.monastery, "enemy_barracks": enemy.barracks,
            "production_sum_ours": sum(producers_o), "production_sum_enemy": sum(producers_e),
            "production_diff": sum(producers_o) - sum(producers_e),
            "playable_count": playable, "dead_count": max(0, 6 - playable),
            "hand_eta_mean": sum(etas) / len(etas), "hand_eta_best": min(etas),
            "same_resource_max": max(types.values(), default=0),
            "must_discard": int(bool(state.must_discard)),
            "first_mover": int(first_actor == "us"),
            "reconnect": int(bool(getattr(state, "reconnect_uncertainty", False))),
            "unknown_transitions": min(10, int(getattr(state, "unknown_transitions", 0))),
            "terminal_pressure": 1 / (1 + min(td_o, td_e, tower_o, tower_e)),
            "wall_fraction_ours": me.wall / max(1, tower_o + me.wall),
            "wall_fraction_enemy": enemy.wall / max(1, tower_e + enemy.wall),
        }
        return [float(values[name]) for name in self.horizon_model["feature_names"]]

    def horizon(self, strategy: Any, state: Any, me: Any, enemy: Any, hand: list[int]) -> float:
        model = self.horizon_model
        raw = self._base_features(strategy, state, me, enemy, hand)
        normalized = [(value - center) / scale for value, center, scale in zip(raw, model["center"], model["scale"])]
        result = model["coefficients_with_intercept"][0] + self._dot(normalized, model["coefficients_with_intercept"][1:])
        if model["target_transform"] == "log1p":
            result = math.expm1(result)
        return max(0.0, min(180.0, result))

    def state_feature_vector(self, strategy: Any, state: Any, me: Any, enemy: Any, hand: list[int]) -> list[float]:
        base = self._base_features(strategy, state, me, enemy, hand)
        h = self.horizon(strategy, state, me, enemy, hand)
        td_us, td_en = max(0, 50 - me.tower), max(0, 50 - enemy.tower)
        hp_us = me.tower + .65 * min(12, me.wall)
        hp_en = enemy.tower + .65 * min(12, enemy.wall)
        r_us, r_en = min(me.ore, me.mana, me.army), min(enemy.ore, enemy.mana, enemy.army)
        production_diff = (me.mine + me.monastery + me.barracks) - (enemy.mine + enemy.monastery + enemy.barracks)
        short = math.exp(-h / 8)
        terminal = 1 / (1 + min(td_us, td_en, me.tower, enemy.tower))
        initiative = 1 if getattr(state, "first_actor", "unknown") == "us" else -1
        ordinary_threat = min(1.0, enemy.army / 18)
        return base + [
            h / 80, short, 1 - short,
            math.exp(-td_us / 6), math.exp(-td_en / 6),
            math.exp(-hp_en / 11), math.exp(-hp_us / 11),
            math.exp(-(150 - min(150, r_us)) / 30), math.exp(-(150 - min(150, r_en)) / 30),
            (td_en - td_us) * short / 20, production_diff * min(h, 60) / 180,
            min(12, me.wall) * ordinary_threat / 12, initiative * terminal,
            math.sqrt(max(0, state.turn)) / 12,
        ]

    def state_pwin(self, strategy: Any, state: Any, me: Any, enemy: Any, hand: list[int]) -> float:
        if me.tower >= 50 or enemy.tower <= 0 or min(me.ore, me.mana, me.army) >= 150:
            return 1.0
        if me.tower <= 0 or enemy.tower >= 50 or min(enemy.ore, enemy.mana, enemy.army) >= 150:
            return 0.0
        model = self.value_model
        raw = self.state_feature_vector(strategy, state, me, enemy, hand)
        normalized = [(value - center) / scale for value, center, scale in zip(raw, model["center"], model["scale"])]
        logit = model["coefficients_with_intercept"][0] + self._dot(normalized, model["coefficients_with_intercept"][1:])
        return self._sigmoid(model["platt_intercept"] + model["platt_slope"] * logit)

    def _action_base(self, strategy: Any, state: Any, action: str, card: Any) -> list[float]:
        me0, en0 = state.me, state.opponent
        me1, en1 = strategy.simulate(card, state) if action == "turn" else (me0, en0)
        dt_me, dt_en = me1.tower - me0.tower, en1.tower - en0.tower
        dw_me, dw_en = me1.wall - me0.wall, en1.wall - en0.wall
        dr_me = (me1.ore + me1.mana + me1.army) - (me0.ore + me0.mana + me0.army)
        dr_en = (en1.ore + en1.mana + en1.army) - (en0.ore + en0.mana + en0.army)
        dp_me = (me1.mine + me1.monastery + me1.barracks) - (me0.mine + me0.monastery + me0.barracks)
        dp_en = (en1.mine + en1.monastery + en1.barracks) - (en0.mine + en0.monastery + en0.barracks)
        effect = card.effect.lower().replace("ё", "е")
        ordinary = strategy.GENERAL_DAMAGE.get(card.id, 0) if action == "turn" else 0
        absorbed = min(en0.wall, ordinary)
        terminal = min(50 - me0.tower, en0.tower, 50 - en0.tower, me0.tower) <= 10
        early = state.turn <= 20
        producer = int(card.id in strategy.PRODUCTION_CARDS and action == "turn")
        extra = int(card.id in strategy.EXTRA_TURN_CARDS and action == "turn")
        self_damage = int(action == "turn" and (dt_me < 0 or "вы теряете" in effect or "вашей башне" in effect))
        symmetric = int(action == "turn" and ("все " in effect or "обоих" in effect))
        resource, _cost = self._cost(card)
        current = getattr(me0, resource)
        producer_name = {"ore": "mine", "mana": "monastery", "army": "barracks"}[resource]
        eta = max(0, math.ceil((card.total_cost - current) / max(1, getattr(me0, producer_name))))
        win = me1.tower >= 50 or en1.tower <= 0 or min(me1.ore, me1.mana, me1.army) >= 150
        return [
            int(action == "turn"), int(action == "drop"), int(win), int(me1.tower <= 0),
            dt_me / 20, dt_en / 20, dw_me / 20, dw_en / 20, dr_me / 30, dr_en / 30,
            dp_me / 3, dp_en / 3,
            ((50 - me0.tower) - (50 - me1.tower) - ((50 - en0.tower) - (50 - en1.tower))) / 20,
            ((en0.tower + .65 * en0.wall) - (en1.tower + .65 * en1.wall)) / 20,
            ((min(me1.ore, me1.mana, me1.army) - min(me0.ore, me0.mana, me0.army)) - (min(en1.ore, en1.mana, en1.army) - min(en0.ore, en0.mana, en0.army))) / 20,
            absorbed / 10, card.ore / 20, card.mana / 20, card.army / 20, (card.total_cost / 20) ** 2,
            extra, producer, symmetric, self_damage, producer * early, producer * (not early),
            terminal * dt_me / 10, terminal * (-dt_en) / 10,
            (me0.tower <= 10) * (dt_me + max(0, dw_me)) / 10,
            (en0.tower >= 40) * (-dt_en) / 10, min(10, eta) / 10,
        ]

    def action_feature_vector(self, strategy: Any, state: Any, action: str, slot: int) -> list[float]:
        card = strategy.catalog[state.hand[slot]]
        me1, enemy1 = strategy.simulate(card, state) if action == "turn" else (state.me, state.opponent)
        retained = [card_id for index, card_id in enumerate(state.hand) if index != slot]
        retained_cards = [strategy.catalog[card_id] for card_id in retained]
        etas = [min(20, self._eta(candidate, me1)) for candidate in retained_cards] or [20]
        playable = sum(strategy._affordable(candidate, me1) for candidate in retained_cards)
        types = Counter(
            "ore" if candidate.ore else "mana" if candidate.mana else "army" if candidate.army else "free"
            for candidate in retained_cards
        )
        post_state = replace(
            state,
            players={state.player_no: me1, state.opponent_no: enemy1},
            hand=retained,
            must_discard=False,
        )
        horizon = self.horizon(strategy, post_state, me1, enemy1, retained)
        short = math.exp(-horizon / 8)
        extra = [
            self.state_pwin(strategy, post_state, me1, enemy1, retained), playable / 5,
            min(etas) / 20, (sum(etas) / len(etas)) / 20,
            sum(candidate.id in strategy.PRODUCTION_CARDS for candidate in retained_cards) / 5,
            sum(candidate.id in strategy.EXTRA_TURN_CARDS for candidate in retained_cards) / 5,
            max(types.values(), default=0) / 5,
            int(action == "drop" and (card.id in strategy.DIRECT_TOWER_DAMAGE or card.id in strategy.GENERAL_DAMAGE)),
            int(action == "drop" and card.id in strategy.PRODUCTION_CARDS),
            int(action == "drop" and card.id in strategy.EXTRA_TURN_CARDS),
            int(action == "turn") * short, int(action == "drop") * short,
            card.total_cost / 20 * short,
        ]
        return self._action_base(strategy, state, action, card) + extra

    def action_score(self, strategy: Any, state: Any, action: str, slot: int) -> float:
        return self._dot(
            self.action_feature_vector(strategy, state, action, slot),
            self.action_model["weights"],
        )
