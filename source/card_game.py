from __future__ import annotations

import html
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from policy_runtime import PolicyRuntime


BASE_URL = "https://www.heroeswm.ru"
CARD_ENDPOINT = BASE_URL + "/cardsgame.php"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)


class CardGameStopped(Exception):
    pass


class CardGameSessionExpired(Exception):
    pass


class CardGameStakeUnavailable(Exception):
    pass


def _resource_path(name: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root / name


def decode_response(response: requests.Response) -> str:
    return response.content.decode("cp1251", errors="replace")


@dataclass(frozen=True)
class CardDefinition:
    id: int
    name: str
    ore: int
    mana: int
    army: int
    effect: str
    image: str = ""

    @property
    def total_cost(self) -> int:
        return self.ore + self.mana + self.army


@dataclass(frozen=True)
class PlayerState:
    ore: int
    mana: int
    army: int
    tower: int
    wall: int
    mine: int
    monastery: int
    barracks: int


@dataclass
class GameState:
    game_id: int
    turn: int
    is_your_turn: bool
    player_no: int
    time_left: int
    players: dict[int, PlayerState]
    hand: list[int]
    winner: int = 0
    finish_reason: int = 0
    last_move: str = ""
    now_player: int = 0
    table: str = ""
    must_discard: bool = False
    tower_goal: int = 50
    resource_goal: int = 150
    stake: int = 0
    nicknames: dict[int, str] = field(default_factory=dict)
    player_ids: dict[int, str] = field(default_factory=dict)
    raw: str = ""
    first_actor: str = "unknown"
    reconnect_uncertainty: bool = False
    unknown_transitions: int = 0

    @property
    def finished(self) -> bool:
        return self.finish_reason > 0

    @property
    def me(self) -> PlayerState:
        return self.players[self.player_no]

    @property
    def opponent_no(self) -> int:
        return 2 if self.player_no == 1 else 1

    @property
    def opponent(self) -> PlayerState:
        return self.players[self.opponent_no]


@dataclass(frozen=True)
class CardDecision:
    action: str
    slot: int
    card: CardDefinition
    score: float
    reasons: tuple[str, ...]
    immediate_score: float | None = None
    response_value: float | None = None
    predicted_me: dict[str, int] | None = None
    predicted_opponent: dict[str, int] | None = None
    winning_replies_before: int | None = None
    winning_replies_after: int | None = None
    discard_retention: float = 0.0
    legal_cost: str = ""
    immediate_state_delta: dict[str, int] | None = None
    p_win: float | None = None
    p_win_next_action: float | None = None
    p_lose_next_turn: float | None = None
    p_win_within_2_own_actions: float | None = None
    p_opponent_win_within_2_actions: float | None = None
    expected_reply_value: float | None = None
    tail_risk: float | None = None
    immediate_terminal_win: bool = False
    policy_rank: int = 0
    pwin_rank: int = 0
    final_rank_reason: str = ""
    resource_unlocks: tuple[str, ...] = ()
    cards_unlocked_next_turn: tuple[str, ...] = ()
    eta_key_hand_cards: dict[str, int | None] | None = None
    extra_turn_continuation: str = ""
    opponent_belief_top_threats: tuple[str, ...] = ()
    particle_count: int = 0
    particle_limit: int = 0
    particles_requested: int = 0
    particles_completed: int = 0
    stopping_reason: str = ""
    deadline_remaining: float | None = None
    decision_margin: float | None = None
    se_diff: float | None = None
    ci_diff: tuple[float, float] | None = None
    policy_score_margin: float | None = None
    policy_score_mc_se_diff: float | None = None
    policy_score_mc_ci_diff: tuple[float, float] | None = None
    stopping_objective: str = ""
    model_policy_uncertainty: str = ""
    confidence_interval_pp: float | None = None
    decision_uncertain: bool = False
    analysis_deadline_hit: bool = False
    sampling_batches: int = 0
    random_seed: str = ""
    analysis_seconds: float | None = None
    analysis_time_budget: float | None = None
    replacement_distribution: dict[str, float] | None = None
    hand_diagnostics: dict[str, Any] | None = None
    policy_score: float | None = None


class CardCatalog:
    def __init__(self, cards: Iterable[CardDefinition]) -> None:
        self.cards = {card.id: card for card in cards}
        if len(self.cards) != 102:
            raise ValueError(f"Ожидалось 102 карты, получено {len(self.cards)}")

    @classmethod
    def load(cls, path: Path | None = None) -> "CardCatalog":
        source = path or _resource_path("cards_catalog.json")
        raw = json.loads(source.read_text(encoding="utf-8"))
        return cls(
            CardDefinition(
                id=int(item["id"]),
                name=html.unescape(str(item["name"])).replace("\xa0", " "),
                ore=int(item.get("ore", 0)),
                mana=int(item.get("mana", 0)),
                army=int(item.get("army", 0)),
                effect=html.unescape(str(item.get("effect", ""))).replace("\xa0", " "),
                image=str(item.get("image", "")),
            )
            for item in raw
        )

    def __getitem__(self, card_id: int) -> CardDefinition:
        return self.cards[card_id]


class CardGameProtocol:
    """Parser for the compact pipe-separated response used by arcomage.js."""

    def __init__(self, game_id: int) -> None:
        self.game_id = game_id
        self.nicknames: dict[int, str] = {}
        self.player_ids: dict[int, str] = {}
        self.tower_goal = 50
        self.resource_goal = 150
        self.stake = 0

    @staticmethod
    def _integer(value: str, default: int = 0) -> int:
        try:
            return int(float(value or default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _nickname(value: str) -> tuple[str, str]:
        separator = "#" if "#" in value else "*"
        if separator in value:
            name, player_id = value.rsplit(separator, 1)
            return name, player_id
        return value, ""

    def parse(self, payload: str) -> GameState:
        values = payload.strip().split("|")
        if len(values) < 25:
            raise ValueError("Слишком короткий ответ карточной игры")
        index = 0
        if not re.fullmatch(r"-?\d+", values[0]):
            name1, id1 = self._nickname(values[0])
            name2, id2 = self._nickname(values[1])
            self.nicknames = {1: name1, 2: name2}
            self.player_ids = {1: id1, 2: id2}
            self.tower_goal = self._integer(values[2], 50)
            self.resource_goal = self._integer(values[3], 150)
            self.stake = self._integer(values[4])
            index = 5

        def take(default: str = "") -> str:
            nonlocal index
            value = values[index] if index < len(values) else default
            index += 1
            return value

        turn = self._integer(take())
        is_your_turn = bool(self._integer(take()))
        player_no = self._integer(take())
        time_left = self._integer(take())

        mine1, mine2 = self._integer(take()), self._integer(take())
        monastery1, monastery2 = self._integer(take()), self._integer(take())
        barracks1, barracks2 = self._integer(take()), self._integer(take())
        ore1, ore2 = self._integer(take()), self._integer(take())
        mana1, mana2 = self._integer(take()), self._integer(take())
        army1, army2 = self._integer(take()), self._integer(take())
        wall1, wall2 = self._integer(take()), self._integer(take())
        tower1, tower2 = self._integer(take()), self._integer(take())
        hand = [self._integer(item, -1) for item in take().split("-") if item != ""]
        winner = self._integer(take())
        finish_reason = self._integer(take())
        last_move = take()
        now_player = self._integer(take())
        table = take()
        must_discard = bool(self._integer(take()))

        if player_no not in {1, 2}:
            # Spectator responses use 0. The bot never acts on such a state.
            player_no = 1
            is_your_turn = False

        return GameState(
            game_id=self.game_id,
            turn=turn,
            is_your_turn=is_your_turn,
            player_no=player_no,
            time_left=max(0, time_left),
            players={
                1: PlayerState(ore1, mana1, army1, tower1, wall1, mine1, monastery1, barracks1),
                2: PlayerState(ore2, mana2, army2, tower2, wall2, mine2, monastery2, barracks2),
            },
            hand=hand,
            winner=winner,
            finish_reason=finish_reason,
            last_move=last_move,
            now_player=now_player,
            table=table,
            must_discard=must_discard,
            tower_goal=self.tower_goal,
            resource_goal=self.resource_goal,
            stake=self.stake,
            nicknames=dict(self.nicknames),
            player_ids=dict(self.player_ids),
            raw=payload,
        )


class CardGameLearner:
    """Neutral usage counters retained for offline analysis.

    A final win/loss is deliberately not used to rate every card in that game:
    doing so punishes good moves from a lost match and is false learning.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"version": 1, "games": 0, "cards": {}}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("cards"), dict):
                self.data = loaded
        except (OSError, ValueError, TypeError):
            pass

    def preference(self, card_id: int) -> float:
        return 0.0

    def record(self, actions: list[tuple[str, int]], won: bool) -> None:
        cards = self.data.setdefault("cards", {})
        for action, card_id in actions:
            item = cards.setdefault(str(card_id), {"plays": 0, "drops": 0, "wins_seen": 0, "losses_seen": 0})
            item["plays" if action == "turn" else "drops"] = int(
                item.get("plays" if action == "turn" else "drops", 0)
            ) + 1
            result_key = "wins_seen" if won else "losses_seen"
            item[result_key] = int(item.get(result_key, 0)) + 1
        self.data["games"] = int(self.data.get("games", 0)) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


class CardStrategy:
    DIRECT_TOWER_DAMAGE = {
        35: 1, 39: 2, 41: 3, 43: 5, 49: 9, 52: 7, 53: 4, 59: 0,
        64: 2, 65: 8, 72: 3, 73: 2, 76: 4, 88: 6, 89: 5, 99: 12,
    }
    GENERAL_DAMAGE = {
        26: 10, 32: 6, 59: 6, 68: 2, 69: 4, 71: 6, 74: 5, 75: 4,
        78: 2, 80: 6, 81: 7, 82: 6, 83: 6, 84: 6, 85: 9, 86: 7, 87: 8,
        90: 8, 92: 10, 93: 10, 94: 20, 95: 2, 96: 3, 97: 8, 98: 13,
    }
    PRODUCTION_CARDS = {3, 4, 5, 10, 13, 17, 20, 31, 37, 40, 44, 45, 46, 48, 56, 70, 77, 79}
    EXTRA_TURN_CARDS = {1, 2, 12, 13, 34, 35, 68, 73, 100, 101}

    def __init__(self, catalog: CardCatalog, learner: CardGameLearner | None = None) -> None:
        self.catalog = catalog
        self.learner = learner
        self._simulate_cache: dict[
            tuple[int, int, PlayerState, PlayerState], tuple[PlayerState, PlayerState]
        ] = {}

    @staticmethod
    def _affordable(card: CardDefinition, player: PlayerState) -> bool:
        return card.ore <= player.ore and card.mana <= player.mana and card.army <= player.army

    def _damage(self, card_id: int, state: GameState) -> tuple[int, int]:
        tower = self.DIRECT_TOWER_DAMAGE.get(card_id, 0)
        general = self.GENERAL_DAMAGE.get(card_id, 0)
        if card_id == 65 and state.me.tower <= state.opponent.wall:
            tower = 0
            general = 8
        elif card_id == 88 and state.me.wall <= state.opponent.wall:
            tower = 0
            general = 6
        elif card_id == 84 and state.opponent.wall == 0:
            general = 10
        elif card_id == 86 and state.opponent.wall > 10:
            general = 10
        elif card_id == 87 and state.me.monastery > state.opponent.monastery:
            general = 12
        elif card_id == 95 and state.me.wall > state.opponent.wall:
            general = 3
        return tower, general

    @staticmethod
    def _gain(effect: str, target: str) -> int:
        match = re.search(rf"\+(\d+)\s+к\s+{target}", effect, re.I)
        return int(match.group(1)) if match else 0

    def score(self, card: CardDefinition, state: GameState, *, for_discard: bool = False) -> tuple[float, list[str]]:
        me, enemy = state.me, state.opponent
        effect = card.effect.lower().replace("ё", "е")
        score = 0.0
        reasons: list[str] = []
        tower_damage, general_damage = self._damage(card.id, state)
        effective_tower_damage = tower_damage + max(0, general_damage - enemy.wall)

        if effective_tower_damage >= enemy.tower and effective_tower_damage > 0:
            score += 10000
            reasons.append("победный урон")
        else:
            damage_value = tower_damage * 4.2 + general_damage * (2.2 if enemy.wall else 3.5)
            score += damage_value
            if damage_value:
                reasons.append(f"урон {tower_damage or general_damage}")

        tower_gain = self._gain(effect, "башне")
        wall_gain = self._gain(effect, "стене")
        if tower_gain and me.tower + tower_gain >= state.tower_goal:
            score += 9000
            reasons.append("достраивает башню до победы")
        else:
            tower_weight = 3.5 + max(0, me.tower - state.tower_goal * 0.65) * 0.08
            score += tower_gain * tower_weight
            if tower_gain:
                reasons.append(f"башня +{tower_gain}")
        score += wall_gain * (1.15 if me.wall > 8 else 1.8)
        if wall_gain:
            reasons.append(f"стена +{wall_gain}")

        if card.id in self.PRODUCTION_CARDS:
            stage = max(0.35, 1.0 - state.turn / 75.0)
            score += 30 * stage
            reasons.append("рост производства")
        if card.id in self.EXTRA_TURN_CARDS:
            score += 18
            reasons.append("дополнительный ход")
        if card.id in {100, 101}:
            score += 8
            reasons.append("обновление руки")

        # Resource victory becomes realistic only when all three piles are close.
        minimum_resource = min(me.ore, me.mana, me.army)
        if minimum_resource >= state.resource_goal * 0.72:
            resource_gain = sum(int(x) for x in re.findall(r"\+(\d+)\s+(?:руды|маны|отряд)", effect))
            score += resource_gain * 4
            if resource_gain:
                reasons.append("приближает победу ресурсами")

        if "враг теряет" in effect or "врага" in effect and ("-1" in effect or "теряет" in effect):
            score += 9
            reasons.append("ослабляет противника")
        if "все игроки" in effect:
            score -= 5
        if "вы теряете" in effect or "вашей башне" in effect or "собственной башне" in effect:
            score -= 12
        if card.id == 7 and me.mine >= enemy.mine:
            score -= 30
        if card.id == 46 and me.monastery >= enemy.monastery:
            score -= 20
        if card.id == 33 and me.wall >= enemy.wall:
            score -= 25
        if card.id == 54 and me.ore < 10:
            score -= 12
        if card.id == 97 and me.tower <= 3:
            score -= 100

        score -= card.total_cost * (0.20 if state.turn < 35 else 0.10)
        if self.learner:
            learned = self.learner.preference(card.id)
            score += learned
            if abs(learned) >= 1:
                reasons.append(f"опыт {learned:+.1f}")

        if for_discard and not self._affordable(card, me):
            production = max(1, me.mine if card.ore else me.monastery if card.mana else me.barracks)
            current = me.ore if card.ore else me.mana if card.mana else me.army
            wait_turns = max(0, math.ceil((card.total_cost - current) / production))
            score -= min(25, wait_turns * 3.5)
        return score, reasons

    def choose(self, state: GameState) -> CardDecision:
        cards = [(slot, self.catalog[card_id]) for slot, card_id in enumerate(state.hand) if card_id in self.catalog.cards]
        if state.must_discard:
            ranked = []
            for slot, card in cards:
                score, reasons = self.score(card, state, for_discard=True)
                ranked.append((score, slot, card, reasons))
            score, slot, card, reasons = min(ranked, key=lambda item: item[0])
            return CardDecision("drop", slot, card, score, tuple(reasons or ["наименее полезная карта"]))

        playable = [(slot, card) for slot, card in cards if self._affordable(card, state.me)]
        if playable:
            ranked = []
            for slot, card in playable:
                score, reasons = self.score(card, state)
                ranked.append((score, slot, card, reasons))
            score, slot, card, reasons = max(ranked, key=lambda item: item[0])
            return CardDecision("turn", slot, card, score, tuple(reasons or ["лучший доступный ход"]))

        ranked = []
        for slot, card in cards:
            score, reasons = self.score(card, state, for_discard=True)
            ranked.append((score, slot, card, reasons))
        score, slot, card, reasons = min(ranked, key=lambda item: item[0])
        return CardDecision("drop", slot, card, score, tuple(reasons or ["нет доступных карт"]))


class StrategicCardStrategy(CardStrategy):
    """Position-based strategy used by Worker 3.1.

    It evaluates the board produced by a card, protects against an affordable
    one-turn enemy kill, values income early, and retains strong near-future
    cards when selecting a discard.  No random choice participates in moves.
    """

    STRATEGY_VERSION = "3.5.1-discard-tiebreak"
    RESPONSE_WEIGHT = 1.0
    HIDDEN_HAND_SIZE = 6
    EXTRA_TURN_REPLY_PENALTY = 35.0
    PRODUCTION_FIELDS = ("mine", "monastery", "barracks")

    SELF_GAINS: dict[int, dict[str, int]] = {
        1: {"ore": 2, "mana": 2}, 2: {"wall": 1}, 3: {"mine": 1},
        5: {"wall": 4, "mine": 1}, 6: {"wall": 5, "mana": -6},
        8: {"wall": 3}, 9: {"wall": 4}, 13: {"monastery": 1},
        15: {"wall": 6}, 17: {"mine": 2}, 18: {"mine": -1, "wall": 10, "mana": 5},
        19: {"wall": 8}, 20: {"wall": 5, "barracks": 1},
        21: {"wall": 7, "mana": 7}, 22: {"wall": 6, "tower": 3},
        23: {"wall": 12}, 24: {"wall": 8, "tower": 5}, 25: {"wall": 15},
        26: {"wall": 6}, 27: {"wall": 20, "tower": 8},
        28: {"wall": 9, "army": -5}, 29: {"wall": 1, "tower": 1, "army": 2},
        32: {"wall": 7}, 34: {"tower": 1}, 36: {"tower": 3},
        37: {"monastery": 1}, 38: {"tower": 8}, 39: {"tower": 2},
        40: {"monastery": 1, "tower": 3}, 42: {"tower": 5},
        44: {"tower": -5, "monastery": 2},
        45: {"monastery": 1, "tower": 3, "wall": 3},
        47: {"tower": 8}, 48: {"tower": 5, "monastery": 1},
        50: {"tower": 5}, 51: {"tower": 11}, 53: {"tower": 6},
        54: {"tower": 7, "ore": -10}, 55: {"tower": 8, "wall": 3},
        56: {"tower": 8, "barracks": 1}, 57: {"tower": 15},
        58: {"tower": 10, "wall": 5, "army": 5},
        59: {"tower": 12}, 60: {"tower": 20}, 61: {"tower": 11, "wall": -6},
        63: {"mana": 3}, 64: {"tower": 4, "army": -3},
        66: {"tower": 13, "army": 6, "ore": 6}, 70: {"barracks": 1},
        75: {"wall": 3}, 77: {"barracks": 2}, 78: {"wall": 4, "tower": 2},
        79: {"army": 3},
        92: {"wall": 4}, 96: {"mana": 1}, 98: {"mana": -3},
    }

    @staticmethod
    def _change(player: PlayerState, **deltas: int) -> PlayerState:
        # This is the hottest deterministic primitive in the search.  Building
        # a recursive ``asdict`` for every tiny delta dominated allocation time.
        # Explicit immutable construction is bit-for-bit equivalent.
        return PlayerState(
            ore=max(0, player.ore + deltas.get("ore", 0)),
            mana=max(0, player.mana + deltas.get("mana", 0)),
            army=max(0, player.army + deltas.get("army", 0)),
            tower=max(0, player.tower + deltas.get("tower", 0)),
            wall=max(0, player.wall + deltas.get("wall", 0)),
            mine=max(0, player.mine + deltas.get("mine", 0)),
            monastery=max(0, player.monastery + deltas.get("monastery", 0)),
            barracks=max(0, player.barracks + deltas.get("barracks", 0)),
        )

    @staticmethod
    def _hurt(player: PlayerState, amount: int, direct: bool = False) -> PlayerState:
        if amount <= 0:
            return player
        if direct:
            return replace(player, tower=max(0, player.tower - amount))
        absorbed = min(player.wall, amount)
        return replace(player, wall=player.wall - absorbed, tower=max(0, player.tower - amount + absorbed))

    def simulate(self, card: CardDefinition, state: GameState) -> tuple[PlayerState, PlayerState]:
        original_me, original_enemy = state.me, state.opponent
        cache_key = (card.id, state.player_no, original_me, original_enemy)
        cached = self._simulate_cache.get(cache_key)
        if cached is not None:
            return cached
        me = self._change(original_me, ore=-card.ore, mana=-card.mana, army=-card.army)
        enemy = original_enemy
        cid = card.id
        if cid in self.SELF_GAINS:
            me = self._change(me, **self.SELF_GAINS[cid])
        if cid == 0:
            me, enemy = self._change(me, ore=-8), self._change(enemy, ore=-8)
        elif cid == 4:
            me = self._change(me, mine=2 if me.mine < enemy.mine else 1)
        elif cid == 7 and me.mine < enemy.mine:
            me = replace(me, mine=enemy.mine)
        elif cid == 10:
            me, enemy = self._change(me, mine=1, mana=4), self._change(enemy, mine=1)
        elif cid == 11:
            me = self._change(me, wall=6 if me.wall == 0 else 3)
        elif cid == 12:
            me, enemy = self._change(me, wall=-5), self._change(enemy, wall=-5)
        elif cid == 14:
            me, enemy = self._change(me, mine=-1), self._change(enemy, mine=-1)
        elif cid == 16:
            enemy = self._change(enemy, mine=-1)
        elif cid == 30:
            if me.wall < enemy.wall:
                me = self._hurt(self._change(me, barracks=-1), 2, True)
            elif enemy.wall < me.wall:
                enemy = self._hurt(self._change(enemy, barracks=-1), 2, True)
        elif cid == 31:
            me = self._change(me, army=6, wall=6, barracks=1 if me.barracks < enemy.barracks else 0)
        elif cid == 33:
            my_wall, enemy_wall = me.wall, enemy.wall
            me, enemy = replace(me, wall=enemy_wall), replace(enemy, wall=my_wall)
        elif cid == 40:
            enemy = self._change(enemy, tower=1)
        elif cid == 46:
            strongest = max(me.monastery, enemy.monastery)
            me, enemy = replace(me, monastery=strongest), replace(enemy, monastery=strongest)
        elif cid == 49:
            me = self._change(me, monastery=-1)
        elif cid == 50:
            enemy = self._change(enemy, ore=-6)
        elif cid == 52:
            me = self._hurt(self._change(me, monastery=-1), 7, True)
            enemy = self._change(enemy, monastery=-1)
        elif cid == 62:
            me = self._change(me, tower=2 if me.tower < enemy.tower else 1)
        elif cid == 63:
            me, enemy = self._change(me, tower=1), self._change(enemy, tower=1)
        elif cid == 67:
            me, enemy = self._change(me, army=-6), self._change(enemy, army=-6)
        elif cid == 69:
            me = self._change(me, mana=-3)
        elif cid == 71:
            me = self._hurt(me, 3)
        elif cid == 72:
            me = self._hurt(me, 1)
        elif cid == 79:
            me, enemy = self._change(me, barracks=1), self._change(enemy, barracks=1)
        elif cid == 82:
            enemy = self._change(enemy, army=-3)
        elif cid == 83:
            me = self._change(me, ore=-5, mana=-5, army=-5)
            enemy = self._change(enemy, ore=-5, mana=-5, army=-5)
        elif cid == 89:
            enemy = self._change(enemy, army=-8)
        elif cid == 90:
            enemy = self._change(enemy, mine=-1)
        elif cid == 91:
            stolen_ore = min(5, original_enemy.ore)
            stolen_mana = min(10, original_enemy.mana)
            me = self._change(
                me,
                ore=(stolen_ore + 1) // 2,
                mana=(stolen_mana + 1) // 2,
            )
            enemy = self._change(enemy, mana=-10, ore=-5)
        elif cid == 93:
            enemy = self._change(enemy, army=-5, barracks=-1)
        elif cid == 94:
            enemy = self._change(enemy, mana=-10, barracks=-1)
        elif cid == 97:
            me = self._hurt(me, 3, True)
        tower_damage, general_damage = self._damage(cid, state)
        enemy = self._hurt(enemy, tower_damage, True)
        enemy = self._hurt(enemy, general_damage)
        if cid == 65 and not (original_me.tower > original_enemy.wall):
            me = self._hurt(me, 8)
        result = (me, enemy)
        self._simulate_cache[cache_key] = result
        return result

    def _max_tower_loss(self, attacker: PlayerState, defender: PlayerState, state: GameState) -> int:
        maximum = 0
        for card in self.catalog.cards.values():
            if self._affordable(card, attacker):
                direct = self.DIRECT_TOWER_DAMAGE.get(card.id, 0)
                general = self.GENERAL_DAMAGE.get(card.id, 0)
                if card.id == 65:
                    direct, general = (8, 0) if attacker.tower > defender.wall else (0, 8)
                elif card.id == 84 and defender.wall == 0:
                    general = 10
                elif card.id == 86 and defender.wall > 10:
                    general = 10
                elif card.id == 87 and attacker.monastery > defender.monastery:
                    general = 12
                elif card.id == 88:
                    direct, general = (6, 0) if attacker.wall > defender.wall else (0, 6)
                elif card.id == 95 and attacker.wall > defender.wall:
                    general = 3
                maximum = max(maximum, direct + max(0, general - defender.wall))
        return maximum

    def _can_build_win(self, player: PlayerState, enemy: PlayerState, state: GameState) -> bool:
        if player.tower >= state.tower_goal:
            return True
        mirrored = replace(state, players={state.player_no: player, state.opponent_no: enemy})
        for card in self.catalog.cards.values():
            if not self._affordable(card, player):
                continue
            projected, _ = self.simulate(card, mirrored)
            if projected.tower >= state.tower_goal:
                return True
        return False

    def _next_turn_win_count(self, attacker: PlayerState, defender: PlayerState, state: GameState) -> int:
        """Count exact one-card wins after the attacker's next-turn income.

        This deliberately evaluates every catalog card because the opponent's
        hand is hidden.  Unlike the old maximum-damage approximation it also
        respects conditional damage, walls, construction and resource wins.
        """
        attacker = self._change(
            attacker,
            ore=attacker.mine,
            mana=attacker.monastery,
            army=attacker.barracks,
        )
        if (
            attacker.tower >= state.tower_goal
            or defender.tower <= 0
            or min(attacker.ore, attacker.mana, attacker.army) >= state.resource_goal
        ):
            return len(self.catalog.cards)
        mirrored = replace(
            state,
            players={state.player_no: attacker, state.opponent_no: defender},
        )
        wins = 0
        for card in self.catalog.cards.values():
            if not self._affordable(card, attacker):
                continue
            projected_attacker, projected_defender = self.simulate(card, mirrored)
            if (
                projected_attacker.tower >= state.tower_goal
                or projected_defender.tower <= 0
                or min(projected_attacker.ore, projected_attacker.mana, projected_attacker.army)
                >= state.resource_goal
            ):
                wins += 1
        return wins

    def _can_win_next_turn(self, attacker: PlayerState, defender: PlayerState, state: GameState) -> bool:
        return self._next_turn_win_count(attacker, defender, state) > 0

    def _static_utility(self, me: PlayerState, enemy: PlayerState, state: GameState) -> float:
        if enemy.tower <= 0 or me.tower >= state.tower_goal or min(me.ore, me.mana, me.army) >= state.resource_goal:
            return 100000.0
        if me.tower <= 0 or enemy.tower >= state.tower_goal or min(enemy.ore, enemy.mana, enemy.army) >= state.resource_goal:
            return -100000.0
        stage = max(.35, 1.0 - state.turn / 90.0)
        value = me.tower * 5 + me.wall * (3 if me.tower <= 12 else 1.7)
        value -= enemy.tower * 4.2 + enemy.wall * 1.15
        value += sum((me.mine, me.monastery, me.barracks)) * 13 * stage
        value -= sum((enemy.mine, enemy.monastery, enemy.barracks)) * 8 * stage
        value += sum((me.ore, me.mana, me.army)) * .42 - sum((enemy.ore, enemy.mana, enemy.army)) * .20
        value += min(me.ore, me.mana, me.army) * (1.8 if min(me.ore, me.mana, me.army) > state.resource_goal * .6 else .25)
        # Building victories were the only source of losses in the first
        # recorded batch.  Make the danger grow non-linearly as the enemy
        # approaches the tower goal so that tower damage and resource denial
        # beat passive wall building in the closing phase.
        enemy_tower_gap = state.tower_goal - enemy.tower
        if enemy_tower_gap <= 15:
            value -= (16 - max(0, enemy_tower_gap)) ** 2 * 4.0
            value -= enemy.mana * (1.1 if enemy_tower_gap <= 8 else .45)
        if me.tower <= 10:
            value -= (11 - me.tower) ** 2 * 5
        return value

    def _utility(self, me: PlayerState, enemy: PlayerState, state: GameState) -> float:
        value = self._static_utility(me, enemy, state)
        if abs(value) >= 100000:
            return value
        winning_replies = self._next_turn_win_count(enemy, me, state)
        if winning_replies:
            # Approximate the chance that at least one of six hidden cards is
            # among the currently winning replies.  This rewards reducing the
            # opponent from two lethal cards to one instead of treating both
            # positions as the same binary danger.
            reply_share = min(1.0, winning_replies / max(1, len(self.catalog.cards)))
            win_probability = 1.0 - (1.0 - reply_share) ** 6
            value -= 3000 * win_probability
        return value

    @staticmethod
    def _income(player: PlayerState) -> PlayerState:
        return replace(
            player,
            ore=player.ore + player.mine,
            mana=player.mana + player.monastery,
            army=player.army + player.barracks,
        )

    @staticmethod
    def _turns_until_affordable(card: CardDefinition, player: PlayerState) -> int | None:
        waits: list[int] = []
        for cost, resource, production in (
            (card.ore, player.ore, player.mine),
            (card.mana, player.mana, player.monastery),
            (card.army, player.army, player.barracks),
        ):
            shortage = max(0, cost - resource)
            if not shortage:
                waits.append(0)
            elif production <= 0:
                return None
            else:
                waits.append(math.ceil(shortage / production))
        return max(waits, default=0)

    def _discard_retention_value(self, card: CardDefinition, state: GameState) -> float:
        """Long-horizon value lost when a production card is discarded.

        The two-ply board search cannot see which card vanished from our hand:
        every discard otherwise produces the same board.  Keep this opportunity
        cost outside the blended reply score so a valuable mine/monastery/
        barracks card is not made almost indistinguishable from filler.
        """
        after_me, after_enemy = self.simulate(card, state)
        gains = {
            field: max(0, getattr(after_me, field) - getattr(state.me, field))
            for field in self.PRODUCTION_FIELDS
        }
        if not any(gains.values()):
            return 0.0

        wait = self._turns_until_affordable(card, state.me)
        # A card that cannot be reached with the current economy still has some
        # option value, but it must not lock the hand forever.
        if wait is None:
            wait_penalty = 38.0
        else:
            wait_penalty = wait * 2.5

        value = 0.0
        for field, gain in gains.items():
            if not gain:
                continue
            ours = getattr(state.me, field)
            theirs = getattr(state.opponent, field)
            catch_up = min(4, max(0, theirs - ours))
            low_economy = max(0, 3 - ours)
            zero_bonus = 12 if ours == 0 else 0
            value += gain * (24 + catch_up * 7 + low_economy * 6 + zero_bonus)

        enemy_gain = sum(
            max(0, getattr(after_enemy, field) - getattr(state.opponent, field))
            for field in self.PRODUCTION_FIELDS
        )
        value -= enemy_gain * 12
        value -= wait_penalty

        self_tower_loss = max(0, state.me.tower - after_me.tower)
        value -= self_tower_loss * (7 if state.me.tower <= 12 else 4)
        if after_me.tower <= 3 and self_tower_loss:
            value *= .15
        return max(0.0, value)

    def _after_decision(self, decision: CardDecision, state: GameState) -> tuple[PlayerState, PlayerState]:
        if decision.action == "turn":
            return self.simulate(decision.card, state)
        return state.me, state.opponent

    def _modeled_reply_value(self, decision: CardDecision, state: GameState) -> float:
        """Expected position after a rational reply from six hidden cards."""
        me, enemy = self._after_decision(decision, state)
        immediate = self._static_utility(me, enemy, state)
        if abs(immediate) >= 100000:
            return immediate
        if decision.action == "turn" and decision.card.id in self.EXTRA_TURN_CARDS:
            return immediate + 40.0

        enemy = self._income(enemy)
        discard_value = self._static_utility(self._income(me), enemy, state)
        mirrored = replace(
            state,
            player_no=state.opponent_no,
            players={state.opponent_no: enemy, state.player_no: me},
            hand=[],
        )
        reply_values: list[float] = []
        for card in self.catalog.cards.values():
            result = discard_value
            if self._affordable(card, enemy):
                reply_enemy, reply_me = self.simulate(card, mirrored)
                result = min(result, self._static_utility(self._income(reply_me), reply_enemy, state))
                if card.id in self.EXTRA_TURN_CARDS:
                    result -= self.EXTRA_TURN_REPLY_PENALTY
            reply_values.append(result)

        reply_values.sort()
        total = len(reply_values)
        expected = 0.0
        # Approximate the best response available in a random six-card hidden
        # hand using the order statistic of six independent catalog draws.
        for index, result in enumerate(reply_values):
            probability = ((total - index) / total) ** self.HIDDEN_HAND_SIZE
            probability -= ((total - index - 1) / total) ** self.HIDDEN_HAND_SIZE
            expected += probability * result
        return expected

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.STRATEGY_VERSION,
            "lookahead_plies": 2,
            "response_weight": self.RESPONSE_WEIGHT,
            "hidden_hand_size": self.HIDDEN_HAND_SIZE,
            "catalog_cards": len(self.catalog.cards),
            "production_retention": True,
        }

    def score(self, card: CardDefinition, state: GameState, *, for_discard: bool = False) -> tuple[float, list[str]]:
        after_me, after_enemy = self.simulate(card, state)
        gain = self._utility(after_me, after_enemy, state) - self._utility(state.me, state.opponent, state)
        reasons: list[str] = []
        threats_before = self._next_turn_win_count(state.opponent, state.me, state)
        threats_after = self._next_turn_win_count(after_enemy, after_me, state)
        if threats_before and threats_after == 0:
            reasons.append("предотвращает победу соперника следующим ходом")
        elif threats_after < threats_before:
            reasons.append(f"снижает число победных ответов соперника: {threats_before}→{threats_after}")
        if after_enemy.tower <= 0:
            reasons.append("победный удар")
        if after_me.tower >= state.tower_goal:
            reasons.append("победа строительством")
        if after_me.tower > state.me.tower or after_me.wall > state.me.wall:
            reasons.append(f"защита {after_me.tower-state.me.tower:+}/{after_me.wall-state.me.wall:+}")
        if after_enemy.tower < state.opponent.tower or after_enemy.wall < state.opponent.wall:
            reasons.append("наносит эффективный урон")
        if (after_me.mine, after_me.monastery, after_me.barracks) != (state.me.mine, state.me.monastery, state.me.barracks):
            reasons.append("усиливает экономику")
        if card.id in self.EXTRA_TURN_CARDS:
            gain += 16
            reasons.append("дополнительный ход")
        if card.id in {100, 101}:
            gain += 9
            reasons.append("обновляет руку")
        if for_discard:
            wait = max(
                math.ceil(max(0, card.ore-state.me.ore)/max(1, state.me.mine)),
                math.ceil(max(0, card.mana-state.me.mana)/max(1, state.me.monastery)),
                math.ceil(max(0, card.army-state.me.army)/max(1, state.me.barracks)),
            )
            return -max(-20, min(100, gain - wait*3)), ("сброс: минимальная будущая ценность", f"ожидание {wait} ход.")
        return gain, reasons or ["лучшее изменение позиции"]

    def rank_choices(self, state: GameState) -> list[CardDecision]:
        immediate_choices: list[CardDecision] = []
        for slot, card_id in enumerate(state.hand):
            if card_id not in self.catalog.cards:
                continue
            card = self.catalog[card_id]
            discard_score, discard_reasons = self.score(card, state, for_discard=True)
            immediate_choices.append(CardDecision("drop", slot, card, discard_score, tuple(discard_reasons)))
            if not state.must_discard and self._affordable(card, state.me):
                score, reasons = self.score(card, state)
                immediate_choices.append(CardDecision("turn", slot, card, score, tuple(reasons)))

        result: list[CardDecision] = []
        threats_before = self._next_turn_win_count(state.opponent, state.me, state)
        for choice in immediate_choices:
            after_me, after_enemy = self._after_decision(choice, state)
            response_value = self._modeled_reply_value(choice, state)
            final_score = (1.0 - self.RESPONSE_WEIGHT) * choice.score
            final_score += self.RESPONSE_WEIGHT * response_value
            discard_retention = 0.0
            if choice.action == "drop":
                discard_retention = self._discard_retention_value(choice.card, state)
                final_score -= discard_retention
            threats_after = self._next_turn_win_count(after_enemy, after_me, state)
            reasons = tuple(choice.reasons)
            if discard_retention:
                reasons += (f"ценность сохранения производства {discard_retention:.1f}",)
            reasons += (f"прогноз после ответа {response_value:.1f}",)
            result.append(
                CardDecision(
                    choice.action,
                    choice.slot,
                    choice.card,
                    final_score,
                    reasons,
                    immediate_score=choice.score,
                    response_value=response_value,
                    predicted_me=asdict(after_me),
                    predicted_opponent=asdict(after_enemy),
                    winning_replies_before=threats_before,
                    winning_replies_after=threats_after,
                    discard_retention=discard_retention,
                )
            )
        # A discard does not change the visible board, therefore every ordinary
        # discard receives the same two-ply response value.  In 3.5.0 Python's
        # stable sort then selected whichever tied card happened to be first in
        # the hand (131 suspicious choices in 155 recorded discards).  Keep the
        # strategic score primary, prefer playing on an exact play/drop tie,
        # and use the already calculated future card value to choose *which*
        # tied card is least painful to discard.
        return sorted(
            result,
            key=lambda item: (
                item.score,
                1 if item.action == "turn" else 0,
                item.immediate_score if item.action == "drop" and item.immediate_score is not None else 0.0,
            ),
            reverse=True,
        )

    def choose(self, state: GameState) -> CardDecision:
        choices = self.rank_choices(state)
        if not choices:
            raise ValueError("В руке нет распознанных карт")
        return choices[0]


class OpponentBelief:
    """Particle belief for the empirically observed cyclic shared deck.

    The 2026-08 dataset proves that PLAY and DISCARD cards return to the common
    draw cycle.  A card is impossible during its observed cooldown, then its
    draw weight rises gradually instead of jumping from zero to fully unseen.
    Server ``turn`` is used as the card-action clock, so extra turns and missing
    local polling rows cannot shorten the cooldown.
    """

    DRAW_COOLDOWN_ACTIONS = 45
    # Empirical CDF from 1,673 exact redraws in 767 games.  This is deliberately
    # stored as data points, not an invented shuffle formula; new archives can
    # replace the table without changing the particle engine.
    RETURN_PROBABILITY_KNOTS = (
        (44, 0.00),
        (45, 0.06),
        (46, 0.12),
        (48, 0.24),
        (50, 0.34),
        (55, 0.53),
        (60, 0.67),
        (65, 0.77),
        (70, 0.84),
        (78, 0.90),
        (100, 0.97),
        (139, 1.00),
    )

    def __init__(
        self,
        catalog: CardCatalog,
        particle_count: int = 256,
        *,
        state_path: Path | None = None,
        history_root: Path | None = None,
    ) -> None:
        self.catalog = catalog
        self.particle_count = max(64, particle_count)
        self.state_path = state_path
        self.history_root = history_root
        self.game_id: int | None = None
        self.observed: Counter[int] = Counter()
        self.opponent_observed: Counter[int] = Counter()
        self.last_seen: dict[int, dict[str, Any]] = {}
        self.seen_keys: set[tuple[int, str]] = set()
        self.unknown_action_indices: set[int] = set()
        self.current_action = 0
        self.reconstructed_from_segments = False
        self.last_resync: dict[str, Any] = {}
        self.hand_particles: list[tuple[int, ...]] = []
        self.particle_signature: str = ""
        self.resample_count = 0
        self.rejuvenation_count = 0
        self.last_conditioning_survivors = 0
        self.last_pre_resample_ess = 0.0

    def reset(self, game_id: int | None = None) -> None:
        if game_id is not None and game_id == self.game_id:
            return
        self.game_id = game_id
        self.observed.clear()
        self.opponent_observed.clear()
        self.last_seen.clear()
        self.seen_keys.clear()
        self.unknown_action_indices.clear()
        self.current_action = 0
        self.reconstructed_from_segments = False
        self.hand_particles.clear()
        self.particle_signature = ""
        self.resample_count = 0
        self.rejuvenation_count = 0
        self.last_conditioning_survivors = 0
        self.last_pre_resample_ess = 0.0
        loaded = game_id is not None and self._load_persisted(game_id)
        if game_id is not None:
            self._reconstruct_segments(game_id)
        if loaded or self.reconstructed_from_segments:
            self._persist()
            return
        self._persist()

    def _load_persisted(self, game_id: int) -> bool:
        if self.state_path is None:
            return False
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("game_id") or 0) != game_id:
                return False
            self.current_action = int(payload.get("current_action") or 0)
            self.last_seen = {
                int(card_id): {
                    "action": int(item["action"]),
                    "owner": str(item.get("owner") or "unknown"),
                    "event": str(item.get("event") or "turn"),
                }
                for card_id, item in (payload.get("last_seen") or {}).items()
                if int(card_id) in self.catalog.cards
            }
            self.seen_keys = {
                (int(item[0]), str(item[1]))
                for item in payload.get("seen_keys") or []
                if isinstance(item, list) and len(item) == 2
            }
            self.unknown_action_indices = {
                int(action_index)
                for action_index in payload.get("unknown_action_indices") or []
                if int(action_index) > 0
            }
            self.observed.update({int(key): int(value) for key, value in (payload.get("observed") or {}).items()})
            self.opponent_observed.update(
                {int(key): int(value) for key, value in (payload.get("opponent_observed") or {}).items()}
            )
            self.hand_particles = [
                tuple(int(card_id) for card_id in hand if int(card_id) in self.catalog.cards)
                for hand in payload.get("hand_particles") or []
                if isinstance(hand, list)
            ]
            self.particle_signature = str(payload.get("particle_signature") or "")
            self.resample_count = int(payload.get("resample_count") or 0)
            self.rejuvenation_count = int(payload.get("rejuvenation_count") or 0)
            self.last_conditioning_survivors = int(payload.get("last_conditioning_survivors") or 0)
            self.last_pre_resample_ess = float(payload.get("last_pre_resample_ess") or 0.0)
            return True
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False

    def _persist(self) -> None:
        if self.state_path is None or self.game_id is None:
            return
        payload = {
            "version": 3,
            "game_id": self.game_id,
            "current_action": self.current_action,
            "cooldown_actions": self.DRAW_COOLDOWN_ACTIONS,
            "last_seen": {str(key): value for key, value in self.last_seen.items()},
            "seen_keys": [[turn, move] for turn, move in sorted(self.seen_keys)],
            "unknown_action_indices": sorted(self.unknown_action_indices),
            "observed": {str(key): value for key, value in self.observed.items()},
            "opponent_observed": {str(key): value for key, value in self.opponent_observed.items()},
            "hand_particles": [list(hand) for hand in self.hand_particles[:1200]],
            "particle_signature": self.particle_signature,
            "resample_count": self.resample_count,
            "rejuvenation_count": self.rejuvenation_count,
            "last_conditioning_survivors": self.last_conditioning_survivors,
            "last_pre_resample_ess": self.last_pre_resample_ess,
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.state_path)
        except OSError:
            pass

    def _record(self, action_index: int, move: str, card_id: int, owner: str, event: str) -> None:
        key = (action_index, move)
        if key in self.seen_keys or card_id not in self.catalog.cards:
            return
        self.seen_keys.add(key)
        self.unknown_action_indices.discard(action_index)
        self.current_action = max(self.current_action, action_index)
        self.observed[card_id] += 1
        if owner == "opponent":
            self.opponent_observed[card_id] += 1
        self.last_seen[card_id] = {
            "action": action_index,
            "owner": owner,
            "event": event,
        }

    def _reconstruct_segments(self, game_id: int) -> None:
        if self.history_root is None or not self.history_root.exists():
            return
        events: dict[tuple[int, str], dict[str, Any]] = {}
        for path in self.history_root.glob(f"*_game_{game_id}.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            for event in payload.get("events") or []:
                before = event.get("before") or {}
                move = str(event.get("server_move") or "")
                turn = int(event.get("turn") or 0)
                if not move or before.get("last_move") == move:
                    continue
                events[(turn, move)] = event
        for (turn, move), event in sorted(events.items()):
            parsed = parse_last_move(move)
            if parsed:
                action, card_id, _slot = parsed
                self._record(turn, move, card_id, str(event.get("actor") or "unknown"), action)
        self.reconstructed_from_segments = bool(events)

    def synchronize_state(self, state: GameState) -> dict[str, Any]:
        """Advance belief across a reconnect/polling gap.

        HeroesWM exposes only the latest public move in the compact state.  Any
        skipped action indices therefore cannot be replayed exactly.  They are
        persisted as unknown transitions and sampled independently inside every
        particle, rather than silently treating the old belief as current.
        """
        if self.game_id != state.game_id:
            self.reset(state.game_id)
        previous_action = self.current_action
        if state.turn <= previous_action:
            self.last_resync = {
                "from_action": previous_action,
                "to_action": state.turn,
                "missing_actions": 0,
                "known_tail_move": False,
            }
            return self.last_resync

        parsed = parse_last_move(state.last_move)
        if previous_action == 0 and state.turn <= 1 and not parsed:
            self.current_action = state.turn
            self.last_resync = {
                "from_action": 0,
                "to_action": state.turn,
                "missing_actions": 0,
                "known_tail_move": False,
            }
            self._persist()
            return self.last_resync
        known_tail = bool(parsed and (state.turn, state.last_move) not in self.seen_keys)
        missing_end = state.turn - 1 if known_tail else state.turn
        if missing_end >= previous_action + 1:
            self.unknown_action_indices.update(range(previous_action + 1, missing_end + 1))
        if known_tail and parsed:
            action, card_id, _slot = parsed
            self._record(state.turn, state.last_move, card_id, "unknown", action)
        self.current_action = max(self.current_action, state.turn)
        self.last_resync = {
            "from_action": previous_action,
            "to_action": state.turn,
            "missing_actions": max(0, missing_end - previous_action),
            "known_tail_move": known_tail,
            "unknown_action_indices": sorted(
                index for index in self.unknown_action_indices if index <= state.turn
            ),
        }
        self._persist()
        return self.last_resync

    def observe_transition(self, before: GameState, after: GameState) -> None:
        self.synchronize_state(before)
        parsed = parse_last_move(after.last_move)
        key = (after.turn, after.last_move)
        if after.turn > self.current_action + 1:
            missing_end = after.turn - 1 if parsed else after.turn
            self.unknown_action_indices.update(range(self.current_action + 1, missing_end + 1))
        if not parsed or key in self.seen_keys:
            self.current_action = max(self.current_action, after.turn)
            self._reconcile_persistent_hands(after)
            self._persist()
            return
        action, card_id, _slot = parsed
        owner = "us" if before.is_your_turn else "opponent"
        self._record(after.turn, after.last_move, card_id, owner, action)
        if owner == "opponent":
            self._condition_persistent_hands(before, after, card_id, after.turn)
        else:
            self._reconcile_persistent_hands(after)
            self.particle_signature = self._particle_state_signature(after)
        self._persist()

    def card_age(self, card_id: int, state: GameState) -> int | None:
        item = self.last_seen.get(card_id)
        return None if item is None else max(0, state.turn - int(item["action"]))

    def return_weight(self, card_id: int, state: GameState) -> float:
        age = self.card_age(card_id, state)
        if age is None:
            return 1.0
        if age < self.DRAW_COOLDOWN_ACTIONS:
            return 0.0
        for (left_age, left_probability), (right_age, right_probability) in zip(
            self.RETURN_PROBABILITY_KNOTS,
            self.RETURN_PROBABILITY_KNOTS[1:],
        ):
            if age <= right_age:
                fraction = (age - left_age) / max(1, right_age - left_age)
                return left_probability + fraction * (right_probability - left_probability)
        return 1.0

    @classmethod
    def _return_weight_at(cls, last_action: int | None, action_index: int) -> float:
        if last_action is None:
            return 1.0
        age = max(0, action_index - last_action)
        if age < cls.DRAW_COOLDOWN_ACTIONS:
            return 0.0
        for (left_age, left_probability), (right_age, right_probability) in zip(
            cls.RETURN_PROBABILITY_KNOTS,
            cls.RETURN_PROBABILITY_KNOTS[1:],
        ):
            if age <= right_age:
                fraction = (age - left_age) / max(1, right_age - left_age)
                return left_probability + fraction * (right_probability - left_probability)
        return 1.0

    def weighted_pool(self, state: GameState) -> list[tuple[int, float]]:
        ours = set(state.hand)
        return [
            (card_id, weight)
            for card_id in self.catalog.cards
            if card_id not in ours and (weight := self.return_weight(card_id, state)) > 0.0
        ]

    def unseen_pool(self, state: GameState) -> list[int]:
        return [card_id for card_id, _weight in self.weighted_pool(state)]

    def seed_text(self, state: GameState) -> str:
        return repr(
            (
                state.game_id,
                state.turn,
                tuple(state.hand),
                tuple(sorted((card_id, item["action"], item["owner"], item["event"]) for card_id, item in self.last_seen.items())),
                tuple(sorted(index for index in self.unknown_action_indices if index <= state.turn)),
                state.opponent.ore,
                state.opponent.mana,
                state.opponent.army,
            )
        )

    @staticmethod
    def _weighted_sample(rng: random.Random, pool: list[tuple[int, float]], count: int) -> tuple[int, ...]:
        if count >= len(pool):
            return tuple(card_id for card_id, _weight in pool)
        ranked = []
        for card_id, weight in pool:
            key = rng.random() ** (1.0 / max(1e-9, weight))
            ranked.append((key, card_id))
        ranked.sort(reverse=True)
        return tuple(card_id for _key, card_id in ranked[:count])

    @staticmethod
    def _weighted_choice(rng: random.Random, pool: list[tuple[int, float]]) -> int | None:
        total = sum(max(0.0, weight) for _card_id, weight in pool)
        if total <= 0.0:
            return None
        needle = rng.random() * total
        for card_id, weight in pool:
            needle -= max(0.0, weight)
            if needle <= 0.0:
                return card_id
        return pool[-1][0]

    def _particle_after_unknown_transitions(
        self,
        state: GameState,
        rng: random.Random,
    ) -> tuple[int, ...]:
        ours = set(state.hand)
        relevant_unknown = sorted(index for index in self.unknown_action_indices if index <= state.turn)
        if not relevant_unknown:
            pool = self.weighted_pool(state)
            return self._weighted_sample(rng, pool, min(6, len(pool))) if pool else tuple()

        first_unknown = relevant_unknown[0]
        known_actions: dict[int, int] = {}
        for action_index, move in self.seen_keys:
            parsed = parse_last_move(move)
            if parsed and action_index <= state.turn:
                known_actions[action_index] = parsed[1]
        # Start immediately before the first missing transition.  Do not leak a
        # later known observation backwards into the latent-gap simulation.
        simulated_last_seen: dict[int, int] = {}
        for action_index, card_id in sorted(known_actions.items()):
            if action_index >= first_unknown:
                break
            simulated_last_seen[card_id] = action_index
        # Each missing public action is a latent PLAY/DISCARD.  Sample a
        # compatible card and put it back onto the same empirical cooldown
        # curve.  This does not pretend to know the server's lower-half order.
        unknown_set = set(relevant_unknown)
        for action_index in sorted(unknown_set | {index for index in known_actions if index >= first_unknown}):
            if action_index in known_actions:
                simulated_last_seen[known_actions[action_index]] = action_index
                continue
            transition_pool = [
                (card_id, weight)
                for card_id in self.catalog.cards
                if not (
                    card_id in ours
                    and state.turn - action_index < self.DRAW_COOLDOWN_ACTIONS
                )
                and (weight := self._return_weight_at(simulated_last_seen.get(card_id), action_index)) > 0.0
            ]
            moved_card = self._weighted_choice(rng, transition_pool)
            if moved_card is not None:
                simulated_last_seen[moved_card] = action_index

        final_pool = [
            (card_id, weight)
            for card_id in self.catalog.cards
            if card_id not in ours
            and (weight := self._return_weight_at(simulated_last_seen.get(card_id), state.turn)) > 0.0
        ]
        if not final_pool:
            return tuple()
        return self._weighted_sample(rng, final_pool, min(6, len(final_pool)))

    def _particle_state_signature(self, state: GameState) -> str:
        return hashlib.sha256(
            repr(
                (
                    state.game_id,
                    state.turn,
                    tuple(state.hand),
                    tuple(sorted((card_id, item["action"]) for card_id, item in self.last_seen.items())),
                    tuple(sorted(index for index in self.unknown_action_indices if index <= state.turn)),
                )
            ).encode("utf-8")
        ).hexdigest()[:20]

    def _reconcile_persistent_hands(self, state: GameState) -> None:
        if not self.hand_particles:
            return
        ours = set(state.hand)
        rng = random.Random(self.seed_text(state) + ":persistent-reconcile")
        reconciled: list[tuple[int, ...]] = []
        for hand in self.hand_particles:
            kept = tuple(card_id for card_id in hand if card_id not in ours)
            pool = [item for item in self.weighted_pool(state) if item[0] not in kept]
            fill = self._weighted_sample(rng, pool, min(max(0, 6 - len(kept)), len(pool))) if pool else tuple()
            reconciled.append((kept + fill)[:6])
        self.hand_particles = reconciled

    def _condition_persistent_hands(
        self,
        before: GameState,
        after: GameState,
        card_id: int,
        action_index: int,
    ) -> None:
        if not self.hand_particles:
            return
        compatible = [hand for hand in self.hand_particles if card_id in hand]
        self.last_conditioning_survivors = len(compatible)
        self.last_pre_resample_ess = float(len(compatible))
        if not compatible:
            # All prior particles receive zero posterior weight.  A deterministic
            # resample from the updated cyclic belief is safer than retaining an
            # impossible hidden hand.
            self.hand_particles.clear()
            self.particle_signature = ""
            self.resample_count += 1
            return
        rng = random.Random(self.seed_text(after) + f":condition:{action_index}:{card_id}")
        conditioned: list[tuple[int, ...]] = []
        target_count = len(self.hand_particles)
        if len(compatible) < target_count:
            self.resample_count += 1
        # Systematic cycling preserves every surviving ancestry before any is
        # duplicated.  Random-with-replacement needlessly collapses diversity.
        compatible = list(compatible)
        rng.shuffle(compatible)
        for _index in range(target_count):
            source = list(compatible[_index % len(compatible)])
            source.remove(card_id)
            pool = [
                (candidate, weight)
                for candidate, weight in self.weighted_pool(after)
                if candidate not in set(after.hand) | set(source)
            ]
            replacement = self._weighted_choice(rng, pool)
            if replacement is not None:
                source.append(replacement)
            conditioned.append(tuple(source[:6]))
        unique_before = len(set(conditioned))
        if unique_before < min(64, max(8, target_count // 4)):
            # Rejuvenate only the newly drawn sixth card.  The five persistent
            # hidden cards remain intact, so observed-hand constraints are not
            # violated merely to manufacture diversity.
            rejuvenated: list[tuple[int, ...]] = []
            for index, hand in enumerate(conditioned):
                persistent = tuple(hand[:5])
                pool = [
                    (candidate, weight)
                    for candidate, weight in self.weighted_pool(after)
                    if candidate not in set(after.hand) | set(persistent)
                ]
                local_rng = random.Random(self.seed_text(after) + f":rejuvenate:{action_index}:{index}")
                replacement = self._weighted_choice(local_rng, pool)
                rejuvenated.append(persistent + ((replacement,) if replacement is not None else tuple()))
            if len(set(rejuvenated)) > unique_before:
                conditioned = rejuvenated
                self.rejuvenation_count += 1
        self.hand_particles = conditioned
        self._reconcile_persistent_hands(after)
        self.particle_signature = self._particle_state_signature(after)

    def particles(
        self,
        state: GameState,
        count: int | None = None,
        *,
        offset: int = 0,
    ) -> list[tuple[int, ...]]:
        requested = self.particle_count if count is None else max(1, count)
        needed = offset + requested
        signature = self._particle_state_signature(state)
        if self.particle_signature != signature:
            # Known sequential transitions update the persistent particles in
            # observe_transition().  A reconnect gap contains latent actions,
            # so resample that unknown transition chain exactly once.
            if self.unknown_action_indices:
                self.hand_particles.clear()
            else:
                self._reconcile_persistent_hands(state)
            self.particle_signature = signature
        pool = self.weighted_pool(state)
        if not pool and not self.unknown_action_indices:
            return [tuple()]
        if len(self.hand_particles) < needed:
            rng = random.Random(self.seed_text(state) + ":persistent-generate")
            target = needed
            if self.unknown_action_indices:
                generated = [self._particle_after_unknown_transitions(state, rng) for _ in range(target)]
            else:
                hand_size = min(6, len(pool))
                generated = [self._weighted_sample(rng, pool, hand_size) for _ in range(target)]
            # Regeneration is deterministic for a fixed public state, so an
            # adaptive 200->400 request is the same prefix as a direct 400 run.
            self.hand_particles = generated
        return self.hand_particles[offset:needed]

    @staticmethod
    def probabilities(
        particles: list[tuple[int, ...]],
        structurally_certain: set[int] | None = None,
    ) -> dict[int, float]:
        if not particles:
            return {}
        counts: Counter[int] = Counter()
        for hand in particles:
            counts.update(set(hand))
        total = len(particles)
        certain = structurally_certain or set()
        # Jeffreys smoothing prevents a finite/degenerate particle sample from
        # asserting false certainty.  Exact 100% is reserved for constraints
        # that prove inclusion independently of Monte Carlo ancestry.
        return {
            card_id: 1.0 if card_id in certain else (count + 0.5) / (total + 1.0)
            for card_id, count in counts.items()
        }

    def diagnostics(self, state: GameState, particles: list[tuple[int, ...]]) -> dict[str, Any]:
        total = len(particles)
        hand_counts = Counter(particles)
        frequencies = [count / total for count in hand_counts.values()] if total else []
        effective_sample_size = 1.0 / sum(value * value for value in frequencies) if frequencies else 0.0
        hand_entropy = -sum(value * math.log(value) for value in frequencies if value > 0)
        pool = self.weighted_pool(state)
        weight_total = sum(weight for _card_id, weight in pool)
        deck_probabilities = [weight / weight_total for _card_id, weight in pool] if weight_total else []
        deck_entropy = -sum(value * math.log(value) for value in deck_probabilities if value > 0)
        return {
            "effective_sample_size": effective_sample_size,
            "unique_particle_count": len(hand_counts),
            "unique_opponent_hands": len(hand_counts),
            "opponent_hand_entropy": hand_entropy,
            "deck_entropy": deck_entropy,
            "max_particle_weight": max(frequencies, default=0.0),
            "resample_count": self.resample_count,
            "rejuvenation_count": self.rejuvenation_count,
            "last_conditioning_survivors": self.last_conditioning_survivors,
            "last_pre_resample_ess": self.last_pre_resample_ess,
            "structurally_certain_cards": [card_id for card_id, _weight in pool] if 0 < len(pool) <= 6 else [],
        }


class ProbabilisticCardStrategy(StrategicCardStrategy):
    """Belief-state strategy with calibrated probabilities and tactical search."""

    STRATEGY_VERSION = "3.9.1-pwin-objective-candidate"
    OOS_SERIES_ID = "3.9.1-clean-oos-pending-2026-08-20"
    PARTICLE_COUNT = 200
    MIN_PARTICLES = 200
    PARTICLE_BATCH = 200
    # The upper bound is shared by every server mode, while the effective
    # per-decision cap is selected from PARTICLE_LIMIT_BY_TIMEOUT.  Keeping a
    # single high hard cap lets the 30/40-second rooms use their spare time
    # without weakening the deadline guard used by the 15-second room.
    MAX_PARTICLES = 6000
    SERVER_MOVE_TIMEOUTS = (15, 30, 40)
    PARTICLE_LIMIT_BY_TIMEOUT = {15: 1600, 30: 4000, 40: 6000}
    MOVE_SAFETY_MARGIN = 4.0
    # Selected on the freeze-validation audit, never on a single battle.
    # 0.05 percentage point is deliberately narrow: it only authorizes a
    # risk/tactical tie-break when the entire paired 95% CI fits inside it.
    PRACTICAL_EQUIVALENCE_EPSILON = 0.0005
    MAX_ANALYSIS_SECONDS = 36.0
    DEADLINE_GUARD_SECONDS = 0.20
    TAIL_FRACTION = 0.10
    NORMAL_RISK_LAMBDA = 0.22
    CRITICAL_RISK_LAMBDA = 0.48
    EXTRA_TURN_DEPTH = 3

    def __init__(
        self,
        catalog: CardCatalog,
        learner: CardGameLearner | None = None,
        *,
        belief_state_path: Path | None = None,
        history_root: Path | None = None,
    ) -> None:
        super().__init__(catalog, learner)
        self.belief = OpponentBelief(
            catalog,
            self.PARTICLE_COUNT,
            state_path=belief_state_path,
            history_root=history_root,
        )
        self.last_sampling: dict[str, Any] = {}
        self._replacement_category_cache: dict[tuple[Any, ...], str] = {}
        self._win_next_cache: dict[
            tuple[PlayerState, PlayerState, tuple[int, ...], int, int, tuple[int, ...]], float
        ] = {}
        self._finisher_pool_cache: dict[
            tuple[PlayerState, PlayerState, int, int, tuple[int, ...]], float
        ] = {}
        self._particle_next_win_cache: dict[
            tuple[PlayerState, PlayerState, tuple[int, ...], int, int], bool
        ] = {}
        self._decision_reply_cache: dict[
            tuple[Any, ...], tuple[float, bool, PlayerState, PlayerState, float]
        ] = {}
        self._decision_extra_reply_cache: dict[
            tuple[Any, ...], dict[int, tuple[float, bool, PlayerState, PlayerState, float]]
        ] = {}
        self._decision_next_win_cache: dict[tuple[Any, ...], bool] = {}
        self._decision_quantile_cache: dict[tuple[str, int, tuple[Any, ...]], float] = {}
        self._decision_state_pwin_cache: dict[tuple[Any, ...], float] = {}
        self._decision_hand_option_cache: dict[tuple[Any, ...], float] = {}
        self.policy_runtime = PolicyRuntime.load()
        self.first_actor = "unknown"
        self.configured_move_timeout: int | None = None

    def configure_move_timeout(self, seconds: int | None) -> None:
        """Bind Monte Carlo limits to the room's selected server timer."""
        parsed = int(seconds or 0)
        self.configured_move_timeout = parsed if parsed in self.SERVER_MOVE_TIMEOUTS else None

    def _analysis_limits(self, state: GameState) -> tuple[float, int, int]:
        """Return wall-clock budget, particle cap and effective server mode.

        The protocol occasionally reports 119/121 seconds on the first move,
        even for a 30-second room.  The configured room timeout therefore caps
        that value.  Conversely, a genuinely depleted server timer always
        shortens the budget, so reconnects and slow polling remain safe.
        """
        if self.configured_move_timeout in self.SERVER_MOVE_TIMEOUTS:
            move_timeout = int(self.configured_move_timeout)
        else:
            remaining = max(0, int(state.time_left))
            if remaining <= 17:
                move_timeout = 15
            elif remaining <= 32:
                move_timeout = 30
            else:
                move_timeout = 40

        nominal_budget = max(0.08, float(move_timeout) - self.MOVE_SAFETY_MARGIN)
        actual_budget = max(0.08, float(state.time_left) - self.MOVE_SAFETY_MARGIN)
        time_budget = min(self.MAX_ANALYSIS_SECONDS, nominal_budget, actual_budget)

        nominal_particle_limit = int(self.PARTICLE_LIMIT_BY_TIMEOUT[move_timeout])
        scaled_limit = int(
            math.ceil(
                nominal_particle_limit * time_budget / nominal_budget / self.PARTICLE_BATCH
            )
            * self.PARTICLE_BATCH
        )
        particle_limit = max(
            self.MIN_PARTICLES,
            min(self.MAX_PARTICLES, nominal_particle_limit, scaled_limit),
        )
        return time_budget, particle_limit, move_timeout

    def reset_game(self, game_id: int | None = None) -> None:
        self.belief.reset(game_id)
        self.first_actor = "unknown"

    def observe_transition(self, before: GameState, after: GameState) -> None:
        self.belief.observe_transition(before, after)

    def synchronize_state(self, state: GameState) -> dict[str, Any]:
        if self.first_actor == "unknown":
            if state.turn == 0 and state.is_your_turn:
                self.first_actor = "us"
            elif state.turn <= 2 and state.last_move:
                self.first_actor = "opponent" if state.is_your_turn else "us"
        state.first_actor = self.first_actor
        result = self.belief.synchronize_state(state)
        state.unknown_transitions = len(
            [index for index in self.belief.unknown_action_indices if index <= state.turn]
        )
        state.reconnect_uncertainty = bool(state.unknown_transitions)
        return result

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-12.0, min(12.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _resources_win(player: PlayerState, goal: int) -> bool:
        return min(player.ore, player.mana, player.army) >= goal

    def _won(self, me: PlayerState, enemy: PlayerState, state: GameState) -> bool:
        return me.tower >= state.tower_goal or enemy.tower <= 0 or self._resources_win(me, state.resource_goal)

    def _lost(self, me: PlayerState, enemy: PlayerState, state: GameState) -> bool:
        return me.tower <= 0 or enemy.tower >= state.tower_goal or self._resources_win(enemy, state.resource_goal)

    def _position_probability(self, me: PlayerState, enemy: PlayerState, state: GameState) -> float:
        if self._won(me, enemy, state):
            return 1.0
        if self._lost(me, enemy, state):
            return 0.0

        tower_gap_us = max(0, state.tower_goal - me.tower)
        tower_gap_enemy = max(0, state.tower_goal - enemy.tower)
        tower_race = math.exp(-tower_gap_us / 7.0) - math.exp(-tower_gap_enemy / 7.0)

        effective_enemy_hp = max(1.0, enemy.tower + enemy.wall * 0.68)
        effective_our_hp = max(1.0, me.tower + me.wall * 0.68)
        destruction_race = math.exp(-effective_enemy_hp / 12.0) - math.exp(-effective_our_hp / 12.0)

        resource_us = min(me.ore, me.mana, me.army) / max(1, state.resource_goal)
        resource_enemy = min(enemy.ore, enemy.mana, enemy.army) / max(1, state.resource_goal)
        production = (
            me.mine + me.monastery + me.barracks
            - enemy.mine - enemy.monastery - enemy.barracks
        )
        stock = (me.ore + me.mana + me.army - enemy.ore - enemy.mana - enemy.army)

        # Wall matters through expected absorbed ordinary damage, not as a
        # second tower.  Its marginal value falls once it already covers the
        # typical 6-10 damage attack.
        wall_absorption = min(me.wall, 9) - min(enemy.wall, 9)
        low_tower = max(0, 10 - me.tower) - max(0, 10 - enemy.tower)
        horizon_factor = max(0.55, min(2.5, min(tower_gap_us, tower_gap_enemy) / 12.0))
        logit = (
            3.20 * tower_race
            + 2.20 * destruction_race
            + 2.00 * (resource_us - resource_enemy)
            + 0.16 * horizon_factor * production
            + 0.010 * stock
            + 0.035 * wall_absorption
            - 0.12 * low_tower
        )
        return self._sigmoid(logit)

    def _state_probability(
        self,
        me: PlayerState,
        enemy: PlayerState,
        state: GameState,
        hand: tuple[int, ...] | list[int],
    ) -> float:
        """Calibrated value of the board *and persistent hand*.

        Equal boards after two different discards are not identical states.
        Keeping the hand in this value prevents the old DISCARD branch
        collapse.  The board heuristic is only a fallback when the frozen
        model bundle is unavailable.
        """
        if self.policy_runtime is not None:
            cache_key = (
                me,
                enemy,
                tuple(hand),
                state.turn,
                state.tower_goal,
                state.resource_goal,
                state.first_actor,
            )
            cached = self._decision_state_pwin_cache.get(cache_key)
            if cached is not None:
                return cached
            post_state = replace(
                state,
                players={state.player_no: me, state.opponent_no: enemy},
                hand=list(hand),
                must_discard=False,
            )
            probability = self.policy_runtime.state_pwin(
                self, post_state, me, enemy, list(hand)
            )
            self._decision_state_pwin_cache[cache_key] = probability
            return probability
        return self._position_probability(me, enemy, state)

    def _state_with(self, state: GameState, me: PlayerState, enemy: PlayerState, *, opponent_actor: bool = False) -> GameState:
        if opponent_actor:
            return replace(
                state,
                player_no=state.opponent_no,
                players={state.opponent_no: enemy, state.player_no: me},
            )
        return replace(
            state,
            player_no=state.player_no,
            players={state.player_no: me, state.opponent_no: enemy},
        )

    @staticmethod
    def _cost_text(card: CardDefinition) -> str:
        parts = []
        if card.ore:
            parts.append(f"руда {card.ore}")
        if card.mana:
            parts.append(f"мана {card.mana}")
        if card.army:
            parts.append(f"отряды {card.army}")
        return ", ".join(parts) if parts else "бесплатно"

    @staticmethod
    def _delta(before_me: PlayerState, before_enemy: PlayerState, me: PlayerState, enemy: PlayerState) -> dict[str, int]:
        result: dict[str, int] = {}
        for prefix, before, after in (("ours", before_me, me), ("enemy", before_enemy, enemy)):
            for field_name in asdict(before):
                change = getattr(after, field_name) - getattr(before, field_name)
                if change:
                    result[f"{prefix}_{field_name}"] = change
        return result

    def _eta_map(self, hand: Iterable[int], player: PlayerState) -> dict[str, int | None]:
        result: dict[str, int | None] = {}
        for card_id in hand:
            card = self.catalog.cards.get(card_id)
            if card:
                result[card.name] = self._turns_until_affordable(card, player)
        return result

    def _card_effect_strength(self, card: CardDefinition, me: PlayerState, enemy: PlayerState, state: GameState) -> float:
        funded = replace(
            me,
            ore=max(me.ore, card.ore),
            mana=max(me.mana, card.mana),
            army=max(me.army, card.army),
        )
        projected_me, projected_enemy = self.simulate(card, self._state_with(state, funded, enemy))
        if self._won(projected_me, projected_enemy, state):
            return 100.0
        delta = self._delta(funded, enemy, projected_me, projected_enemy)
        value = 0.0
        value += delta.get("ours_tower", 0) * 2.0 - delta.get("enemy_tower", 0) * 2.2
        value += delta.get("ours_wall", 0) * 0.55 - delta.get("enemy_wall", 0) * 0.45
        value += sum(delta.get(f"ours_{field}", 0) for field in self.PRODUCTION_FIELDS) * 8.0
        value -= sum(delta.get(f"enemy_{field}", 0) for field in self.PRODUCTION_FIELDS) * 6.0
        value += sum(delta.get(f"ours_{field}", 0) for field in ("ore", "mana", "army")) * 0.22
        value -= sum(delta.get(f"enemy_{field}", 0) for field in ("ore", "mana", "army")) * 0.25
        return value

    def _estimated_horizon(self, me: PlayerState, enemy: PlayerState, state: GameState) -> int:
        if self.policy_runtime is not None:
            learned = self.policy_runtime.horizon(self, state, me, enemy, list(state.hand))
            return max(2, min(40, round(learned)))
        tower_distance = min(max(1, state.tower_goal - me.tower), max(1, state.tower_goal - enemy.tower))
        kill_distance = min(max(1, me.tower + me.wall), max(1, enemy.tower + enemy.wall))
        return max(2, min(18, math.ceil(min(tower_distance / 4.0, kill_distance / 6.0)) + 2))

    def _retention_probability_loss(self, card: CardDefinition, state: GameState) -> float:
        if self._affordable(card, state.me):
            projected_me, projected_enemy = self.simulate(card, state)
            if self._lost(projected_me, projected_enemy, state):
                return 0.0
        eta = self._turns_until_affordable(card, state.me)
        if eta is None:
            return 0.005
        horizon = self._estimated_horizon(state.me, state.opponent, state)
        if eta > horizon:
            return max(0.003, 0.018 - 0.002 * (eta - horizon))
        strength = max(0.0, self._card_effect_strength(card, state.me, state.opponent, state))
        reach = 1.0 / (1.0 + eta)
        value = min(0.18, 0.008 + strength * 0.0014 * reach)
        # A strong card that becomes affordable well inside the remaining
        # horizon has option value on every later cycle, not only on the first
        # affordable turn.  This generic horizon term fixes the old bias toward
        # retaining merely-free cards over materially stronger future cards.
        value += strength * max(0, horizon - eta) * 0.0001
        if card.id in self.PRODUCTION_CARDS:
            value += min(0.08, horizon * 0.004 * reach)
        if strength >= 90 and eta <= 2:
            value = max(value, 0.16)
        return min(0.24, value)

    def _hand_option_probability(
        self,
        me: PlayerState,
        enemy: PlayerState,
        state: GameState,
        hand: tuple[int, ...] | list[int],
    ) -> float:
        """Probability-scale option value of the exact cards still held.

        The frozen state model only sees aggregate hand features (playable
        count, ETA and congestion).  It cannot distinguish a harmless free
        discard from a free symmetric card that damages us more than the
        opponent.  Comparing that model directly with the board heuristic also
        mixed two unrelated probability calibrations and produced very large,
        spurious DISCARD gaps.

        Use the already generic ETA/effect/producer retention model instead.
        The value is only consumed as a *delta* from the root hand, so the
        board probability remains calibrated while PLAY, DISCARD and the
        stochastic replacement all keep their real card identity.
        """
        hand_tuple = tuple(hand)
        cache_key = (
            me,
            enemy,
            hand_tuple,
            state.turn,
            state.tower_goal,
            state.resource_goal,
            state.first_actor,
        )
        cached = self._decision_hand_option_cache.get(cache_key)
        if cached is not None:
            return cached
        post_state = replace(
            state,
            players={state.player_no: me, state.opponent_no: enemy},
            hand=list(hand_tuple),
            must_discard=False,
        )
        value = sum(
            self._retention_probability_loss(self.catalog[card_id], post_state)
            for card_id in hand_tuple
            if card_id in self.catalog.cards
        )
        self._decision_hand_option_cache[cache_key] = value
        return value

    def _counterfactual_replacement(
        self,
        state: GameState,
        our_hand: tuple[int, ...],
        opponent_hand: tuple[int, ...],
        last_seen: dict[int, int],
        action_index: int,
        particle_index: int,
        label: tuple[Any, ...],
    ) -> int | None:
        excluded = set(our_hand) | set(opponent_hand)
        pool = [
            (card_id, weight)
            for card_id in self.catalog.cards
            if card_id not in excluded
            and (weight := self.belief._return_weight_at(last_seen.get(card_id), action_index)) > 0.0
        ]
        total = sum(weight for _card_id, weight in pool)
        if total <= 0.0:
            return None
        # The root label deliberately does not contain PLAY/DISCARD or card id:
        # competing actions use the same quantile over their conditioned pools.
        seed_text = self.belief.seed_text(state)
        quantile_key = (seed_text, particle_index, label)
        unit_quantile = self._decision_quantile_cache.get(quantile_key)
        if unit_quantile is None:
            payload = repr((seed_text, particle_index, label)).encode("utf-8")
            unit_quantile = (int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") + 0.5) / (2**64)
            self._decision_quantile_cache[quantile_key] = unit_quantile
        needle = unit_quantile * total
        for card_id, weight in pool:
            needle -= weight
            if needle <= 0.0:
                return card_id
        return pool[-1][0]

    def _transition_hand_with_draw(
        self,
        state: GameState,
        hand: tuple[int, ...],
        opponent_hand: tuple[int, ...],
        slot: int,
        card_id: int,
        last_seen: dict[int, int],
        action_index: int,
        particle_index: int,
        label: tuple[Any, ...],
    ) -> tuple[tuple[int, ...], dict[int, int], int | None]:
        remaining = list(hand)
        if slot >= len(remaining) or remaining[slot] != card_id:
            return hand, last_seen, None
        remaining.pop(slot)
        next_seen = dict(last_seen)
        next_seen[card_id] = action_index
        replacement = self._counterfactual_replacement(
            state,
            tuple(remaining),
            opponent_hand,
            next_seen,
            action_index,
            particle_index,
            label,
        )
        if replacement is not None:
            remaining.insert(min(slot, len(remaining)), replacement)
        return tuple(remaining), next_seen, replacement

    def _replacement_categories(
        self,
        card_ids: list[int],
        me: PlayerState,
        enemy: PlayerState,
        state: GameState,
    ) -> dict[str, float]:
        if not card_ids:
            return {}
        counts: Counter[str] = Counter()
        future_me = self._income(me)
        mirrored = self._state_with(state, future_me, enemy)
        for card_id in card_ids:
            card = self.catalog.cards.get(card_id)
            if card is None:
                continue
            cache_key = (card_id, me, enemy, state.tower_goal, state.resource_goal)
            cached = self._replacement_category_cache.get(cache_key)
            if cached is not None:
                counts[cached] += 1
                continue
            if not self._affordable(card, future_me):
                eta = self._turns_until_affordable(card, future_me)
                category = "neutral" if eta is not None and eta <= 2 else "bad_or_dead"
                self._replacement_category_cache[cache_key] = category
                counts[category] += 1
                continue
            projected_me, projected_enemy = self.simulate(card, mirrored)
            if self._won(projected_me, projected_enemy, state):
                self._replacement_category_cache[cache_key] = "immediate_win"
                counts["immediate_win"] += 1
                continue
            before = self._position_probability(future_me, enemy, state)
            after = self._position_probability(projected_me, projected_enemy, state)
            category = (
                "strong_finisher_or_defense"
                if after - before >= 0.10
                else "useful_playable"
                if after - before >= 0.025
                else "neutral"
            )
            self._replacement_category_cache[cache_key] = category
            counts[category] += 1
        total = max(1, sum(counts.values()))
        return {name: count / total for name, count in sorted(counts.items())}

    def _hand_quality_diagnostics(self, state: GameState) -> dict[str, Any]:
        etas: list[int] = []
        playable = strong = finishers = defense = 0
        resources: Counter[str] = Counter()
        base = self._state_probability(state.me, state.opponent, state, state.hand)
        for slot, card_id in enumerate(state.hand):
            card = self.catalog[card_id]
            eta = self._turns_until_affordable(card, state.me)
            etas.append(99 if eta is None else eta)
            resource = "ore" if card.ore else "mana" if card.mana else "army" if card.army else "free"
            resources[resource] += 1
            if not self._affordable(card, state.me):
                continue
            playable += 1
            projected_me, projected_enemy = self.simulate(card, state)
            next_hand = tuple(
                held for index, held in enumerate(state.hand) if index != slot
            )
            probability = self._state_probability(
                projected_me, projected_enemy, state, next_hand
            )
            strong += int(probability - base >= 0.04)
            finishers += int(self._won(projected_me, projected_enemy, state))
            defense += int(
                state.me.tower <= 12
                and projected_me.tower + projected_me.wall >= state.me.tower + state.me.wall + 5
            )
        horizon = self._estimated_horizon(state.me, state.opponent, state)
        # Congestion is an operational cycling diagnostic, not a claim that a
        # card is worthless for the entire predicted game.  A hand whose cards
        # all need many own turns is effectively dead even when the match-level
        # horizon model predicts a long game.
        cycle_window = min(horizon, 6)
        dead = sum(eta > cycle_window for eta in etas)
        hand_size = max(1, len(state.hand))
        return {
            "hand_playable_count": playable,
            "hand_strong_playable_count": strong,
            "hand_dead_count": dead,
            "hand_avg_eta": statistics.fmean(etas) if etas else None,
            "hand_best_eta": min(etas) if etas else None,
            "hand_finisher_count": finishers,
            "hand_finisher_probability": finishers / hand_size,
            "hand_defense_count": defense,
            "hand_immediate_defense_probability": defense / hand_size,
            "hand_resource_congestion": max(resources.values(), default=0) / hand_size,
            "hand_same_resource_congestion": dict(resources),
            "hand_terminal_relevance": (finishers + defense + strong) / hand_size,
            "hand_cycle_value": dead / hand_size,
            "estimated_horizon": horizon,
        }

    def _win_on_next_action_probability(
        self,
        me: PlayerState,
        enemy: PlayerState,
        hand: tuple[int, ...],
        state: GameState,
        unseen_pool: list[int],
    ) -> float:
        pool_key = tuple(unseen_pool)
        cache_key = (me, enemy, hand, state.tower_goal, state.resource_goal, pool_key)
        cached = self._win_next_cache.get(cache_key)
        if cached is not None:
            return cached
        me = self._income(me)
        mirrored = self._state_with(state, me, enemy)
        for card_id in hand:
            card = self.catalog.cards.get(card_id)
            if card and self._affordable(card, me):
                projected_me, projected_enemy = self.simulate(card, mirrored)
                if self._won(projected_me, projected_enemy, state):
                    self._win_next_cache[cache_key] = 1.0
                    return 1.0
        pool_cache_key = (me, enemy, state.tower_goal, state.resource_goal, pool_key)
        result = self._finisher_pool_cache.get(pool_cache_key)
        if result is None:
            finishers = 0
            for card_id in unseen_pool:
                card = self.catalog[card_id]
                finishers += int(self._fast_immediate_win(card, me, enemy, mirrored))
            result = finishers / max(1, len(unseen_pool))
            self._finisher_pool_cache[pool_cache_key] = result
        self._win_next_cache[cache_key] = result
        return result

    def _fast_immediate_win(
        self,
        card: CardDefinition,
        me: PlayerState,
        enemy: PlayerState,
        state: GameState,
    ) -> bool:
        """Exact fast path for terminal checks without building two dataclasses."""
        if not self._affordable(card, me):
            return False
        tower_gain = self.SELF_GAINS.get(card.id, {}).get("tower", 0)
        if card.id == 62:
            tower_gain = 2 if me.tower < enemy.tower else 1
        elif card.id == 63:
            tower_gain = 1
        if me.tower + tower_gain >= state.tower_goal:
            return True
        direct, general = self._damage(card.id, state)
        if direct + max(0, general - enemy.wall) >= enemy.tower:
            return True
        # Resource finishes are rare and conditional; near the threshold use
        # the full literal simulator rather than an approximate shortcut.
        if min(me.ore, me.mana, me.army) >= state.resource_goal - 20:
            projected_me, projected_enemy = self.simulate(card, state)
            return self._won(projected_me, projected_enemy, state)
        return False

    def _our_extra_continuation(
        self,
        me: PlayerState,
        enemy: PlayerState,
        remaining: tuple[int, ...],
        state: GameState,
        depth: int,
    ) -> tuple[PlayerState, PlayerState, tuple[int, ...], tuple[str, ...]]:
        if depth <= 0 or self._won(me, enemy, state):
            return me, enemy, remaining, tuple()
        candidates: list[tuple[float, PlayerState, PlayerState, tuple[int, ...], tuple[str, ...]]] = []
        mirrored = self._state_with(state, me, enemy)
        for index, card_id in enumerate(remaining):
            card = self.catalog.cards.get(card_id)
            if not card or not self._affordable(card, me):
                continue
            projected_me, projected_enemy = self.simulate(card, mirrored)
            next_hand = remaining[:index] + remaining[index + 1 :]
            sequence = (card.name,)
            if card.id in self.EXTRA_TURN_CARDS and not self._won(projected_me, projected_enemy, state):
                projected_me, projected_enemy, next_hand, tail = self._our_extra_continuation(
                    projected_me, projected_enemy, next_hand, state, depth - 1
                )
                sequence += tail
            probability = self._state_probability(
                projected_me, projected_enemy, state, next_hand
            )
            candidates.append((probability, projected_me, projected_enemy, next_hand, sequence))
        if not candidates:
            return me, enemy, remaining[1:] if remaining else remaining, ("сброс",)
        _probability, best_me, best_enemy, best_hand, best_sequence = max(candidates, key=lambda item: item[0])
        return best_me, best_enemy, best_hand, best_sequence

    def _our_extra_continuation_with_draws(
        self,
        me: PlayerState,
        enemy: PlayerState,
        hand: tuple[int, ...],
        opponent_hand: tuple[int, ...],
        state: GameState,
        particle_index: int,
        action_index: int,
        last_seen: dict[int, int],
        depth: int,
        *,
        forced_discard: bool = False,
    ) -> tuple[PlayerState, PlayerState, tuple[int, ...], tuple[str, ...]]:
        if depth <= 0 or self._won(me, enemy, state) or not hand:
            return me, enemy, hand, tuple()
        candidates: list[tuple[float, PlayerState, PlayerState, tuple[int, ...], tuple[str, ...]]] = []
        mirrored = self._state_with(state, me, enemy)
        for slot, card_id in enumerate(hand):
            card = self.catalog.cards.get(card_id)
            if card is None:
                continue
            action_types = ("drop",) if forced_discard else (
                ("turn", "drop") if self._affordable(card, me) else ("drop",)
            )
            for action_type in action_types:
                next_me, next_enemy = me, enemy
                if action_type == "turn":
                    next_me, next_enemy = self.simulate(card, mirrored)
                next_hand, next_seen, _replacement = self._transition_hand_with_draw(
                    state,
                    hand,
                    opponent_hand,
                    slot,
                    card_id,
                    last_seen,
                    action_index + 1,
                    particle_index,
                    ("extra-draw", depth, forced_discard, slot),
                )
                sequence = (("СБРОС " if action_type == "drop" else "") + card.name,)
                if forced_discard and not self._won(next_me, next_enemy, state):
                    next_me, next_enemy, next_hand, tail = self._our_extra_continuation_with_draws(
                        next_me,
                        next_enemy,
                        next_hand,
                        opponent_hand,
                        state,
                        particle_index,
                        action_index + 1,
                        next_seen,
                        depth - 1,
                    )
                    sequence += tail
                elif action_type == "turn" and card.id in self.EXTRA_TURN_CARDS and not self._won(next_me, next_enemy, state):
                    next_me, next_enemy, next_hand, tail = self._our_extra_continuation_with_draws(
                        next_me,
                        next_enemy,
                        next_hand,
                        opponent_hand,
                        state,
                        particle_index,
                        action_index + 1,
                        next_seen,
                        depth - 1,
                        forced_discard=card.id in {100, 101},
                    )
                    sequence += tail
                probability = self._state_probability(
                    next_me, next_enemy, state, next_hand
                )
                candidates.append((probability, next_me, next_enemy, next_hand, sequence))
        if not candidates:
            return me, enemy, hand, tuple()
        _probability, best_me, best_enemy, best_hand, best_sequence = max(
            candidates, key=lambda item: item[0]
        )
        return best_me, best_enemy, best_hand, best_sequence

    def _opponent_reply(
        self,
        me: PlayerState,
        enemy: PlayerState,
        opponent_hand: tuple[int, ...],
        our_remaining_hand: tuple[int, ...],
        state: GameState,
        unseen_pool: list[int],
        reply_cache: dict[tuple[Any, ...], tuple[float, bool, PlayerState, PlayerState, float]],
        extra_reply_cache: dict[tuple[Any, ...], dict[int, tuple[float, bool, PlayerState, PlayerState, float]]],
        next_win_cache: dict[tuple[Any, ...], bool],
        depth: int = 2,
        deadline: float | None = None,
    ) -> tuple[float, bool, bool, float, int | None]:
        enemy = self._income(enemy)
        if self._won(enemy, me, self._state_with(state, me, enemy, opponent_actor=True)):
            return 0.0, True, True, 0.0, None

        # Discard is always legal and passes the move back to us.
        base_probability = self._position_probability(self._income(me), enemy, state)
        candidates: list[
            tuple[float, bool, float, int | None, PlayerState, PlayerState, int, bool | None]
        ] = [
            (
                base_probability,
                False,
                self._win_on_next_action_probability(me, enemy, our_remaining_hand, state, unseen_pool),
                None,
                enemy,
                me,
                -1,
                None,
            )
        ]
        mirrored = self._state_with(state, me, enemy, opponent_actor=True)
        for index, card_id in enumerate(opponent_hand):
            if deadline is not None and time.monotonic() >= deadline:
                break
            card = self.catalog.cards.get(card_id)
            if not card or not self._affordable(card, enemy):
                continue
            remaining_enemy = opponent_hand[:index] + opponent_hand[index + 1 :]
            if card.id in self.EXTRA_TURN_CARDS and depth > 1:
                extra_cache_key = (card.id, me, enemy, our_remaining_hand)
                continuations = extra_reply_cache.get(extra_cache_key)
                if continuations is None:
                    first_enemy, first_me = self.simulate(card, mirrored)
                    first_loss = self._won(first_enemy, first_me, mirrored)
                    first_probability = 0.0 if first_loss else self._position_probability(
                        self._income(first_me), first_enemy, state
                    )
                    first_win_next = 0.0 if first_loss else self._win_on_next_action_probability(
                        first_me, first_enemy, our_remaining_hand, state, unseen_pool
                    )
                    continuations = {
                        -1: (first_probability, first_loss, first_enemy, first_me, first_win_next)
                    }
                    if not first_loss:
                        second_state = self._state_with(state, first_me, first_enemy, opponent_actor=True)
                        for second_id in unseen_pool:
                            if deadline is not None and time.monotonic() >= deadline:
                                break
                            if second_id == card.id:
                                continue
                            second_card = self.catalog[second_id]
                            if not self._affordable(second_card, first_enemy):
                                continue
                            second_enemy, second_me = self.simulate(second_card, second_state)
                            second_loss = self._won(second_enemy, second_me, second_state)
                            second_probability = 0.0 if second_loss else self._position_probability(
                                self._income(second_me), second_enemy, state
                            )
                            second_win_next = 0.0 if second_loss else self._win_on_next_action_probability(
                                second_me, second_enemy, our_remaining_hand, state, unseen_pool
                            )
                            continuations[second_id] = (
                                second_probability,
                                second_loss,
                                second_enemy,
                                second_me,
                                second_win_next,
                            )
                    extra_reply_cache[extra_cache_key] = continuations
                available = [continuations[-1]]
                available.extend(
                    continuations[second_id]
                    for second_id in remaining_enemy
                    if second_id in continuations
                )
                probability, immediate_loss, reply_enemy, reply_me, our_win_next = min(
                    available, key=lambda item: item[0]
                )
                candidates.append(
                    (
                        probability,
                        immediate_loss,
                        our_win_next,
                        card_id,
                        reply_enemy,
                        reply_me,
                        index,
                        immediate_loss,
                    )
                )
                continue
            reply_cache_key = (card_id, me, enemy, our_remaining_hand)
            cached = reply_cache.get(reply_cache_key)
            if cached is None:
                reply_enemy, reply_me = self.simulate(card, mirrored)
                immediate_loss = self._won(reply_enemy, reply_me, mirrored)
                probability = 0.0 if immediate_loss else self._position_probability(
                    self._income(reply_me), reply_enemy, state
                )
                our_win_next = 0.0 if immediate_loss else self._win_on_next_action_probability(
                    reply_me, reply_enemy, our_remaining_hand, state, unseen_pool
                )
                cached = (probability, immediate_loss, reply_enemy, reply_me, our_win_next)
                reply_cache[reply_cache_key] = cached
            probability, immediate_loss, reply_enemy, reply_me, our_win_next = cached
            candidates.append(
                (probability, immediate_loss, our_win_next, card_id, reply_enemy, reply_me, index, None)
            )

        best = min(candidates, key=lambda item: item[0])
        probability, immediate_loss, our_win_next, reply_card, reply_enemy, reply_me, index, opponent_override = best
        if opponent_override is not None:
            opponent_win_later = opponent_override
        else:
            remaining_enemy = opponent_hand if index < 0 else opponent_hand[:index] + opponent_hand[index + 1 :]
            opponent_win_later = immediate_loss
            for remaining_id in remaining_enemy:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                cache_key = (
                    reply_card if reply_card is not None else -1,
                    remaining_id,
                    reply_me,
                    reply_enemy,
                )
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
        return probability, immediate_loss, opponent_win_later, our_win_next, reply_card

    def _opponent_extra_reply(
        self,
        me: PlayerState,
        enemy: PlayerState,
        opponent_hand: tuple[int, ...],
        our_remaining_hand: tuple[int, ...],
        state: GameState,
        unseen_pool: list[int],
        depth: int,
        deadline: float | None = None,
    ) -> tuple[float, bool, bool, float]:
        base = (
            self._position_probability(self._income(me), enemy, state),
            False,
            False,
            self._win_on_next_action_probability(me, enemy, our_remaining_hand, state, unseen_pool),
        )
        if depth <= 0:
            return base
        choices = [base]
        mirrored = self._state_with(state, me, enemy, opponent_actor=True)
        for index, card_id in enumerate(opponent_hand):
            if deadline is not None and time.monotonic() >= deadline:
                break
            card = self.catalog.cards.get(card_id)
            if not card or not self._affordable(card, enemy):
                continue
            next_enemy, next_me = self.simulate(card, mirrored)
            if self._won(next_enemy, next_me, mirrored):
                choices.append((0.0, True, True, 0.0))
                continue
            remaining = opponent_hand[:index] + opponent_hand[index + 1 :]
            if card.id in self.EXTRA_TURN_CARDS:
                choices.append(
                    self._opponent_extra_reply(
                        next_me,
                        next_enemy,
                        remaining,
                        our_remaining_hand,
                        state,
                        unseen_pool,
                        depth - 1,
                        deadline,
                    )
                )
            else:
                choices.append(
                    (
                        self._position_probability(self._income(next_me), next_enemy, state),
                        False,
                        self._particle_has_next_win(next_enemy, next_me, remaining, state),
                        self._win_on_next_action_probability(
                            next_me, next_enemy, our_remaining_hand, state, unseen_pool
                        ),
                    )
                )
        return min(choices, key=lambda item: item[0])

    def _particle_has_next_win(
        self,
        attacker: PlayerState,
        defender: PlayerState,
        hand: tuple[int, ...],
        state: GameState,
    ) -> bool:
        cache_key = (attacker, defender, hand, state.tower_goal, state.resource_goal)
        cached = self._particle_next_win_cache.get(cache_key)
        if cached is not None:
            return cached
        attacker = self._income(attacker)
        mirrored = self._state_with(state, defender, attacker, opponent_actor=True)
        for card_id in hand:
            card = self.catalog.cards.get(card_id)
            if card and self._affordable(card, attacker):
                projected_attacker, projected_defender = self.simulate(card, mirrored)
                if self._won(projected_attacker, projected_defender, mirrored):
                    self._particle_next_win_cache[cache_key] = True
                    return True
        self._particle_next_win_cache[cache_key] = False
        return False

    def _belief_threats(
        self,
        state: GameState,
        particles: list[tuple[int, ...]],
    ) -> tuple[str, ...]:
        pool = self.belief.weighted_pool(state)
        structurally_certain = {card_id for card_id, _weight in pool} if 0 < len(pool) <= 6 else set()
        probabilities = self.belief.probabilities(particles, structurally_certain)
        enemy = self._income(state.opponent)
        mirrored = self._state_with(state, state.me, enemy, opponent_actor=True)
        threats: list[tuple[float, str]] = []
        for card_id, probability in probabilities.items():
            card = self.catalog[card_id]
            if not self._affordable(card, enemy):
                continue
            projected_enemy, projected_me = self.simulate(card, mirrored)
            lethal = self._won(projected_enemy, projected_me, mirrored)
            swing = self._position_probability(state.me, state.opponent, state) - self._position_probability(
                projected_me, projected_enemy, state
            )
            priority = probability * (5.0 if lethal else max(0.0, swing))
            if lethal or swing >= 0.04:
                threats.append((priority, f"{card.name}: {probability:.1%}"))
        threats.sort(reverse=True)
        return tuple(text for _priority, text in threats[:5])

    def _evaluate_choice(
        self,
        action: str,
        slot: int,
        card: CardDefinition,
        state: GameState,
        particles: list[tuple[int, ...]],
        unseen_pool: list[int],
        top_threats: tuple[str, ...],
        *,
        sample_sink: dict[str, Any] | None = None,
        deadline: float | None = None,
        sample_offset: int = 0,
    ) -> CardDecision:
        remaining_hand = tuple(card_id for index, card_id in enumerate(state.hand) if index != slot)
        if action == "turn":
            after_me, after_enemy = self.simulate(card, state)
        else:
            after_me, after_enemy = state.me, state.opponent
        sequence: tuple[str, ...] = tuple()

        immediate_probability = self._position_probability(after_me, after_enemy, state)
        eta_before = self._eta_map(remaining_hand, state.me)
        eta_after = self._eta_map(remaining_hand, after_me)
        unlocks = tuple(
            name
            for name, eta in eta_after.items()
            if eta is not None and (eta_before.get(name) is None or eta < eta_before[name])
        )
        next_income = self._income(after_me)
        unlocked_next = tuple(
            self.catalog[card_id].name
            for card_id in remaining_hand
            if card_id in self.catalog.cards
            and not self._affordable(self.catalog[card_id], after_me)
            and self._affordable(self.catalog[card_id], next_income)
        )

        if self._won(after_me, after_enemy, state):
            expected = cvar = adjusted = 1.0
            lose_next = opponent_two = 0.0
            win_two = 1.0
            reply_card_ids: list[int] = []
            if sample_sink is not None:
                sample_sink["terminal"] = True
                sample_sink["terminal_value"] = 1.0
                sample_sink["outcomes"] = []
                sample_sink["lose_flags"] = []
                sample_sink["opponent_two_flags"] = []
                sample_sink["win_two_values"] = []
                sample_sink["reply_card_ids"] = []
        elif self._lost(after_me, after_enemy, state):
            expected = cvar = adjusted = 0.0
            lose_next = opponent_two = 1.0
            win_two = 0.0
            reply_card_ids = []
            if sample_sink is not None:
                sample_sink["terminal"] = True
                sample_sink["terminal_value"] = 0.0
                sample_sink["outcomes"] = []
                sample_sink["lose_flags"] = []
                sample_sink["opponent_two_flags"] = []
                sample_sink["win_two_values"] = []
                sample_sink["reply_card_ids"] = []
        else:
            outcomes: list[float] = []
            lose_flags: list[bool] = []
            opponent_two_flags: list[bool] = []
            win_two_values: list[float] = []
            reply_card_ids = []
            replacement_card_ids: list[int] = []
            sequence_counts: Counter[tuple[str, ...]] = Counter()
            # Reuse deterministic reply prefixes across candidate actions and
            # adaptive batches within this decision.
            reply_cache = self._decision_reply_cache
            extra_reply_cache = self._decision_extra_reply_cache
            next_win_cache = self._decision_next_win_cache
            root_last_seen = {
                card_id: int(item["action"])
                for card_id, item in self.belief.last_seen.items()
            }
            root_last_seen[card.id] = state.turn + 1
            for particle_index, particle in enumerate(particles):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                global_particle_index = sample_offset + particle_index
                last_seen = root_last_seen
                replacement = self._counterfactual_replacement(
                    state,
                    remaining_hand,
                    particle,
                    last_seen,
                    state.turn + 1,
                    global_particle_index,
                    ("root-draw",),
                )
                particle_hand_list = list(remaining_hand)
                if replacement is not None:
                    particle_hand_list.insert(min(slot, len(particle_hand_list)), replacement)
                particle_hand = tuple(particle_hand_list)
                if replacement is not None:
                    replacement_card_ids.append(replacement)
                particle_me, particle_enemy = after_me, after_enemy
                particle_sequence: tuple[str, ...] = tuple()
                if action == "turn" and card.id in self.EXTRA_TURN_CARDS and not self._won(
                    particle_me, particle_enemy, state
                ):
                    particle_me, particle_enemy, particle_hand, particle_sequence = (
                        self._our_extra_continuation_with_draws(
                            particle_me,
                            particle_enemy,
                            particle_hand,
                            particle,
                            state,
                            global_particle_index,
                            state.turn + 1,
                            last_seen,
                            self.EXTRA_TURN_DEPTH,
                            forced_discard=card.id in {100, 101},
                        )
                    )
                    sequence_counts[particle_sequence] += 1
                hand_value_adjustment = (
                    self._state_probability(
                        particle_me, particle_enemy, state, particle_hand
                    )
                    - self._position_probability(particle_me, particle_enemy, state)
                )
                if self._won(particle_me, particle_enemy, state):
                    probability, immediate_loss, opponent_two, win_two_value, reply_card = (
                        1.0,
                        False,
                        False,
                        1.0,
                        None,
                    )
                else:
                    probability, immediate_loss, opponent_two, win_two_value, reply_card = self._opponent_reply(
                        particle_me,
                        particle_enemy,
                        particle,
                        particle_hand,
                        state,
                        unseen_pool,
                        reply_cache,
                        extra_reply_cache,
                        next_win_cache,
                        deadline=deadline,
                    )
                    if not immediate_loss:
                        probability = min(
                            1.0, max(0.0, probability + hand_value_adjustment)
                        )
                outcomes.append(probability)
                lose_flags.append(immediate_loss)
                opponent_two_flags.append(opponent_two)
                win_two_values.append(win_two_value)
                if reply_card is not None:
                    reply_card_ids.append(reply_card)
            if sample_sink is not None:
                sample_sink["terminal"] = False
                sample_sink["outcomes"] = outcomes
                sample_sink["lose_flags"] = lose_flags
                sample_sink["opponent_two_flags"] = opponent_two_flags
                sample_sink["win_two_values"] = win_two_values
                sample_sink["reply_card_ids"] = reply_card_ids
                sample_sink["replacement_card_ids"] = replacement_card_ids
            expected = sum(outcomes) / max(1, len(outcomes))
            tail_count = max(1, math.ceil(len(outcomes) * self.TAIL_FRACTION))
            cvar = sum(sorted(outcomes)[:tail_count]) / tail_count
            tail_gap = max(0.0, expected - cvar)
            current_threat = sum(lose_flags) / max(1, len(lose_flags))
            risk_lambda = self.CRITICAL_RISK_LAMBDA if current_threat >= 0.18 or state.me.tower <= 10 else self.NORMAL_RISK_LAMBDA
            survival_penalty = current_threat * (0.15 if state.me.tower <= 10 else 0.03)
            adjusted = expected - risk_lambda * tail_gap - survival_penalty
            lose_next = current_threat
            opponent_two = sum(opponent_two_flags) / max(1, len(opponent_two_flags))
            win_two = sum(win_two_values) / max(1, len(win_two_values))

        # Unlocking a strong card on the next cycle is a real option, but keep
        # it inside probability scale rather than adding thousands of points.
        unlock_bonus = 0.0
        for card_id in remaining_hand:
            held = self.catalog.cards.get(card_id)
            if held and held.name in unlocked_next:
                strength = self._card_effect_strength(held, after_me, after_enemy, state)
                unlock_bonus += min(0.045, max(0.0, strength) * 0.00055)
        adjusted = min(1.0, max(0.0, adjusted + unlock_bonus))

        # The learned state value sees aggregate hand properties but not exact
        # card identity.  Charge the generic horizon-aware option cost of the
        # discarded card once; the stochastic replacement remains modeled in
        # every particle.
        retention_loss = self._retention_probability_loss(card, state) if action == "drop" else 0.0
        discard_identity_adjustment = 0.0
        forced_cycle = state.must_discard or not any(
            candidate_id in self.catalog.cards
            and self._affordable(self.catalog[candidate_id], state.me)
            for candidate_id in state.hand
        )
        if action == "drop" and forced_cycle:
            exact_strength = self._card_effect_strength(
                card, state.me, state.opponent, state
            )
            # Aggregate state-value features cannot tell two exact card ids
            # apart.  Losing a useful option is a cost; cycling a currently
            # harmful option is a benefit.  Both remain generic functions of
            # literal effect and horizon, never card-id bonuses.
            discard_identity_adjustment = (
                -retention_loss + min(0.06, max(0.0, -exact_strength) * 0.03)
            )
        symmetric_exchange_adjustment = 0.0
        normalized_effect = card.effect.lower().replace("ё", "е")
        if action == "turn" and ("все " in normalized_effect or "обоих" in normalized_effect):
            our_effect_losses = 0
            enemy_effect_losses = 0
            for field_name, paid in (
                ("ore", card.ore),
                ("mana", card.mana),
                ("army", card.army),
            ):
                our_effect_delta = (
                    getattr(after_me, field_name) - getattr(state.me, field_name) + paid
                )
                enemy_effect_delta = getattr(after_enemy, field_name) - getattr(
                    state.opponent, field_name
                )
                our_effect_losses += max(0, -our_effect_delta)
                enemy_effect_losses += max(0, -enemy_effect_delta)
            symmetric_exchange_adjustment = max(
                -0.04,
                min(0.04, (enemy_effect_losses - our_effect_losses) * 0.002),
            )
        adjusted = min(
            1.0,
            max(
                0.0,
                adjusted
                + discard_identity_adjustment
                + symmetric_exchange_adjustment,
            ),
        )
        if sample_sink is not None:
            sample_sink["unlock_bonus"] = unlock_bonus
            sample_sink["retention_loss"] = retention_loss
            sample_sink["discard_identity_adjustment"] = discard_identity_adjustment
            sample_sink["symmetric_exchange_adjustment"] = symmetric_exchange_adjustment
        tail_risk = max(0.0, expected - cvar)
        delta = self._delta(state.me, state.opponent, after_me, after_enemy)
        reply_name = "сброс"
        if reply_card_ids:
            reply_name = self.catalog[Counter(reply_card_ids).most_common(1)[0][0]].name
        if 'sequence_counts' in locals() and sequence_counts:
            sequence = sequence_counts.most_common(1)[0][0]
        reasons: list[str] = []
        if adjusted >= 0.999:
            reasons.append("немедленная или extra-turn победа")
        if lose_next > 0:
            reasons.append(f"риск поражения следующим ходом {lose_next:.1%}")
        if unlocked_next:
            reasons.append("следующим циклом доступны: " + ", ".join(unlocked_next[:3]))
        if unlocks:
            reasons.append("ускоряет доступ: " + ", ".join(unlocks[:3]))
        if action == "drop":
            replacement_count = len(replacement_card_ids) if 'replacement_card_ids' in locals() else 0
            reasons.append(f"stochastic replacement смоделирован ({replacement_count} частиц)")
        if sequence:
            reasons.append("продолжение: " + " → ".join(sequence))
        reasons.append(f"типичный лучший ответ: {reply_name}")
        reasons.append(f"P(win) {adjusted:.1%}, хвостовой риск {tail_risk:.1%}")

        return CardDecision(
            action,
            slot,
            card,
            adjusted * 100.0,
            tuple(reasons),
            immediate_score=immediate_probability * 100.0,
            response_value=expected * 100.0,
            predicted_me=asdict(after_me),
            predicted_opponent=asdict(after_enemy),
            winning_replies_before=round(lose_next * self.PARTICLE_COUNT),
            winning_replies_after=round(lose_next * self.PARTICLE_COUNT),
            discard_retention=retention_loss * 100.0,
            legal_cost=self._cost_text(card),
            immediate_state_delta=delta,
            p_win=adjusted,
            p_win_next_action=win_two,
            p_lose_next_turn=lose_next,
            p_win_within_2_own_actions=win_two,
            p_opponent_win_within_2_actions=opponent_two,
            expected_reply_value=expected,
            tail_risk=tail_risk,
            immediate_terminal_win=self._won(after_me, after_enemy, state),
            resource_unlocks=unlocks,
            cards_unlocked_next_turn=unlocked_next,
            eta_key_hand_cards=eta_after,
            extra_turn_continuation=" → ".join(sequence),
            opponent_belief_top_threats=top_threats,
            replacement_distribution=self._replacement_categories(
                replacement_card_ids if 'replacement_card_ids' in locals() else [],
                after_me,
                after_enemy,
                state,
            ),
        )

    def score(self, card: CardDefinition, state: GameState, *, for_discard: bool = False) -> tuple[float, list[str]]:
        if for_discard:
            loss = self._retention_probability_loss(card, state)
            return -loss * 100.0, [f"будущая ценность карты {loss:.1%}"]
        after_me, after_enemy = self.simulate(card, state)
        probability = self._position_probability(after_me, after_enemy, state)
        delta = self._delta(state.me, state.opponent, after_me, after_enemy)
        return probability * 100.0, ["фактический эффект: " + (", ".join(f"{key} {value:+d}" for key, value in delta.items()) or "нет")]

    def _aggregate_samples(
        self,
        base: CardDecision,
        samples: dict[str, Any],
        state: GameState,
        top_threats: tuple[str, ...],
    ) -> CardDecision:
        outcomes = list(samples.get("outcomes") or [])
        terminal = bool(samples.get("terminal"))
        if terminal:
            terminal_value = float(samples.get("terminal_value", 1.0))
            expected = cvar = adjusted = terminal_value
            lose_next = opponent_two = 1.0 - terminal_value
            win_two = terminal_value
            reply_card_ids: list[int] = []
            particle_count = len(outcomes)
        else:
            particle_count = len(outcomes)
            expected = statistics.fmean(outcomes) if outcomes else 0.0
            tail_count = max(1, math.ceil(max(1, particle_count) * self.TAIL_FRACTION))
            cvar = sum(sorted(outcomes)[:tail_count]) / tail_count if outcomes else 0.0
            lose_flags = list(samples.get("lose_flags") or [])
            opponent_two_flags = list(samples.get("opponent_two_flags") or [])
            win_two_values = list(samples.get("win_two_values") or [])
            lose_next = statistics.fmean(lose_flags) if lose_flags else 0.0
            opponent_two = statistics.fmean(opponent_two_flags) if opponent_two_flags else 0.0
            win_two = statistics.fmean(win_two_values) if win_two_values else 0.0
            tail_gap = max(0.0, expected - cvar)
            risk_lambda = (
                self.CRITICAL_RISK_LAMBDA
                if lose_next >= 0.18 or state.me.tower <= 10
                else self.NORMAL_RISK_LAMBDA
            )
            survival_penalty = lose_next * (0.15 if state.me.tower <= 10 else 0.03)
            adjusted = expected - risk_lambda * tail_gap - survival_penalty
            reply_card_ids = list(samples.get("reply_card_ids") or [])
        retention_loss = float(samples.get("retention_loss") or 0.0)
        discard_identity_adjustment = float(
            samples.get("discard_identity_adjustment") or 0.0
        )
        symmetric_exchange_adjustment = float(
            samples.get("symmetric_exchange_adjustment") or 0.0
        )
        adjusted = min(
            1.0,
            max(
                0.0,
                adjusted
                + float(samples.get("unlock_bonus") or 0.0)
                + discard_identity_adjustment
                + symmetric_exchange_adjustment,
            ),
        )
        replacement_card_ids = list(samples.get("replacement_card_ids") or [])
        tail_risk = max(0.0, expected - cvar)
        reply_name = "сброс"
        if reply_card_ids:
            reply_name = self.catalog[Counter(reply_card_ids).most_common(1)[0][0]].name
        reasons: list[str] = []
        if adjusted >= 0.999:
            reasons.append("немедленная или extra-turn победа")
        if lose_next > 0:
            reasons.append(f"риск поражения следующим ходом {lose_next:.1%}")
        if base.cards_unlocked_next_turn:
            reasons.append("следующим циклом доступны: " + ", ".join(base.cards_unlocked_next_turn[:3]))
        if base.resource_unlocks:
            reasons.append("ускоряет доступ: " + ", ".join(base.resource_unlocks[:3]))
        if base.action == "drop":
            reasons.append(
                "retention diagnostics: "
                f"ценность опции {retention_loss:.1%}, "
                f"exact-card поправка {discard_identity_adjustment:+.1%}"
            )
        if symmetric_exchange_adjustment:
            reasons.append(
                "фактический симметричный обмен "
                f"{symmetric_exchange_adjustment:+.1%} к P(win)"
            )
        if base.extra_turn_continuation:
            reasons.append("продолжение: " + base.extra_turn_continuation)
        reasons.append(f"типичный лучший ответ: {reply_name}")
        reasons.append(f"P(win) {adjusted:.1%}, хвостовой риск {tail_risk:.1%}")
        return replace(
            base,
            score=adjusted * 100.0,
            reasons=tuple(reasons),
            response_value=expected * 100.0,
            winning_replies_before=round(lose_next * particle_count),
            winning_replies_after=round(lose_next * particle_count),
            p_win=adjusted,
            p_win_next_action=win_two,
            p_lose_next_turn=lose_next,
            p_win_within_2_own_actions=win_two,
            p_opponent_win_within_2_actions=opponent_two,
            expected_reply_value=expected,
            tail_risk=tail_risk,
            opponent_belief_top_threats=top_threats,
            particle_count=particle_count,
            replacement_distribution=self._replacement_categories(
                replacement_card_ids,
                state.me,
                state.opponent,
                state,
            ),
        )

    @staticmethod
    def _choice_sort_key(item: CardDecision) -> tuple[float, float, float, float, int, int]:
        """Final action objective, ordered from strongest to weakest key.

        The 200-particle holdout audit showed that displayed P(win)-first has
        materially lower high-budget oracle regret than learned-policy-first.
        The learned policy is diagnostic-only.  It never changes production
        selection, including exact floating-point ties.
        """
        return (
            1.0 if item.immediate_terminal_win else 0.0,
            item.p_win if item.p_win is not None else -1.0,
            -(item.p_lose_next_turn if item.p_lose_next_turn is not None else 1.0),
            -(item.tail_risk if item.tail_risk is not None else 1.0),
            1 if item.action == "turn" else 0,
            -item.slot,
        )

    @staticmethod
    def _policy_diagnostic_sort_key(
        item: CardDecision,
    ) -> tuple[float, float, float, float, float]:
        return (
            1.0 if item.immediate_terminal_win else 0.0,
            item.policy_score
            if item.policy_score is not None
            else (item.p_win if item.p_win is not None else -1.0),
            item.p_win if item.p_win is not None else -1.0,
            -(item.p_lose_next_turn if item.p_lose_next_turn is not None else 1.0),
            -(item.tail_risk if item.tail_risk is not None else 1.0),
        )

    @staticmethod
    def _practical_tie_key(item: CardDecision) -> tuple[float, float, float, float, int, int]:
        """Tie-break only after paired P(win) practical equivalence is proven."""
        return (
            -(item.p_lose_next_turn if item.p_lose_next_turn is not None else 1.0),
            # When survival risk is identical, discard the least valuable
            # future option before consulting noisy lower-tail differences.
            -item.discard_retention if item.action == "drop" else 0.0,
            -(item.tail_risk if item.tail_risk is not None else 1.0),
            item.p_win_within_2_own_actions
            if item.p_win_within_2_own_actions is not None
            else -1.0,
            1 if item.action == "turn" else 0,
            -item.slot,
        )

    def _with_diagnostic_ranks(
        self,
        choices: list[CardDecision],
        *,
        selected_key: tuple[str, int] | None = None,
        practical_equivalence: bool = False,
    ) -> list[CardDecision]:
        pwin_order = sorted(choices, key=self._choice_sort_key, reverse=True)
        policy_order = sorted(choices, key=self._policy_diagnostic_sort_key, reverse=True)
        pwin_ranks = {(item.action, item.slot): index for index, item in enumerate(pwin_order, 1)}
        policy_ranks = {(item.action, item.slot): index for index, item in enumerate(policy_order, 1)}
        result = []
        for choice in choices:
            key = (choice.action, choice.slot)
            pwin_rank = pwin_ranks[key]
            policy_rank = policy_ranks[key]
            is_selected = selected_key == key if selected_key is not None else pwin_rank == 1
            if is_selected and practical_equivalence and pwin_rank != 1:
                reason = "выбран risk/tactical tie-break внутри доказанной P(win)-equivalence zone"
            elif pwin_rank == 1 and policy_rank == 1:
                reason = "P(win) и learned policy согласны"
            elif pwin_rank == 1:
                reason = "выбран по главному objective P(win); learned policy не согласен"
            elif policy_rank == 1:
                reason = "learned policy предпочитает, но P(win) ниже"
            else:
                reason = "уступает по P(win) и learned policy"
            result.append(
                replace(
                    choice,
                    policy_rank=policy_rank,
                    pwin_rank=pwin_rank,
                    final_rank_reason=reason,
                )
            )
        result.sort(key=self._choice_sort_key, reverse=True)
        if selected_key is not None:
            result.sort(
                key=lambda item: (item.action, item.slot) == selected_key,
                reverse=True,
            )
        return result

    def _pwin_influence_samples(
        self,
        samples: dict[str, Any],
        state: GameState,
    ) -> list[float]:
        """Influence values for the actual risk-adjusted P(win) objective."""
        outcomes = list(samples.get("outcomes") or [])
        if not outcomes:
            return []
        if (choice_pwin := samples.get("aggregated_pwin")) is not None and (
            float(choice_pwin) <= 0.0 or float(choice_pwin) >= 1.0
        ):
            return [float(choice_pwin)] * len(outcomes)
        lose_flags = list(samples.get("lose_flags") or [False] * len(outcomes))
        tail_count = max(1, math.ceil(len(outcomes) * self.TAIL_FRACTION))
        threshold = sorted(outcomes)[tail_count - 1]
        alpha = tail_count / len(outcomes)
        lose_next = statistics.fmean(lose_flags) if lose_flags else 0.0
        risk_lambda = (
            self.CRITICAL_RISK_LAMBDA
            if lose_next >= 0.18 or state.me.tower <= 10
            else self.NORMAL_RISK_LAMBDA
        )
        survival_weight = 0.15 if state.me.tower <= 10 else 0.03
        deterministic_adjustment = (
            float(samples.get("unlock_bonus") or 0.0)
            + float(samples.get("discard_identity_adjustment") or 0.0)
            + float(samples.get("symmetric_exchange_adjustment") or 0.0)
        )
        result = []
        for index, outcome in enumerate(outcomes):
            cvar_influence = threshold + min(0.0, outcome - threshold) / alpha
            result.append(
                (1.0 - risk_lambda) * outcome
                + risk_lambda * cvar_influence
                - survival_weight * float(bool(lose_flags[index]))
                + deterministic_adjustment
            )
        return result

    def _policy_mc_coefficient(self, choice: CardDecision, state: GameState) -> float:
        if choice.policy_score is None:
            return 0.0
        coefficient = 4.0 if state.me.tower <= 10 else 0.75
        if choice.action == "turn":
            projected_me, projected_enemy = self.simulate(choice.card, state)
            if self._lost(projected_me, projected_enemy, state):
                return 0.0
            production_gain = sum(
                max(0, getattr(projected_me, field) - getattr(state.me, field))
                for field in self.PRODUCTION_FIELDS
            )
            if production_gain:
                horizon = self.policy_runtime.horizon(
                    self, state, state.me, state.opponent, list(state.hand)
                )
                coefficient += 0.015 * min(40.0, horizon) * production_gain
        return coefficient

    def _policy_score_samples(
        self,
        choice: CardDecision,
        samples: dict[str, Any],
        state: GameState,
    ) -> list[float]:
        lose_flags = list(samples.get("lose_flags") or [])
        if not lose_flags or choice.policy_score is None:
            return []
        coefficient = self._policy_mc_coefficient(choice, state)
        deterministic = float(choice.policy_score) + coefficient * float(
            choice.p_lose_next_turn or 0.0
        )
        return [deterministic - coefficient * float(bool(flag)) for flag in lose_flags]

    @staticmethod
    def _paired_statistics(
        margin: float,
        left: list[float],
        right: list[float],
    ) -> tuple[float | None, tuple[float, float] | None, bool]:
        paired_count = min(len(left), len(right))
        if paired_count < 2:
            return None, None, False
        differences = [left[index] - right[index] for index in range(paired_count)]
        se_diff = statistics.pstdev(differences) / math.sqrt(paired_count)
        half_width = 1.96 * se_diff
        exact_equivalence = max(differences) - min(differences) <= 1e-12 and abs(margin) <= 1e-12
        return se_diff, (margin - half_width, margin + half_width), exact_equivalence

    def _with_policy_scores(self, state: GameState, choices: list[CardDecision]) -> list[CardDecision]:
        if self.policy_runtime is None:
            return choices
        horizon = self.policy_runtime.horizon(self, state, state.me, state.opponent, list(state.hand))
        scored = []
        for choice in choices:
            score = self.policy_runtime.action_score(self, state, choice.action, choice.slot)
            risk_weight = 4.0 if state.me.tower <= 10 else 0.75
            score -= risk_weight * (choice.p_lose_next_turn or 0.0)
            if choice.action == "drop":
                score -= 2.0 * max(0.0, choice.discard_retention)
                score -= 0.15 * self._card_effect_strength(
                    choice.card, state.me, state.opponent, state
                )
            else:
                projected_me, projected_enemy = self.simulate(choice.card, state)
                if self._lost(projected_me, projected_enemy, state):
                    score = -1_000_000.0
                production_gain = sum(
                    max(0, getattr(projected_me, field) - getattr(state.me, field))
                    for field in self.PRODUCTION_FIELDS
                )
                # Horizon-aware option value: each extra producer is useful
                # only while the game is likely to survive.  This is a generic
                # mechanism and contains no per-card exception.
                if production_gain:
                    score += 0.015 * min(40.0, horizon) * production_gain * (1.0 - (choice.p_lose_next_turn or 0.0))
            scored.append(replace(choice, policy_score=score))
        return scored

    def rank_choices(self, state: GameState) -> list[CardDecision]:
        # Decision-scoped caches only memoize deterministic calculations.  They
        # are cleared between states so memory stays bounded and reconnect
        # history can never leak into another decision.
        self._simulate_cache.clear()
        self._win_next_cache.clear()
        self._finisher_pool_cache.clear()
        self._particle_next_win_cache.clear()
        self._decision_reply_cache.clear()
        self._decision_extra_reply_cache.clear()
        self._decision_next_win_cache.clear()
        self._decision_quantile_cache.clear()
        self._decision_state_pwin_cache.clear()
        self._decision_hand_option_cache.clear()
        if self.first_actor == "unknown":
            if state.turn == 0 and state.is_your_turn:
                self.first_actor = "us"
            elif state.turn <= 2 and state.last_move:
                self.first_actor = "opponent" if state.is_your_turn else "us"
        state.first_actor = self.first_actor
        unseen_pool = self.belief.unseen_pool(state)
        candidates: list[tuple[str, int, CardDefinition]] = []
        for slot, card_id in enumerate(state.hand):
            card = self.catalog.cards.get(card_id)
            if not card:
                continue
            candidates.append(("drop", slot, card))
            if not state.must_discard and self._affordable(card, state.me):
                candidates.append(("turn", slot, card))
        if not candidates:
            return []

        started = time.monotonic()
        time_budget, particle_limit, move_timeout = self._analysis_limits(state)
        # effective_compute_deadline = min(
        #     decision_start + mode_compute_cap,
        #     server_deadline - safety_reserve,
        # )
        # _analysis_limits calculates the same two relative caps and clamps an
        # already-expired timer to the emergency best-so-far window.
        deadline = started + time_budget
        hard_stop = deadline - self.DEADLINE_GUARD_SECONDS
        # Stochastic replacement makes one candidate more expensive than in
        # 3.7.1.  Never gamble the whole deadline on an indivisible 200-sample
        # batch: only completed batches may enter best-so-far.
        if time_budget < 0.75:
            batch_size = 2
        elif move_timeout <= 15:
            batch_size = 4
        elif move_timeout <= 30:
            batch_size = 8
        else:
            batch_size = 12
        accumulators: dict[tuple[str, int], dict[str, Any]] = {
            (action, slot): {
                "terminal": False,
                "outcomes": [],
                "lose_flags": [],
                "opponent_two_flags": [],
                "win_two_values": [],
                "reply_card_ids": [],
                "replacement_card_ids": [],
            }
            for action, slot, _card in candidates
        }
        bases: dict[tuple[str, int], CardDecision] = {}
        all_particles: list[tuple[int, ...]] = []
        total_particles = 0
        sampling_batches = 0
        previous_best: tuple[str, int] | None = None
        stable_batches = 0
        confidence_half_width = 1.0
        statistically_stable = False
        deadline_hit = False
        stopping_reason = ""
        particles_requested = 0
        latest_choices: list[CardDecision] = []
        previous_batch_seconds = 0.0
        se_diff: float | None = None
        ci_diff: tuple[float, float] | None = None

        while total_particles < particle_limit:
            now = time.monotonic()
            predicted_batch = max(0.03, previous_batch_seconds * 1.25)
            if now + predicted_batch >= hard_stop:
                deadline_hit = True
                stopping_reason = "deadline_precheck"
                break
            current_batch = min(batch_size, particle_limit - total_particles)
            particles = self.belief.particles(state, current_batch, offset=total_particles)
            particles_requested += current_batch
            batch_started = time.monotonic()
            temporary: dict[tuple[str, int], dict[str, Any]] = {}
            batch_bases: dict[tuple[str, int], CardDecision] = {}
            complete = True
            for action, slot, card in candidates:
                sink: dict[str, Any] = {}
                decision = self._evaluate_choice(
                    action,
                    slot,
                    card,
                    state,
                    particles,
                    unseen_pool,
                    tuple(),
                    sample_sink=sink,
                    deadline=hard_stop,
                    sample_offset=total_particles,
                )
                key = (action, slot)
                temporary[key] = sink
                batch_bases[key] = decision
                if not sink.get("terminal") and len(sink.get("outcomes") or []) != current_batch:
                    complete = False
                    break
            previous_batch_seconds = time.monotonic() - batch_started
            if not complete:
                deadline_hit = True
                stopping_reason = "deadline_in_batch"
                break

            all_particles.extend(particles)
            total_particles += current_batch
            sampling_batches += 1
            for key, sink in temporary.items():
                bases.setdefault(key, batch_bases[key])
                accumulator = accumulators[key]
                accumulator["terminal"] = bool(accumulator.get("terminal") or sink.get("terminal"))
                if sink.get("terminal"):
                    accumulator["terminal_value"] = sink.get("terminal_value", 1.0)
                accumulator["unlock_bonus"] = sink.get("unlock_bonus", accumulator.get("unlock_bonus", 0.0))
                accumulator["retention_loss"] = sink.get("retention_loss", accumulator.get("retention_loss", 0.0))
                accumulator["discard_identity_adjustment"] = sink.get(
                    "discard_identity_adjustment",
                    accumulator.get("discard_identity_adjustment", 0.0),
                )
                accumulator["symmetric_exchange_adjustment"] = sink.get(
                    "symmetric_exchange_adjustment",
                    accumulator.get("symmetric_exchange_adjustment", 0.0),
                )
                for field_name in (
                    "outcomes",
                    "lose_flags",
                    "opponent_two_flags",
                    "win_two_values",
                    "reply_card_ids",
                    "replacement_card_ids",
                ):
                    accumulator[field_name].extend(sink.get(field_name) or [])

            latest_choices = [
                self._aggregate_samples(bases[key], accumulators[key], state, tuple())
                for key in bases
            ]
            latest_choices = self._with_policy_scores(state, latest_choices)
            latest_choices.sort(key=self._choice_sort_key, reverse=True)
            for choice in latest_choices:
                accumulators[(choice.action, choice.slot)]["aggregated_pwin"] = choice.p_win
            best_key = (latest_choices[0].action, latest_choices[0].slot)
            if best_key == previous_best:
                stable_batches += 1
            else:
                stable_batches = 1
                previous_best = best_key

            if len(latest_choices) == 1 or latest_choices[0].immediate_terminal_win:
                statistically_stable = True
                stopping_reason = "terminal"
                break
            if total_particles < self.MIN_PARTICLES:
                continue

            runner_key = (latest_choices[1].action, latest_choices[1].slot)
            margin = (latest_choices[0].p_win or 0.0) - (latest_choices[1].p_win or 0.0)
            top_samples = self._pwin_influence_samples(accumulators[best_key], state)
            runner_samples = self._pwin_influence_samples(accumulators[runner_key], state)
            se_diff, ci_diff, exact_equivalence = self._paired_statistics(
                margin, top_samples, runner_samples
            )
            confidence_half_width = 1.96 * se_diff if se_diff is not None else 1.0
            loss_probability = latest_choices[0].p_lose_next_turn or 0.0
            loss_se = math.sqrt(max(0.0, loss_probability * (1.0 - loss_probability)) / max(1, total_particles))
            confidence_excludes_zero = ci_diff is not None and ci_diff[0] > 0.0
            practical_equivalence = bool(
                ci_diff is not None
                and ci_diff[0] >= -self.PRACTICAL_EQUIVALENCE_EPSILON
                and ci_diff[1] <= self.PRACTICAL_EQUIVALENCE_EPSILON
            )
            if practical_equivalence or (
                exact_equivalence and margin <= self.PRACTICAL_EQUIVALENCE_EPSILON
            ):
                statistically_stable = True
                stopping_reason = "practical_equivalence"
                break
            # Stopping is driven by paired statistical uncertainty, rare-event
            # uncertainty and ranking stability.  Margin alone is insufficient.
            strong_first_batch = margin >= 0.040 and confidence_half_width <= 0.010
            statistically_stable = (
                (stable_batches >= 2 or strong_first_batch)
                and confidence_excludes_zero
                and confidence_half_width <= 0.010
                and loss_se <= 0.015
            )
            if statistically_stable:
                stopping_reason = "statistical_convergence"
                break

        if not stopping_reason and total_particles >= particle_limit:
            stopping_reason = "max_particles"

        if not latest_choices:
            # A nearly expired timer still returns a complete best-so-far
            # ranking.  One shared particle is preferable to timing out.
            particles = self.belief.particles(state, 1)
            particles_requested += 1
            all_particles = particles
            for action, slot, card in candidates:
                key = (action, slot)
                sink: dict[str, Any] = {}
                bases[key] = self._evaluate_choice(
                    action, slot, card, state, particles, unseen_pool, tuple(), sample_sink=sink
                )
                accumulators[key].update(sink)
            total_particles = 1
            sampling_batches = 1
            latest_choices = [
                self._aggregate_samples(bases[key], accumulators[key], state, tuple())
                for key in bases
            ]
            latest_choices = self._with_policy_scores(state, latest_choices)
            latest_choices.sort(key=self._choice_sort_key, reverse=True)
            deadline_hit = True
            stopping_reason = "emergency_fallback"

        top_threats = self._belief_threats(state, all_particles)
        margin = (
            max(0.0, (latest_choices[0].p_win or 0.0) - (latest_choices[1].p_win or 0.0))
            if len(latest_choices) > 1
            else 1.0
        )
        if len(latest_choices) > 1:
            final_top_key = (latest_choices[0].action, latest_choices[0].slot)
            final_runner_key = (latest_choices[1].action, latest_choices[1].slot)
            final_top_samples = self._pwin_influence_samples(accumulators[final_top_key], state)
            final_runner_samples = self._pwin_influence_samples(accumulators[final_runner_key], state)
            se_diff, ci_diff, exact_equivalence = self._paired_statistics(
                margin, final_top_samples, final_runner_samples
            )
            confidence_half_width = 1.96 * se_diff if se_diff is not None else 1.0
            final_loss = latest_choices[0].p_lose_next_turn or 0.0
            final_loss_se = math.sqrt(
                max(0.0, final_loss * (1.0 - final_loss)) / max(1, total_particles)
            )
            final_practical_equivalence = bool(
                ci_diff is not None
                and ci_diff[0] >= -self.PRACTICAL_EQUIVALENCE_EPSILON
                and ci_diff[1] <= self.PRACTICAL_EQUIVALENCE_EPSILON
            )
            statistically_stable = statistically_stable or final_practical_equivalence or (
                ci_diff is not None
                and ci_diff[0] > 0.0
                and confidence_half_width <= 0.010
                and final_loss_se <= 0.025
            )
        else:
            final_practical_equivalence = False

        # A P(win)-equivalent action may be selected by the validated
        # risk/tactical tie-break.  The learned score is deliberately last.
        pwin_order = sorted(latest_choices, key=self._choice_sort_key, reverse=True)
        pwin_best = pwin_order[0]
        equivalent_pool = [pwin_best]
        if not pwin_best.immediate_terminal_win:
            best_key = (pwin_best.action, pwin_best.slot)
            best_samples = self._pwin_influence_samples(accumulators[best_key], state)
            for candidate in pwin_order[1:]:
                candidate_key = (candidate.action, candidate.slot)
                candidate_margin = (pwin_best.p_win or 0.0) - (candidate.p_win or 0.0)
                _candidate_se, candidate_ci, _exact = self._paired_statistics(
                    candidate_margin,
                    best_samples,
                    self._pwin_influence_samples(accumulators[candidate_key], state),
                )
                if (
                    candidate_ci is not None
                    and candidate_ci[0] >= -self.PRACTICAL_EQUIVALENCE_EPSILON
                    and candidate_ci[1] <= self.PRACTICAL_EQUIVALENCE_EPSILON
                ):
                    equivalent_pool.append(candidate)
        practical_equivalence_used = len(equivalent_pool) > 1
        selected = max(equivalent_pool, key=self._practical_tie_key)
        selected_key = (selected.action, selected.slot)

        policy_order = sorted(latest_choices, key=self._policy_diagnostic_sort_key, reverse=True)
        policy_margin = 1.0
        policy_se_diff: float | None = None
        policy_ci_diff: tuple[float, float] | None = None
        if len(policy_order) > 1:
            policy_top, policy_runner = policy_order[:2]
            policy_margin = (policy_top.policy_score or 0.0) - (policy_runner.policy_score or 0.0)
            policy_se_diff, policy_ci_diff, _unused = self._paired_statistics(
                policy_margin,
                self._policy_score_samples(
                    policy_top, accumulators[(policy_top.action, policy_top.slot)], state
                ),
                self._policy_score_samples(
                    policy_runner,
                    accumulators[(policy_runner.action, policy_runner.slot)],
                    state,
                ),
            )
        uncertain = len(latest_choices) > 1 and not statistically_stable
        seed_digest = hashlib.sha256(self.belief.seed_text(state).encode("utf-8")).hexdigest()[:16]
        deadline_remaining = max(0.0, deadline - time.monotonic())
        latest_choices = [
            replace(
                choice,
                opponent_belief_top_threats=top_threats,
                particle_count=total_particles,
                particle_limit=particle_limit,
                particles_requested=particles_requested,
                particles_completed=total_particles,
                stopping_reason=stopping_reason,
                deadline_remaining=deadline_remaining,
                decision_margin=margin,
                se_diff=se_diff,
                ci_diff=ci_diff,
                policy_score_margin=policy_margin,
                policy_score_mc_se_diff=policy_se_diff,
                policy_score_mc_ci_diff=policy_ci_diff,
                stopping_objective="risk-adjusted P(win) with paired common random numbers",
                model_policy_uncertainty="not estimated: learned policy is a deterministic point model",
                confidence_interval_pp=confidence_half_width * 100.0,
                decision_uncertain=uncertain,
                analysis_deadline_hit=deadline_hit,
                sampling_batches=sampling_batches,
                random_seed=seed_digest,
                analysis_time_budget=time_budget,
                hand_diagnostics={
                    **self._hand_quality_diagnostics(state),
                    "belief": self.belief.diagnostics(state, all_particles),
                },
            )
            for choice in latest_choices
        ]
        latest_choices = self._with_diagnostic_ranks(
            latest_choices,
            selected_key=selected_key,
            practical_equivalence=practical_equivalence_used,
        )
        self.last_sampling = {
            "particle_count": total_particles,
            "particles_requested": particles_requested,
            "particles_completed": total_particles,
            "batches": sampling_batches,
            "margin": margin,
            "se_diff": se_diff,
            "ci_diff": ci_diff,
            "policy_score_margin": policy_margin,
            "policy_score_mc_se_diff": policy_se_diff,
            "policy_score_mc_ci_diff": policy_ci_diff,
            "stopping_objective": "risk-adjusted P(win) with paired common random numbers",
            "practical_equivalence_epsilon": self.PRACTICAL_EQUIVALENCE_EPSILON,
            "practical_equivalence_used": practical_equivalence_used,
            "selected_key": selected_key,
            "confidence_half_width": confidence_half_width,
            "uncertain": uncertain,
            "deadline_hit": deadline_hit,
            "stopping_reason": stopping_reason,
            "deadline_remaining": deadline_remaining,
            "time_budget": time_budget,
            "move_timeout": move_timeout,
            "particle_limit": particle_limit,
            "elapsed": time.monotonic() - started,
            "random_seed": seed_digest,
            "belief_diagnostics": self.belief.diagnostics(state, all_particles),
        }
        return latest_choices

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.STRATEGY_VERSION,
            "oos_series_id": self.OOS_SERIES_ID,
            "lookahead_plies": 2,
            "response_weight": 1.0,
            "extra_turn_depth": self.EXTRA_TURN_DEPTH,
            "particle_count": "adaptive",
            "particle_start": self.MIN_PARTICLES,
            "particle_batch": self.PARTICLE_BATCH,
            "particle_max": self.MAX_PARTICLES,
            "particle_limits_by_server_timeout": dict(self.PARTICLE_LIMIT_BY_TIMEOUT),
            "analysis_safety_margin_seconds": self.MOVE_SAFETY_MARGIN,
            "analysis_max_seconds": self.MAX_ANALYSIS_SECONDS,
            "analysis_budget_by_server_timeout": {
                seconds: min(self.MAX_ANALYSIS_SECONDS, seconds - self.MOVE_SAFETY_MARGIN)
                for seconds in self.SERVER_MOVE_TIMEOUTS
            },
            "tail_fraction": self.TAIL_FRACTION,
            "hidden_hand_size": self.HIDDEN_HAND_SIZE,
            "belief_excludes_our_hand": True,
            "belief_uses_observed_history": True,
            "belief_deck_model": "cyclic-cooldown-empirical-age-cdf-weight",
            "belief_cooldown_actions": self.belief.DRAW_COOLDOWN_ACTIONS,
            "belief_return_probability_knots": list(self.belief.RETURN_PROBABILITY_KNOTS),
            "belief_persisted_across_reconnect": True,
            "belief_unknown_reconnect_actions": len(self.belief.unknown_action_indices),
            "belief_gap_model": "particle-resampled-latent-play-discard",
            "common_random_numbers": True,
            "result_scale": "P_win_0_1",
            "selection_objective": [
                "exact_immediate_terminal_win",
                "risk_adjusted_P_win",
                "paired_practical_equivalence_95CI",
                "lower_P_lose_next",
                "lower_tail_risk",
                "better_tactical_horizon",
            "deterministic_stable_order",
            ],
            "stopping_objective": "paired uncertainty of risk-adjusted P(win)",
            "practical_equivalence_epsilon": self.PRACTICAL_EQUIVALENCE_EPSILON,
            "learned_policy_role": "diagnostic-only; never changes production selection",
            "model_policy_uncertainty": "not estimated: deterministic point model",
            "catalog_cards": len(self.catalog.cards),
        }


CardStrategy = ProbabilisticCardStrategy


def parse_last_move(value: str) -> tuple[str, int, int] | None:
    """Return action, card id and zero-based hand slot.

    The number after the dash in the HeroesWM protocol is the position of the
    played card in the player's hand (0..5), not a player number.  The actor is
    therefore determined from the state that existed before the move.
    """
    match = re.fullmatch(r"([td])(\d+)-(\d+)", value or "")
    if not match:
        return None
    return ("drop" if match.group(1) == "d" else "turn", int(match.group(2)), int(match.group(3)))


def parse_game_id(url: str, page_text: str = "") -> int | None:
    query = parse_qs(urlparse(url).query)
    if query.get("gameid"):
        try:
            return int(query["gameid"][0])
        except ValueError:
            pass
    match = re.search(r"cgame\.php\?gameid=(\d+)", page_text, re.I)
    return int(match.group(1)) if match else None


class CardGameRecorder:
    """Lossless per-game telemetry without credentials or session cookies."""

    def __init__(
        self,
        root: Path,
        game_id: int,
        catalog: CardCatalog,
        strategy_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.catalog = catalog
        self.game_id = game_id
        self.started_at = datetime.now()
        stamp = self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        self.games_dir = root / "card_games"
        self.logs_dir = root / "card_logs"
        self.games_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.games_dir / f"{stamp}_game_{game_id}.json"
        self.log_path = self.logs_dir / f"{stamp}_game_{game_id}.txt"
        self.events: list[dict[str, Any]] = []
        self.initial: dict[str, Any] | None = None
        self.last_move_key: tuple[int, str] | None = None
        self.strategy_metadata = dict(strategy_metadata or {})

    @staticmethod
    def _player(player: PlayerState) -> dict[str, int]:
        return asdict(player)

    def _state(self, state: GameState, *, include_raw: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "turn": state.turn,
            "is_your_turn": state.is_your_turn,
            "player_no": state.player_no,
            "time_left": state.time_left,
            "me": self._player(state.me),
            "opponent": self._player(state.opponent),
            "hand": list(state.hand),
            "winner": state.winner,
            "finish_reason": state.finish_reason,
            "last_move": state.last_move,
            "now_player": state.now_player,
            "table": state.table,
            "must_discard": state.must_discard,
        }
        if include_raw:
            result["raw_response"] = state.raw
        return result

    def begin(self, state: GameState) -> None:
        self.initial = self._state(state)
        self._flush()

    def append_log(self, level: str, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now():%H:%M:%S} [{level}] {message}\n")

    def observe(
        self,
        before: GameState,
        after: GameState,
        *,
        selected: CardDecision | None = None,
        rankings: list[CardDecision] | None = None,
    ) -> dict[str, Any] | None:
        parsed = parse_last_move(after.last_move)
        if not parsed:
            return None
        key = (after.turn, after.last_move)
        if key == self.last_move_key:
            return None
        self.last_move_key = key
        action, card_id, hand_slot = parsed
        card = self.catalog.cards.get(card_id)
        actor = "us" if before.is_your_turn else "opponent"
        actor_no = after.player_no if actor == "us" else after.opponent_no
        event: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "turn": after.turn,
            "actor": actor,
            "actor_no": actor_no,
            "hand_slot": hand_slot,
            "action": action,
            "card_id": card_id,
            "card_name": card.name if card else None,
            "card_effect": card.effect if card else None,
            "server_move": after.last_move,
            "before": self._state(before, include_raw=False),
            "after": self._state(after),
        }
        if selected:
            event["selected"] = {
                "slot": selected.slot,
                "score": selected.score,
                "immediate_score": selected.immediate_score,
                "response_value": selected.response_value,
                "predicted_me": selected.predicted_me,
                "predicted_opponent": selected.predicted_opponent,
                "winning_replies_before": selected.winning_replies_before,
                "winning_replies_after": selected.winning_replies_after,
                "discard_retention": selected.discard_retention,
                "legal_cost": selected.legal_cost,
                "immediate_state_delta": selected.immediate_state_delta,
                "p_win": selected.p_win,
                "p_win_next_action": selected.p_win_next_action,
                "p_lose_next_turn": selected.p_lose_next_turn,
                "p_win_within_2_own_actions": selected.p_win_within_2_own_actions,
                "p_opponent_win_within_2_actions": selected.p_opponent_win_within_2_actions,
                "expected_reply_value": selected.expected_reply_value,
                "tail_risk": selected.tail_risk,
                "immediate_terminal_win": selected.immediate_terminal_win,
                "policy_score": selected.policy_score,
                "policy_rank": selected.policy_rank,
                "pwin_rank": selected.pwin_rank,
                "final_rank_reason": selected.final_rank_reason,
                "resource_unlocks": list(selected.resource_unlocks),
                "cards_unlocked_next_turn": list(selected.cards_unlocked_next_turn),
                "eta_key_hand_cards": selected.eta_key_hand_cards,
                "extra_turn_continuation": selected.extra_turn_continuation,
                "opponent_belief_top_threats": list(selected.opponent_belief_top_threats),
                "particle_count": selected.particle_count,
                "particle_limit": selected.particle_limit,
                "particles_requested": selected.particles_requested,
                "particles_completed": selected.particles_completed,
                "stopping_reason": selected.stopping_reason,
                "deadline_remaining": selected.deadline_remaining,
                "decision_margin": selected.decision_margin,
                "se_diff": selected.se_diff,
                "ci_diff": list(selected.ci_diff) if selected.ci_diff is not None else None,
                "policy_score_margin": selected.policy_score_margin,
                "policy_score_mc_se_diff": selected.policy_score_mc_se_diff,
                "policy_score_mc_ci_diff": list(selected.policy_score_mc_ci_diff)
                if selected.policy_score_mc_ci_diff is not None
                else None,
                "stopping_objective": selected.stopping_objective,
                "model_policy_uncertainty": selected.model_policy_uncertainty,
                "confidence_interval_pp": selected.confidence_interval_pp,
                "decision_uncertain": selected.decision_uncertain,
                "analysis_deadline_hit": selected.analysis_deadline_hit,
                "sampling_batches": selected.sampling_batches,
                "random_seed": selected.random_seed,
                "analysis_seconds": selected.analysis_seconds,
                "analysis_time_budget": selected.analysis_time_budget,
                "replacement_distribution": selected.replacement_distribution,
                "hand_diagnostics": selected.hand_diagnostics,
                "reasons": list(selected.reasons),
            }
        if rankings:
            event["candidates"] = [
                {
                    "action": choice.action,
                    "slot": choice.slot,
                    "card_id": choice.card.id,
                    "card_name": choice.card.name,
                    "score": round(choice.score, 4),
                    "immediate_score": None if choice.immediate_score is None else round(choice.immediate_score, 4),
                    "response_value": None if choice.response_value is None else round(choice.response_value, 4),
                    "predicted_me": choice.predicted_me,
                    "predicted_opponent": choice.predicted_opponent,
                    "winning_replies_before": choice.winning_replies_before,
                    "winning_replies_after": choice.winning_replies_after,
                    "discard_retention": round(choice.discard_retention, 4),
                    "legal_cost": choice.legal_cost,
                    "immediate_state_delta": choice.immediate_state_delta,
                    "p_win": choice.p_win,
                    "p_win_next_action": choice.p_win_next_action,
                    "p_lose_next_turn": choice.p_lose_next_turn,
                    "p_win_within_2_own_actions": choice.p_win_within_2_own_actions,
                    "p_opponent_win_within_2_actions": choice.p_opponent_win_within_2_actions,
                    "expected_reply_value": choice.expected_reply_value,
                    "tail_risk": choice.tail_risk,
                    "immediate_terminal_win": choice.immediate_terminal_win,
                    "policy_score": choice.policy_score,
                    "policy_rank": choice.policy_rank,
                    "pwin_rank": choice.pwin_rank,
                    "final_rank_reason": choice.final_rank_reason,
                    "resource_unlocks": list(choice.resource_unlocks),
                    "cards_unlocked_next_turn": list(choice.cards_unlocked_next_turn),
                    "eta_key_hand_cards": choice.eta_key_hand_cards,
                    "extra_turn_continuation": choice.extra_turn_continuation,
                    "opponent_belief_top_threats": list(choice.opponent_belief_top_threats),
                    "particle_count": choice.particle_count,
                    "particle_limit": choice.particle_limit,
                    "particles_requested": choice.particles_requested,
                    "particles_completed": choice.particles_completed,
                    "stopping_reason": choice.stopping_reason,
                    "deadline_remaining": choice.deadline_remaining,
                    "decision_margin": choice.decision_margin,
                    "se_diff": choice.se_diff,
                    "ci_diff": list(choice.ci_diff) if choice.ci_diff is not None else None,
                    "policy_score_margin": choice.policy_score_margin,
                    "policy_score_mc_se_diff": choice.policy_score_mc_se_diff,
                    "policy_score_mc_ci_diff": list(choice.policy_score_mc_ci_diff)
                    if choice.policy_score_mc_ci_diff is not None
                    else None,
                    "stopping_objective": choice.stopping_objective,
                    "model_policy_uncertainty": choice.model_policy_uncertainty,
                    "confidence_interval_pp": choice.confidence_interval_pp,
                    "decision_uncertain": choice.decision_uncertain,
                    "analysis_deadline_hit": choice.analysis_deadline_hit,
                    "sampling_batches": choice.sampling_batches,
                    "random_seed": choice.random_seed,
                    "analysis_seconds": choice.analysis_seconds,
                    "analysis_time_budget": choice.analysis_time_budget,
                    "replacement_distribution": choice.replacement_distribution,
                    "hand_diagnostics": choice.hand_diagnostics,
                    "reasons": list(choice.reasons),
                }
                for choice in rankings
            ]
        self.events.append(event)
        self._flush()
        return event

    def finish(self, state: GameState, won: bool) -> None:
        self.finished_at = datetime.now()
        self.final = self._state(state)
        self.won = won
        self._flush()
        summary_path = self.root / "card_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            summary = {"version": 1, "games": 0, "wins": 0, "losses": 0, "draws": 0}
        summary["games"] = int(summary.get("games", 0)) + 1
        result_key = "wins" if won else "draws" if state.winner == 0 else "losses"
        summary[result_key] = int(summary.get(result_key, 0)) + 1
        summary["last_game_id"] = self.game_id
        summary["updated_at"] = datetime.now().isoformat(timespec="seconds")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _flush(self) -> None:
        payload: dict[str, Any] = {
            "version": 3,
            "game_id": self.game_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "strategy": self.strategy_metadata,
            "initial": self.initial,
            "events": self.events,
        }
        if hasattr(self, "finished_at"):
            payload.update(
                finished_at=self.finished_at.isoformat(timespec="seconds"),
                won=self.won,
                final=self.final,
            )
        temporary = self.json_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.json_path)


class CardGameBot:
    def __init__(
        self,
        session: requests.Session,
        *,
        login_name: str,
        ensure_login: Callable[[], None],
        stop_event: threading.Event | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        state_callback: Callable[[GameState], None] | None = None,
        timeout: int = 40,
        deck_type: int = 1,
        continuous: bool = False,
        max_stake: int = 0,
        work_pause_callback: Callable[[], bool] | None = None,
        learner_path: Path | None = None,
        data_root: Path | None = None,
        catalog: CardCatalog | None = None,
    ) -> None:
        self.session = session
        self.login_name = login_name
        self.ensure_login = ensure_login
        self.stop_event = stop_event or threading.Event()
        self.log_callback = log_callback
        self.state_callback = state_callback
        self.timeout = timeout if timeout in {15, 30, 40} else 40
        self.deck_type = deck_type if deck_type in {1, 2} else 1
        self.continuous = continuous
        self.max_stake = max(0, int(max_stake))
        self.work_pause_callback = work_pause_callback
        self.catalog = catalog or CardCatalog.load()
        self.data_root = data_root or Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HeroesWMWorker"
        learner_file = learner_path or self.data_root / "card_learning.json"
        self.learner = CardGameLearner(learner_file)
        self.strategy = CardStrategy(
            self.catalog,
            self.learner,
            belief_state_path=self.data_root / "card_belief_state.json",
            history_root=self.data_root / "card_games",
        )
        if hasattr(self.strategy, "configure_move_timeout"):
            self.strategy.configure_move_timeout(self.timeout)
        self.actions: list[tuple[str, int]] = []
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.current_game_id: int | None = None
        self.recorder: CardGameRecorder | None = None
        self._work_pause_announced = False
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.9"})

    def emit(self, level: str, message: str) -> None:
        if self.recorder:
            try:
                self.recorder.append_log(level, message)
            except OSError:
                pass
        if self.log_callback:
            self.log_callback(level, message)

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise CardGameStopped()

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0, seconds)
        while time.monotonic() < deadline:
            self._check_stop()
            self.stop_event.wait(min(0.5, deadline - time.monotonic()))

    @staticmethod
    def _move_delay(time_left: int, rng: Callable[[float, float], float] = random.uniform) -> float:
        """Human-like 3-10 second pause while preserving a timer safety margin."""
        available = max(0.0, float(time_left) - 2.5)
        if available >= 3.0:
            return min(rng(3.0, 10.0), available)
        return max(0.15, available)

    @staticmethod
    def _expired(response: requests.Response) -> bool:
        low = response.url.lower()
        return "login.php" in low or "auth_error" in low

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        self._check_stop()
        # Matchmaking is polled frequently.  A shorter read timeout keeps the
        # Stop button responsive when HeroesWM leaves a request hanging.
        response = self.session.get(url, timeout=(5, 10), **kwargs)
        response.raise_for_status()
        if self._expired(response):
            raise CardGameSessionExpired()
        return response

    def _work_pause_requested(self) -> bool:
        if self.work_pause_callback is None:
            return False
        try:
            return bool(self.work_pause_callback())
        except Exception as exc:
            self.emit("warning", f"Не удалось проверить срок устройства на работу: {exc}")
            return False

    @staticmethod
    def _offer_stake(invitation: Any) -> int | None:
        href = str(invitation.get("href") or "")
        query = parse_qs(urlparse(href).query)
        for key in ("bet", "stake", "rate", "gold", "sum"):
            value = query.get(key)
            if value and str(value[0]).isdigit():
                return int(value[0])
        row = invitation.find_parent("tr")
        text = row.get_text(" ", strip=True) if row else ""
        match = re.search(r"ставк\w*\s*[:\-]?\s*(\d+)", text, re.I)
        if match:
            return int(match.group(1))
        if row is None:
            return None

        # The current tavern table often shows only a gold icon and an amount,
        # without the literal word "ставка".  Read the numeric value
        # from that icon's cell instead of treating the offer as unknown.
        for image in row.find_all("img"):
            image_hint = " ".join(
                str(image.get(attribute) or "")
                for attribute in ("src", "alt", "title", "class")
            )
            if not re.search(r"gold|золот", image_hint, re.I):
                continue
            cell = image.find_parent("td")
            cell_text = cell.get_text(" ", strip=True) if cell else ""
            amount = re.search(r"\b(\d[\d\s]*)\b", cell_text)
            if amount:
                return int(re.sub(r"\s+", "", amount.group(1)))

        gold_text = re.search(r"\b(\d[\d\s]*)\s*(?:золот\w*|gold)\b", text, re.I)
        return int(re.sub(r"\s+", "", gold_text.group(1))) if gold_text else None

    def _create_game_request(self, page_url: str, page_text: str) -> tuple[str, dict[str, str], int]:
        """Read the live tavern form and select a stake within the user's cap."""
        soup = BeautifulSoup(page_text, "html.parser")
        form = soup.find("form", action=re.compile(r"create_card_game\.php", re.I))
        if form is None:
            # Historical form: omitting the bet field created a zero-stake game.
            return (
                BASE_URL + "/create_card_game.php",
                {"timeout": str(self.timeout), "ktype": str(self.deck_type)},
                0,
            )

        params: dict[str, str] = {}
        for field in form.find_all("input"):
            name = str(field.get("name") or "")
            field_type = str(field.get("type") or "text").lower()
            if not name or field_type in {"submit", "button", "image", "file"}:
                continue
            if field_type in {"checkbox", "radio"} and not field.has_attr("checked"):
                continue
            params[name] = str(field.get("value") or "")

        stake_name: str | None = None
        # HeroesWM currently submits an internal code (gold=1..5), while the
        # visible option text contains the real wager (40..1000 gold).
        stake_choices: list[tuple[int, str]] = []
        for select in form.find_all("select"):
            name = str(select.get("name") or "")
            if not name:
                continue
            options = [
                (str(option.get("value") if option.has_attr("value") else option.get_text(strip=True)), option)
                for option in select.find_all("option")
            ]
            values = [value for value, _option in options]
            low_name = name.lower()
            context = select.parent.get_text(" ", strip=True).lower() if select.parent else ""
            if "timeout" in low_name or "время" in context:
                params[name] = str(self.timeout) if str(self.timeout) in values else values[0]
            elif low_name in {"ktype", "deck", "deck_type"} or "колод" in context:
                params[name] = str(self.deck_type) if str(self.deck_type) in values else values[0]
            elif re.search(r"bet|stake|rate|gold|sum|stav", low_name) or "ставк" in context:
                stake_name = name
                for submitted_value, option in options:
                    option_text = option.get_text(" ", strip=True)
                    visible_numbers = re.findall(r"\d[\d\s]*", option_text)
                    if visible_numbers:
                        real_stake = int(re.sub(r"\s+", "", visible_numbers[0]))
                    elif re.fullmatch(r"\d+", submitted_value):
                        real_stake = int(submitted_value)
                    else:
                        continue
                    stake_choices.append((real_stake, submitted_value))
            else:
                selected = next((value for value, option in options if option.has_attr("selected")), values[0])
                params[name] = selected

        if stake_name is None:
            chosen_stake = 0
        else:
            stake_choices = sorted(set(stake_choices), key=lambda item: (item[0], item[1]))
            available_stakes = sorted({stake for stake, _value in stake_choices})
            self.emit(
                "info",
                "Доступные ставки сервера: "
                + (", ".join(map(str, available_stakes)) if available_stakes else "не распознаны"),
            )
            allowed = [choice for choice in stake_choices if choice[0] <= self.max_stake]
            if not allowed:
                minimum = min(available_stakes) if available_stakes else "неизвестна"
                raise CardGameStakeUnavailable(
                    f"Бесплатной ставки в форме нет; минимум {minimum}, "
                    f"разрешённый максимум {self.max_stake}"
                )
            # Treat the UI value as a ceiling and choose the strongest exact
            # server option within it.  For example, a 40-gold cap submits
            # gold=1 but is logged and checked as the real 40-gold wager.
            chosen_stake, submitted_stake = max(allowed, key=lambda item: item[0])
            params[stake_name] = submitted_stake

        action = urljoin(page_url, str(form.get("action") or "/create_card_game.php"))
        return action, params, chosen_stake

    def _find_or_create_game(self) -> int:
        announced_wait = False
        accepted_opponent = False
        while True:
            self._check_stop()
            response = self._get(BASE_URL + "/tavern.php")
            text = decode_response(response)
            game_id = parse_game_id(response.url, text)
            if game_id:
                return game_id
            soup = BeautifulSoup(text, "html.parser")

            cancel_link = soup.find("a", href=re.compile(r"^cancel_card_game\.php", re.I))
            has_own_application = cancel_link is not None
            if self._work_pause_requested():
                if has_own_application:
                    self._get(urljoin(response.url, cancel_link.get("href")), headers={"Referer": response.url})
                    self.emit("info", "Заявка карточной игры отменена: освобождаю персонажа для устройства на работу")
                elif not self._work_pause_announced:
                    self.emit("info", "Карточные игры на паузе: приближается попытка устройства на работу")
                self._work_pause_announced = True
                announced_wait = False
                self._sleep(2)
                continue
            if self._work_pause_announced:
                self.emit("info", "Устройство завершено или отложено; возобновляю поиск карточной игры")
                self._work_pause_announced = False

            # A host does not choose another wager here.  Once our room is
            # published, acard_game.php is the server's "accept opponent"
            # action for that already-priced room.  Requiring a stake next to
            # this link made the bot ignore a real opponent indefinitely.
            invitation = soup.find(
                "a", href=re.compile(r"^acard_game\.php\?id=\d+", re.I)
            ) if has_own_application and not accepted_opponent else None
            if invitation is not None:
                opponent = invitation.find_previous("b")
                opponent_text = opponent.get_text(" ", strip=True) if opponent else "соперник"
                self.emit("info", f"Соперник подключился к нашей комнате: {opponent_text}. Принимаю бой")
                accepted = self._get(urljoin(response.url, invitation.get("href")), headers={"Referer": response.url})
                accepted_text = decode_response(accepted)
                game_id = parse_game_id(accepted.url, accepted_text)
                if game_id:
                    return game_id
                accepted_opponent = True
                self.emit("info", "Бой принят; жду переход сервера в игру")
                announced_wait = True
                self._sleep(1)
                continue

            if accepted_opponent:
                # Do not publish a second room during the short transition
                # between accepting the opponent and receiving cgame.php.
                self._sleep(1)
                continue

            if not has_own_application:
                form_page = self._get(BASE_URL + "/tavern.php?form=1", headers={"Referer": response.url})
                form_text = decode_response(form_page)
                action_url, params, chosen_stake = self._create_game_request(form_page.url, form_text)
                self.emit("info", f"Создаю заявку: ставка {chosen_stake}, ход {self.timeout} сек., " + ("одна колода" if self.deck_type == 1 else "бесконечная колода"))
                created = self._get(
                    action_url,
                    params=params,
                    headers={"Referer": form_page.url},
                )
                created_text = decode_response(created)
                game_id = parse_game_id(created.url, created_text)
                if game_id:
                    return game_id
                announced_wait = False
            elif not announced_wait:
                self.emit("info", "Заявка опубликована; жду соперника")
                announced_wait = True
            self._sleep(2)

    def _request_state(self, protocol: CardGameProtocol, *, first: bool = False) -> GameState:
        params: dict[str, str] = {
            "html": "1",
            "gameid": str(protocol.game_id),
            "lchatid": "0",
            "rand": str(random.random()),
        }
        if first:
            params["action"] = "getnicks"
            player_id = self.session.cookies.get("pl_id")
            if player_id:
                params["pl_id"] = player_id
        response = self._get(CARD_ENDPOINT, params=params, headers={"Referer": BASE_URL + f"/cgame.php?gameid={protocol.game_id}"})
        payload = decode_response(response)
        if "<html" in payload.lower():
            raise CardGameSessionExpired()
        return protocol.parse(payload)

    def _send_decision(self, protocol: CardGameProtocol, state: GameState, decision: CardDecision) -> GameState:
        params = {
            "html": "1",
            "gameid": str(protocol.game_id),
            "lchatid": "0",
            "action": decision.action,
            "cardid": str(decision.card.id),
            "cardn": str(decision.slot),
            "turn": str(state.turn),
            "rand2": f"{random.uniform(15, 180):.6f}",
            "rand": str(random.random()),
        }
        response = self._get(CARD_ENDPOINT, params=params, headers={"Referer": BASE_URL + f"/cgame.php?gameid={protocol.game_id}"})
        self.actions.append((decision.action, decision.card.id))
        return protocol.parse(decode_response(response))

    @staticmethod
    def _finish_text(reason: int) -> str:
        return {
            1: "огромная башня",
            2: "башня противника разрушена",
            3: "накоплены все ресурсы",
            4: "время игрока истекло",
            5: "общий таймаут игры",
            6: "одновременное достижение условия",
        }.get(reason, f"причина {reason}")

    @staticmethod
    def _state_delta(before: GameState, after: GameState) -> str:
        values: list[str] = []
        pairs = (
            ("наша башня", before.me.tower, after.me.tower),
            ("наша стена", before.me.wall, after.me.wall),
            ("башня соперника", before.opponent.tower, after.opponent.tower),
            ("стена соперника", before.opponent.wall, after.opponent.wall),
            ("наша шахта", before.me.mine, after.me.mine),
            ("наш монастырь", before.me.monastery, after.me.monastery),
            ("наши казармы", before.me.barracks, after.me.barracks),
            ("шахта соперника", before.opponent.mine, after.opponent.mine),
            ("монастырь соперника", before.opponent.monastery, after.opponent.monastery),
            ("казармы соперника", before.opponent.barracks, after.opponent.barracks),
        )
        for label, old, new in pairs:
            if new != old:
                values.append(f"{label} {new-old:+}")
        return ", ".join(values) if values else "башни/стены/производства без изменений"

    def _emit_move(self, event: dict[str, Any], before: GameState, after: GameState) -> None:
        side = "МЫ" if event["actor"] == "us" else "СОПЕРНИК"
        verb = "СБРОС" if event["action"] == "drop" else "КАРТА"
        name = event["card_name"] or f"ID {event['card_id']}"
        self.emit("info", f"Ход {before.turn} — {side}: {verb} «{name}»")
        if event.get("card_effect"):
            self.emit("info", f"  Эффект: {event['card_effect']}")
        self.emit("info", f"  Итог: {self._state_delta(before, after)}")

    def _play(self, game_id: int) -> bool:
        self.current_game_id = game_id
        self.actions = []
        if hasattr(self.strategy, "reset_game"):
            self.strategy.reset_game(game_id)
        protocol = CardGameProtocol(game_id)
        state = self._request_state(protocol, first=True)
        resync: dict[str, Any] = {}
        if hasattr(self.strategy, "synchronize_state"):
            resync = self.strategy.synchronize_state(state)
        metadata = self.strategy.metadata() if hasattr(self.strategy, "metadata") else {}
        if resync:
            metadata["initial_resync"] = resync
        self.recorder = CardGameRecorder(self.data_root, game_id, self.catalog, metadata)
        self.recorder.begin(state)
        names = state.nicknames
        self.emit("info", f"Игра #{game_id}: {names.get(1, 'Игрок 1')} — {names.get(2, 'Игрок 2')}")
        self.emit("info", f"Условия победы: башня {state.tower_goal}, каждый ресурс {state.resource_goal}")
        if int(resync.get("missing_actions") or 0) > 0:
            self.emit(
                "warning",
                f"Reconnect/resync: пропущено {resync['missing_actions']} карточных действий "
                f"({resync['from_action']}→{resync['to_action']}); "
                "неизвестные PLAY/DISCARD будут пересэмплированы отдельно в каждой частице",
            )
        last_logged_turn = -1
        work_pause_announced = False

        while not state.finished:
            self._check_stop()
            if self._work_pause_requested() and not work_pause_announced:
                self.emit(
                    "warning",
                    "Приближается устройство на работу: текущую партию завершу, следующую заявку не создам",
                )
                work_pause_announced = True
            if self.state_callback:
                self.state_callback(state)
            if state.turn != last_logged_turn:
                me, enemy = state.me, state.opponent
                side = "МЫ ДУМАЕМ" if state.is_your_turn else "ЖДЁМ СОПЕРНИКА"
                self.emit(
                    "info",
                    f"──── Ход {state.turn} — {side} ────\n"
                    f"Позиция: мы — башня {me.tower}, стена {me.wall}, "
                    f"ресурсы {me.ore}/{me.mana}/{me.army}; противник — "
                    f"башня {enemy.tower}, стена {enemy.wall}, ресурсы {enemy.ore}/{enemy.mana}/{enemy.army}",
                )
                last_logged_turn = state.turn

            if state.is_your_turn:
                analysis_started = time.monotonic()
                rankings = self.strategy.rank_choices(state)
                analysis_seconds = time.monotonic() - analysis_started
                rankings = [replace(choice, analysis_seconds=analysis_seconds) for choice in rankings]
                decision = rankings[0]
                hand_text = ", ".join(
                    f"{slot + 1}:{self.catalog.cards[card_id].name}"
                    for slot, card_id in enumerate(state.hand)
                    if card_id in self.catalog.cards
                )
                self.emit("info", f"Рука: {hand_text}")
                verb = "Разыгрываю" if decision.action == "turn" else "Сбрасываю"
                reason = ", ".join(decision.reasons[:3])
                self.emit(
                    "info",
                    f"План: {verb.lower()} «{decision.card.name}» — {reason} "
                    f"(P(win) {(decision.p_win or 0.0):.2%}, rank #{decision.pwin_rank}; "
                    f"policy_score {(decision.policy_score or 0.0):.4f}, rank #{decision.policy_rank})",
                )
                if decision.replacement_distribution:
                    replacement_text = ", ".join(
                        f"{name} {probability:.1%}"
                        for name, probability in decision.replacement_distribution.items()
                    )
                    self.emit("info", "Replacement draw: " + replacement_text)
                if decision.hand_diagnostics:
                    hand_diag = decision.hand_diagnostics
                    self.emit(
                        "info",
                        "Качество руки: "
                        f"playable {hand_diag.get('hand_playable_count')}, "
                        f"strong {hand_diag.get('hand_strong_playable_count')}, "
                        f"dead {hand_diag.get('hand_dead_count')}, "
                        f"congestion {float(hand_diag.get('hand_resource_congestion') or 0.0):.1%}",
                    )
                alternatives = []
                for choice in rankings[:7]:
                    action = "карта" if choice.action == "turn" else "сброс"
                    alternatives.append(
                        f"{action} «{choice.card.name}»: "
                        f"P(win) {(choice.p_win or 0.0):.2%} (#{choice.pwin_rank}), "
                        f"policy {float(choice.policy_score or 0.0):.4f} (#{choice.policy_rank}), "
                        f"проигрыш следующим ходом {(choice.p_lose_next_turn or 0.0):.1%}, "
                        f"наш финиш ≤2 действий {(choice.p_win_within_2_own_actions or 0.0):.1%}, "
                        f"хвост {(choice.tail_risk or 0.0):.1%}"
                    )
                self.emit("info", "Лучшие варианты: " + "; ".join(alternatives))
                if len(rankings) > 1:
                    runner_up = rankings[1]
                    pwin_delta = (decision.p_win or 0.0) - (runner_up.p_win or 0.0)
                    self.emit(
                        "info",
                        f"Финальный selector: {decision.final_rank_reason}. "
                        f"P(win) {decision.card.name} {(decision.p_win or 0.0):.2%} vs "
                        f"{runner_up.card.name} {(runner_up.p_win or 0.0):.2%} "
                        f"(delta {pwin_delta * 100:+.2f} п.п.); "
                        f"policy_score {float(decision.policy_score or 0.0):.4f} vs "
                        f"{float(runner_up.policy_score or 0.0):.4f}; "
                        f"P(lose next) {(decision.p_lose_next_turn or 0.0):.1%} vs "
                        f"{(runner_up.p_lose_next_turn or 0.0):.1%}; tail "
                        f"{(decision.tail_risk or 0.0):.1%} vs {(runner_up.tail_risk or 0.0):.1%}",
                    )
                uncertainty_text = (
                    f"±{(decision.confidence_interval_pp or 0.0):.2f} п.п."
                    if decision.confidence_interval_pp is not None
                    else "без оценки интервала"
                )
                sampling_label = "DECISION UNCERTAIN" if decision.decision_uncertain else "ranking устойчив"
                deadline_label = "; достигнут лимит времени" if decision.analysis_deadline_hit else ""
                ci_text = (
                    f"[{decision.ci_diff[0] * 100:+.3f}, "
                    f"{decision.ci_diff[1] * 100:+.3f}] п.п."
                    if decision.ci_diff is not None
                    else "недоступен"
                )
                self.emit(
                    "warning" if decision.decision_uncertain else "info",
                    f"Monte Carlo: requested/completed "
                    f"{decision.particles_requested}/{decision.particles_completed}, "
                    f"cap {decision.particle_limit}, "
                    f"{decision.sampling_batches} полных пачек; stop={decision.stopping_reason}; "
                    f"objective=P(win); top gap {(decision.decision_margin or 0.0) * 100:.2f} п.п., "
                    f"SE(diff) {(decision.se_diff or 0.0) * 100:.3f} п.п., "
                    f"95% CI(diff) {ci_text}, "
                    f"policy margin {float(decision.policy_score_margin or 0.0):.4f}, "
                    f"policy MC-SE {float(decision.policy_score_mc_se_diff or 0.0):.4f}; "
                    f"model uncertainty: {decision.model_policy_uncertainty}; "
                    f"бюджет анализа {(decision.analysis_time_budget or 0.0):.1f} сек. "
                    f"из серверных {self.timeout} сек., "
                    f"до deadline {(decision.deadline_remaining or 0.0):.2f} сек.; "
                    f"{sampling_label} {uncertainty_text}{deadline_label}",
                )
                if decision.opponent_belief_top_threats:
                    self.emit(
                        "info",
                        "Вероятные угрозы в скрытой руке: "
                        + "; ".join(decision.opponent_belief_top_threats),
                    )
                total_think_target = self._move_delay(state.time_left)
                delay = max(0.0, total_think_target - analysis_seconds)
                self.emit(
                    "info",
                    f"Анализ {analysis_seconds:.1f} сек., дополнительная пауза {delay:.1f} сек. "
                    f"(на таймере было {state.time_left} сек.)",
                )
                self._sleep(delay)
                fresh = self._request_state(protocol)
                if fresh.finished:
                    event = self.recorder.observe(state, fresh)
                    if hasattr(self.strategy, "observe_transition"):
                        self.strategy.observe_transition(state, fresh)
                    if event:
                        self._emit_move(event, state, fresh)
                    state = fresh
                elif fresh.turn != state.turn or not fresh.is_your_turn or fresh.must_discard != state.must_discard:
                    event = self.recorder.observe(state, fresh)
                    if hasattr(self.strategy, "observe_transition"):
                        self.strategy.observe_transition(state, fresh)
                    if event:
                        self._emit_move(event, state, fresh)
                    else:
                        self.emit("warning", f"Ход {state.turn} изменился до отправки; наш запланированный ход отменён")
                    state = fresh
                else:
                    after = self._send_decision(protocol, fresh, decision)
                    event = self.recorder.observe(fresh, after, selected=decision, rankings=rankings)
                    if hasattr(self.strategy, "observe_transition"):
                        self.strategy.observe_transition(fresh, after)
                    if event:
                        self._emit_move(event, fresh, after)
                    else:
                        self.emit("warning", f"Ход {fresh.turn} — МЫ: сервер не вернул код применённой карты; сырой ответ сохранён")
                    state = after
            else:
                self._sleep(min(3.0, max(1.0, state.time_left / 8)))
                after = self._request_state(protocol)
                event = self.recorder.observe(state, after)
                if hasattr(self.strategy, "observe_transition"):
                    self.strategy.observe_transition(state, after)
                if event:
                    self._emit_move(event, state, after)
                elif after.turn != state.turn:
                    self.emit("warning", f"Ход {state.turn} — СОПЕРНИК: карта не указана сервером (таймаут/пропуск); состояние сохранено")
                state = after

        if self.state_callback:
            self.state_callback(state)
        won = state.winner == state.player_no
        self.games_played += 1
        self.wins += int(won)
        self.losses += int(not won and state.winner != 0)
        self.learner.record(self.actions, won)
        self.recorder.finish(state, won)
        result = "ПОБЕДА" if won else "НИЧЬЯ" if state.winner == 0 else "ПОРАЖЕНИЕ"
        self.emit("info" if won else "warning", f"{result}: {self._finish_text(state.finish_reason)}. Счёт с запуска {self.wins}:{self.losses}")
        self.emit("info", f"Запись партии: {self.recorder.json_path}")
        self.recorder = None
        return won

    def run(self) -> None:
        self.emit("info", "Автоигра в карточные баталии запущена")
        self.emit("info", f"Каталог загружен: {len(self.catalog.cards)} карты; максимальная ставка {self.max_stake}")
        try:
            while True:
                self._check_stop()
                try:
                    self.ensure_login()
                    game_id = self._find_or_create_game()
                    self._play(game_id)
                    if not self.continuous:
                        break
                    self.emit("info", "Следующую бесплатную игру начну через 8–15 сек")
                    self._sleep(random.uniform(8, 15))
                except CardGameSessionExpired:
                    self.emit("warning", "Сессия карточной игры истекла; выполняю повторный вход")
                    self.ensure_login()
                except CardGameStakeUnavailable as exc:
                    self.emit("warning", f"{exc}; повторная проверка формы через 30 сек")
                    self._sleep(30)
                except requests.RequestException as exc:
                    self.emit("warning", f"Ошибка сети: {exc}; повтор через 10 сек")
                    self._sleep(10)
        except CardGameStopped:
            self.emit("info", "Автоигра остановлена")
        finally:
            self.current_game_id = None
