from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from train_opponent_policy import feature_vector as opponent_features
from train_state_value import feature_vector as value_features


def load_module(source: Path):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("policy39_stage_a_card_game", source / "card_game.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load card_game.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dataset_rows(path: Path, splits: set[str]):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            # Split is near the front of every canonical line.  Parsing is fine
            # for selected splits; holdout is deliberately skipped by caller.
            row = json.loads(line)
            if row.get("split") in splits:
                yield row


def player(module: Any, raw: dict[str, Any]):
    return module.PlayerState(**{
        name: int(raw.get(name) or 0)
        for name in ("ore", "mana", "army", "tower", "wall", "mine", "monastery", "barracks")
    })


def state_from_row(module: Any, row: dict[str, Any], hand: list[int] | None = None, me: Any = None, enemy: Any = None):
    raw = row["visible_state"]
    me = me or player(module, raw["me"])
    enemy = enemy or player(module, raw["opponent"])
    return module.GameState(
        game_id=int(row["game_id"]), turn=int(raw.get("turn") or 0), is_your_turn=True,
        player_no=1, time_left=40, players={1: me, 2: enemy},
        hand=list(row["our_hand"] if hand is None else hand), winner=0, finish_reason=0,
        last_move=str(raw.get("last_move") or ""), now_player=1,
        table=str(raw.get("table") or ""), must_discard=bool(raw.get("must_discard")),
        first_actor=str(row.get("initiative") or "unknown"),
        reconnect_uncertainty=bool(row.get("reconnect")),
        unknown_transitions=int(row.get("reconnect_unknown_transitions_before") or 0),
    )


def row_for_position(row: dict[str, Any], state: Any, strategy: Any) -> dict[str, Any]:
    raw = {
        "turn": state.turn, "is_your_turn": True, "player_no": 1,
        "time_left": 40, "me": asdict(state.me), "opponent": asdict(state.opponent),
        "hand": list(state.hand), "winner": 0, "finish_reason": 0,
        "last_move": state.last_move, "now_player": 1, "table": state.table,
        "must_discard": state.must_discard,
    }
    legal = []
    for slot, card_id in enumerate(state.hand):
        legal.append({"action": "drop", "slot": slot, "card_id": card_id})
        card = strategy.catalog.cards.get(card_id)
        if card and not state.must_discard and strategy._affordable(card, state.me):
            legal.append({"action": "turn", "slot": slot, "card_id": card_id})
    result = dict(row)
    result["visible_state"] = raw
    result["our_hand"] = list(state.hand)
    result["legal_actions"] = legal
    return result


class Models:
    def __init__(self, catalog_raw: dict[int, dict[str, Any]], horizon: dict[str, Any], value: dict[str, Any], opponent: dict[str, Any]):
        self.catalog_raw = catalog_raw
        self.horizon = horizon
        self.value = value
        self.opponent = opponent
        self.value_center = np.asarray(value["center"])
        self.value_scale = np.asarray(value["scale"])
        self.value_beta = np.asarray(value["coefficients_with_intercept"])
        self.opponent_beta = np.asarray(opponent["weights"])

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-max(-30, min(30, x))))

    def pwin(self, row: dict[str, Any], state: Any, strategy: Any) -> float:
        if state.me.tower >= 50 or state.opponent.tower <= 0 or min(state.me.ore, state.me.mana, state.me.army) >= 150:
            return 1.0
        if state.me.tower <= 0 or state.opponent.tower >= 50 or min(state.opponent.ore, state.opponent.mana, state.opponent.army) >= 150:
            return 0.0
        values = np.asarray(value_features(row_for_position(row, state, strategy), self.catalog_raw, self.horizon))
        z = (values - self.value_center) / self.value_scale
        raw_logit = float(np.r_[1.0, z] @ self.value_beta)
        calibrated = self.value["platt_intercept"] + self.value["platt_slope"] * raw_logit
        return self.sigmoid(calibrated)

    def opponent_score(self, strategy: Any, state: Any, action: str, card_id: int) -> float:
        return float(opponent_features(strategy, state, action, card_id) @ self.opponent_beta)


def weighted_pool(module: Any, row: dict[str, Any], excluded: set[int], turn_offset: int = 1) -> list[tuple[int, float]]:
    current = int(row["card_action_index"]) + turn_offset
    ages = {int(card_id): int(age) for card_id, age in row.get("cooldown_last_seen", {}).items()}
    pool = []
    for card_id in range(102):
        if card_id in excluded:
            continue
        old_age = ages.get(card_id)
        weight = 1.0 if old_age is None else module.OpponentBelief._return_weight_at(0, old_age + turn_offset)
        if weight > 0:
            pool.append((card_id, weight))
    return pool


def common_draws(pool: list[tuple[int, float]], uniforms: list[float]) -> list[int]:
    if not pool:
        return [-1] * len(uniforms)
    total = sum(weight for _card_id, weight in pool)
    cumulative, running = [], 0.0
    for card_id, weight in sorted(pool):
        running += weight / total
        cumulative.append((running, card_id))
    result = []
    for uniform in uniforms:
        for threshold, card_id in cumulative:
            if uniform <= threshold:
                result.append(card_id)
                break
        else:
            result.append(cumulative[-1][1])
    return result


def draw_category(strategy: Any, card_id: int, me: Any) -> tuple[Any, ...]:
    if card_id < 0:
        return ("none",)
    card = strategy.catalog[card_id]
    resource = "ore" if card.ore else "mana" if card.mana else "army" if card.army else "free"
    cost = card.total_cost
    band = 0 if cost <= 3 else 1 if cost <= 7 else 2 if cost <= 12 else 3
    return (
        resource, band,
        card_id in strategy.PRODUCTION_CARDS,
        card_id in strategy.EXTRA_TURN_CARDS,
        strategy._affordable(card, me),
    )


def income(strategy: Any, actor: Any) -> Any:
    return strategy._income(actor)


def mirrored_state(module: Any, base: Any, actor: Any, defender: Any, hand: list[int] | None = None):
    return replace(base, player_no=2, players={1: defender, 2: actor}, hand=list(hand or []), must_discard=False)


def own_state(module: Any, base: Any, me: Any, enemy: Any, hand: list[int]):
    return replace(base, player_no=1, players={1: me, 2: enemy}, hand=list(hand), must_discard=False)


def evaluate_extra_continuation(module: Any, strategy: Any, models: Models, row: dict[str, Any], state: Any) -> tuple[float, bool]:
    best, immediate = models.pwin(row, state, strategy), False
    for card_id in state.hand:
        card = strategy.catalog.cards.get(card_id)
        if card is None or not strategy._affordable(card, state.me):
            continue
        me2, enemy2 = strategy.simulate(card, state)
        next_state = own_state(module, state, me2, enemy2, list(state.hand))
        win = me2.tower >= 50 or enemy2.tower <= 0 or min(me2.ore, me2.mana, me2.army) >= 150
        immediate |= win
        best = max(best, 1.0 if win else models.pwin(row, next_state, strategy))
    return best, immediate


def opponent_responses(
    module: Any, strategy: Any, models: Models, row: dict[str, Any], state: Any,
    retained_hand: list[int], representative_draw: int, pool: list[tuple[int, float]],
) -> dict[str, float]:
    enemy = income(strategy, state.opponent)
    if enemy.tower >= 50 or state.me.tower <= 0 or min(enemy.ore, enemy.mana, enemy.army) >= 150:
        return {"empirical": 0.0, "cvar": 0.0, "adversarial": 0.0, "p_lose_next": 1.0}
    our_hand = retained_hand + ([representative_draw] if representative_draw >= 0 else [])
    actor_state = mirrored_state(module, state, enemy, state.me)
    candidate_cards = [card_id for card_id, _weight in sorted(pool, key=lambda item: (-item[1], item[0]))[:10]]
    candidates: list[tuple[float, float, bool]] = []
    # Card-specific discards are behaviorally distinct but board-equivalent.
    for card_id in candidate_cards:
        drop_score = models.opponent_score(strategy, actor_state, "drop", card_id)
        after_drop = own_state(module, state, income(strategy, state.me), enemy, our_hand)
        candidates.append((drop_score, models.pwin(row, after_drop, strategy), False))
        card = strategy.catalog[card_id]
        if strategy._affordable(card, enemy):
            reply_enemy, reply_me = strategy.simulate(card, actor_state)
            lost = reply_enemy.tower >= 50 or reply_me.tower <= 0 or min(reply_enemy.ore, reply_enemy.mana, reply_enemy.army) >= 150
            after_reply = own_state(module, state, income(strategy, reply_me), reply_enemy, our_hand)
            candidates.append((models.opponent_score(strategy, actor_state, "turn", card_id), 0.0 if lost else models.pwin(row, after_reply, strategy), lost))
    if not candidates:
        return {"empirical": models.pwin(row, state, strategy), "cvar": models.pwin(row, state, strategy), "adversarial": models.pwin(row, state, strategy), "p_lose_next": 0.0}
    logits = np.asarray([item[0] for item in candidates])
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    values = np.asarray([item[1] for item in candidates])
    empirical = float(probabilities @ values)
    order = np.argsort(values)
    remaining, tail_sum = .10, 0.0
    for index in order:
        portion = min(remaining, float(probabilities[index]))
        tail_sum += portion * float(values[index])
        remaining -= portion
        if remaining <= 1e-12:
            break
    cvar = tail_sum / max(1e-9, .10 - remaining)
    p_lose = float(sum(probability for probability, item in zip(probabilities, candidates) if item[2]))
    return {"empirical": empirical, "cvar": cvar, "adversarial": float(values.min()), "p_lose_next": p_lose}


def evaluate_action(module: Any, strategy: Any, models: Models, row: dict[str, Any], state: Any, action: dict[str, Any], uniforms: list[float]) -> dict[str, Any]:
    slot, card_id, action_type = int(action["slot"]), int(action["card_id"]), str(action["action"])
    retained = list(state.hand)
    retained.pop(slot)
    me1, enemy1 = state.me, state.opponent
    if action_type == "turn":
        me1, enemy1 = strategy.simulate(strategy.catalog[card_id], state)
    immediate_win = me1.tower >= 50 or enemy1.tower <= 0 or min(me1.ore, me1.mana, me1.army) >= 150
    pool = weighted_pool(module, row, set(retained) | {card_id}, 1)
    draws = common_draws(pool, uniforms)
    draw_counts = Counter(draws)
    grouped: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
    for draw_id, count in draw_counts.items():
        grouped[draw_category(strategy, draw_id, me1)][draw_id] += count
    sample_values: list[float] = []
    win_two = 0.0
    for category_counts in grouped.values():
        draw_id, _representative_count = category_counts.most_common(1)[0]
        count = sum(category_counts.values())
        hand = retained + ([draw_id] if draw_id >= 0 else [])
        post = own_state(module, state, me1, enemy1, hand)
        if immediate_win:
            value, next_win = 1.0, True
        elif action_type == "turn" and card_id in strategy.EXTRA_TURN_CARDS:
            value, next_win = evaluate_extra_continuation(module, strategy, models, row, post)
        else:
            value, next_win = models.pwin(row, post, strategy), False
        sample_values.extend([value] * count)
        if next_win:
            win_two += count
    mean_before_reply = statistics.fmean(sample_values)
    representative = draw_counts.most_common(1)[0][0]
    if immediate_win:
        response = {"empirical": 1.0, "cvar": 1.0, "adversarial": 1.0, "p_lose_next": 0.0}
    elif action_type == "turn" and card_id in strategy.EXTRA_TURN_CARDS:
        response = {"empirical": mean_before_reply, "cvar": mean_before_reply, "adversarial": mean_before_reply, "p_lose_next": 0.0}
    else:
        representative_state = own_state(module, state, me1, enemy1, retained + ([representative] if representative >= 0 else []))
        response = opponent_responses(module, strategy, models, row, representative_state, retained, representative, pool)
    # Preserve replacement-draw variation and apply the response model as a
    # board-level correction.  Robust mixture was intentionally not hardcoded
    # to pure minimax: population expectation dominates, tail remains explicit.
    robust_reply = .70 * response["empirical"] + .20 * response["cvar"] + .10 * response["adversarial"]
    q_emp = max(0.0, min(1.0, mean_before_reply + response["empirical"] - models.pwin(row, own_state(module, state, me1, enemy1, retained + ([representative] if representative >= 0 else [])), strategy)))
    q_robust = max(0.0, min(1.0, mean_before_reply + robust_reply - models.pwin(row, own_state(module, state, me1, enemy1, retained + ([representative] if representative >= 0 else [])), strategy)))
    q_adv = max(0.0, min(1.0, mean_before_reply + response["adversarial"] - models.pwin(row, own_state(module, state, me1, enemy1, retained + ([representative] if representative >= 0 else [])), strategy)))
    se = statistics.pstdev(sample_values) / math.sqrt(len(sample_values)) if len(sample_values) > 1 else 0.0
    return {
        "action": action_type, "slot": slot, "card_id": card_id,
        "q_empirical": q_emp, "q_robust": q_robust, "q_adversarial": q_adv,
        "q_se": se, "ci95": [max(0.0, q_robust - 1.96 * se), min(1.0, q_robust + 1.96 * se)],
        "cvar": response["cvar"], "p_lose_next_turn": response["p_lose_next"],
        "p_win_within_2_own_actions": win_two / len(uniforms),
        "immediate_win": immediate_win, "horizon_estimate": None,
        "replacement_unique": len(draw_counts), "replacement_categories": len(grouped),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("horizon_model", type=Path)
    parser.add_argument("value_model", type=Path)
    parser.add_argument("opponent_model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--seed-suffix", default="v1")
    args = parser.parse_args()
    splits = set(args.splits.split(","))
    if "holdout" in splits:
        raise ValueError("Stage A development run must not open holdout")
    selected_ids: set[str] | None = None
    selection_kind: dict[str, str] = {}
    if args.selection:
        selection_payload = json.loads(args.selection.read_text(encoding="utf-8"))
        selection_kind = {str(item["state_id"]): str(item.get("selection") or "selected") for item in selection_payload["states"]}
        selected_ids = set(selection_kind)
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    catalog_raw = {int(card["id"]): card for card in json.loads((args.source / "cards_catalog.json").read_text(encoding="utf-8"))}
    horizon = json.loads(args.horizon_model.read_text(encoding="utf-8"))
    value = json.loads(args.value_model.read_text(encoding="utf-8"))
    opponent = json.loads(args.opponent_model.read_text(encoding="utf-8"))
    models = Models(catalog_raw, horizon, value, opponent)
    strategy = module.StrategicCardStrategy(catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter()
    regrets: list[float] = []
    runtimes: list[float] = []
    started = time.monotonic()
    with gzip.open(args.output, "wt", encoding="utf-8", newline="\n") as output:
        selected_index = 0
        for global_index, row in enumerate(dataset_rows(args.dataset, splits)):
            if selected_ids is not None and row["state_id"] not in selected_ids:
                continue
            if global_index % args.shard_count != args.shard_index:
                continue
            if args.limit and selected_index >= args.limit:
                break
            selected_index += 1
            state_started = time.monotonic()
            state = state_from_row(module, row)
            seed = hashlib.sha256(f"stage-a:{row['state_id']}:{args.seed_suffix}".encode()).digest()
            rng = random.Random(seed)
            uniforms = [rng.random() for _ in range(args.particles)]
            actions = [evaluate_action(module, strategy, models, row, state, action, uniforms) for action in row["legal_actions"]]
            actions.sort(key=lambda item: item["q_robust"], reverse=True)
            historical = row["historical_choice"]
            chosen = next(item for item in actions if item["action"] == historical["action"] and item["slot"] == historical["slot"])
            regret = max(0.0, actions[0]["q_robust"] - chosen["q_robust"])
            margin = actions[0]["q_robust"] - actions[1]["q_robust"] if len(actions) > 1 else 1.0
            elapsed = time.monotonic() - state_started
            result = {
                "state_id": row["state_id"], "game_id": row["game_id"], "split": row["split"],
                "decision_index": row["decision_index"], "card_action_index": row["card_action_index"],
                "phase": row["phase"], "outcome": row["outcome"], "initiative": row["initiative"],
                "historical_choice": historical, "stage_a_best": {key: actions[0][key] for key in ("action", "slot", "card_id")},
                "historical_regret": regret, "decision_margin": margin,
                "particles_requested": args.particles, "particles_completed": args.particles,
                "common_random_numbers": True, "runtime_seconds": elapsed, "actions": actions,
                "selection": selection_kind.get(row["state_id"]), "seed_suffix": args.seed_suffix,
            }
            output.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            regrets.append(regret)
            runtimes.append(elapsed)
            totals["states"] += 1
            totals[f"states_{row['split']}"] += 1
            totals["legal_actions"] += len(actions)
            totals["disagreements"] += int(actions[0]["action"] != historical["action"] or actions[0]["slot"] != historical["slot"])
            if totals["states"] % 500 == 0:
                print(json.dumps({"states": totals["states"], "elapsed": time.monotonic() - started, "last_runtime": elapsed}), flush=True)
    summary = {
        "version": "stage-a-oracle-3.9-v1", "splits": sorted(splits),
        "shard_count": args.shard_count, "shard_index": args.shard_index,
        "holdout_opened": False, "particles": args.particles,
        "totals": dict(totals), "runtime_seconds": time.monotonic() - started,
        "runtime": {
            "p50": statistics.median(runtimes), "p90": float(np.quantile(runtimes, .9)),
            "p95": float(np.quantile(runtimes, .95)), "p99": float(np.quantile(runtimes, .99)), "max": max(runtimes),
        },
        "regret": {
            "mean": statistics.fmean(regrets), "median": statistics.median(regrets),
            "p90": float(np.quantile(regrets, .9)), "p95": float(np.quantile(regrets, .95)),
            "gt_2pp": sum(value > .02 for value in regrets), "gt_5pp": sum(value > .05 for value in regrets),
            "gt_10pp": sum(value > .10 for value in regrets),
        },
        "caveat": "Stage A is a 500-world learned-value screen; Stage B/C establish expensive labels for suspicious and random states.",
    }
    summary_path = args.output.with_suffix("").with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
