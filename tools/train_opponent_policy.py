from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from audit_card_dataset import load_segments, merge_events, parsed_move  # noqa: E402


FEATURES = (
    "play", "drop", "immediate_win", "self_terminal_loss",
    "our_tower", "enemy_tower", "our_wall", "enemy_wall",
    "our_resources", "enemy_resources", "our_production", "enemy_production",
    "tower_race_swing", "destruction_swing", "resource_race_swing",
    "normal_damage_absorbed", "ore_cost", "mana_cost", "army_cost", "cost_sq",
    "extra_turn", "production_card", "symmetric", "self_damage",
    "early_production", "late_production", "terminal_tower", "terminal_damage",
    "low_tower_defense", "enemy_near_50_defense", "eta_if_retained",
)


def load_module(source: Path):
    spec = importlib.util.spec_from_file_location("policy39_opponent_card_game", source / "card_game.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load card_game.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def player(module: Any, raw: dict[str, Any]):
    return module.PlayerState(**{
        name: int(raw.get(name) or 0)
        for name in ("ore", "mana", "army", "tower", "wall", "mine", "monastery", "barracks")
    })


def actor_state(module: Any, game_id: int, raw: dict[str, Any]):
    # Historical snapshots are always stored from our perspective.  The actor
    # in this trainer is the opponent, therefore swap the two public players.
    return module.GameState(
        game_id=game_id,
        turn=int(raw.get("turn") or 0),
        is_your_turn=True,
        player_no=1,
        time_left=int(raw.get("time_left") or 0),
        players={1: player(module, raw.get("opponent") or {}), 2: player(module, raw.get("me") or {})},
        hand=[],
        winner=0,
        finish_reason=0,
        last_move=str(raw.get("last_move") or ""),
        now_player=1,
        table=str(raw.get("table") or ""),
        must_discard=bool(raw.get("must_discard")),
    )


def won(p: Any, e: Any) -> bool:
    return p.tower >= 50 or e.tower <= 0 or min(p.ore, p.mana, p.army) >= 150


def feature_vector(strategy: Any, state: Any, action: str, card_id: int) -> np.ndarray:
    card = strategy.catalog[card_id]
    me0, en0 = state.me, state.opponent
    if action == "turn":
        me1, en1 = strategy.simulate(card, state)
    else:
        me1, en1 = me0, en0
    dt_me, dt_en = me1.tower - me0.tower, en1.tower - en0.tower
    dw_me, dw_en = me1.wall - me0.wall, en1.wall - en0.wall
    dr_me = (me1.ore + me1.mana + me1.army) - (me0.ore + me0.mana + me0.army)
    dr_en = (en1.ore + en1.mana + en1.army) - (en0.ore + en0.mana + en0.army)
    dp_me = (me1.mine + me1.monastery + me1.barracks) - (me0.mine + me0.monastery + me0.barracks)
    dp_en = (en1.mine + en1.monastery + en1.barracks) - (en0.mine + en0.monastery + en0.barracks)
    effect = card.effect.lower().replace("ё", "е")
    ordinary = strategy.GENERAL_DAMAGE.get(card_id, 0) if action == "turn" else 0
    absorbed = min(en0.wall, ordinary)
    terminal = min(50 - me0.tower, en0.tower, 50 - en0.tower, me0.tower) <= 10
    early = state.turn <= 20
    producer = int(card_id in strategy.PRODUCTION_CARDS and action == "turn")
    extra = int(card_id in strategy.EXTRA_TURN_CARDS and action == "turn")
    self_damage = int(action == "turn" and (dt_me < 0 or "вы теряете" in effect or "вашей башне" in effect))
    symmetric = int(action == "turn" and ("все " in effect or "обоих" in effect))
    production = max(1, me0.mine if card.ore else me0.monastery if card.mana else me0.barracks)
    current = me0.ore if card.ore else me0.mana if card.mana else me0.army
    eta = max(0, math.ceil((card.total_cost - current) / production)) if card.total_cost else 0
    values = (
        int(action == "turn"), int(action == "drop"), int(won(me1, en1)), int(me1.tower <= 0),
        dt_me / 20, dt_en / 20, dw_me / 20, dw_en / 20,
        dr_me / 30, dr_en / 30, dp_me / 3, dp_en / 3,
        ((50 - me0.tower) - (50 - me1.tower) - ((50 - en0.tower) - (50 - en1.tower))) / 20,
        ((en0.tower + .65 * en0.wall) - (en1.tower + .65 * en1.wall)) / 20,
        ((min(me1.ore, me1.mana, me1.army) - min(me0.ore, me0.mana, me0.army))
         - (min(en1.ore, en1.mana, en1.army) - min(en0.ore, en0.mana, en0.army))) / 20,
        absorbed / 10, card.ore / 20, card.mana / 20, card.army / 20, (card.total_cost / 20) ** 2,
        extra, producer, symmetric, self_damage,
        producer * early, producer * (not early), terminal * dt_me / 10, terminal * (-dt_en) / 10,
        (me0.tower <= 10) * (dt_me + max(0, dw_me)) / 10,
        (en0.tower >= 40) * (-dt_en) / 10,
        min(10, eta) / 10,
    )
    return np.asarray(values, dtype=np.float32)


def return_weight(module: Any, age: int | None) -> float:
    if age is None:
        return 1.0
    return module.OpponentBelief._return_weight_at(0, age)


def sample_other_cards(
    module: Any,
    catalog: Any,
    raw: dict[str, Any],
    actual: int,
    last_seen: dict[int, int],
    rng: random.Random,
) -> list[int]:
    turn = int(raw.get("turn") or 0)
    ours = set(int(card_id) for card_id in raw.get("hand") or [])
    pool = []
    for card_id in catalog.cards:
        if card_id == actual or card_id in ours:
            continue
        age = None if card_id not in last_seen else max(0, turn - last_seen[card_id])
        weight = return_weight(module, age)
        if weight > 0:
            pool.append((card_id, weight))
    selected: list[int] = []
    while pool and len(selected) < 5:
        total = sum(weight for _card_id, weight in pool)
        needle = rng.random() * total
        index = 0
        for index, (_card_id, weight) in enumerate(pool):
            needle -= weight
            if needle <= 0:
                break
        selected.append(pool.pop(index)[0])
    return selected


def choice_set(strategy: Any, state: Any, cards: list[int]) -> list[tuple[str, int]]:
    result = [("drop", card_id) for card_id in cards]
    if not state.must_discard:
        result.extend(
            ("turn", card_id)
            for card_id in cards
            if strategy._affordable(strategy.catalog[card_id], state.me)
        )
    return result


def collect_examples(module: Any, catalog: Any, card_games: Path, assignments: dict[int, str]):
    games = load_segments(card_games)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    for game_id, segments in sorted(games.items()):
        split = assignments[game_id]
        if split == "holdout":
            continue
        events, artifacts, _duplicates = merge_events(segments)
        last_seen: dict[int, int] = {}
        previous_turn = 0
        strategy = module.CardStrategy(catalog)
        for event in events:
            turn = int(event.get("turn") or 0)
            move = parsed_move(event)
            before = event.get("before") or {}
            gap = max(0, turn - int(before.get("turn") or previous_turn) - 1)
            if str(event.get("actor")) == "opponent" and move is not None and gap == 0:
                actual_action, actual_card = move
                state = actor_state(module, game_id, before)
                # Four independent compatible pseudo-hands reduce hidden-choice
                # bias without pretending that the actual opponent hand is known.
                for particle_no in range(4):
                    seed = hashlib.sha256(f"opp:{game_id}:{turn}:{particle_no}".encode()).digest()
                    rng = random.Random(seed)
                    cards = [actual_card] + sample_other_cards(module, catalog, before, actual_card, last_seen, rng)
                    candidates = choice_set(strategy, state, cards)
                    actual = (actual_action, actual_card)
                    if actual not in candidates:
                        counters["illegal_or_unreconstructable_actual"] += 1
                        continue
                    examples[split].append({
                        "game_id": game_id,
                        "turn": turn,
                        "actual": actual,
                        "features": np.stack([
                            feature_vector(strategy, state, action, card_id)
                            for action, card_id in candidates
                        ]),
                        "candidates": candidates,
                        "actual_index": candidates.index(actual),
                    })
                counters[f"events_{split}"] += 1
                counters[f"actions_{split}_{actual_action}"] += 1
            if move is not None:
                last_seen[move[1]] = turn
            previous_turn = max(previous_turn, turn)
        counters["terminal_artifacts_removed"] += artifacts
    return examples, counters


def pairwise_matrix(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    game_counts = Counter(item["game_id"] for item in examples)
    rows, weights = [], []
    for item in examples:
        x = item["features"]
        chosen = x[item["actual_index"]]
        alternatives = np.delete(x, item["actual_index"], axis=0)
        if len(alternatives) == 0:
            continue
        rows.append(chosen - alternatives)
        weights.extend([1.0 / game_counts[item["game_id"]] / len(alternatives)] * len(alternatives))
    matrix = np.concatenate(rows).astype(np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    sample_weights *= len(sample_weights) / sample_weights.sum()
    return matrix, sample_weights


def train_pairwise(x: np.ndarray, sample_weights: np.ndarray, l2: float, epochs: int = 180) -> np.ndarray:
    # Deterministic mini-batch Adam keeps the audit practical on the full
    # archive while still optimizing the convex pairwise objective.
    weights = np.zeros(x.shape[1], dtype=np.float64)
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    rng = np.random.default_rng(39017)
    batch_size = min(32768, len(x))
    sampling_p = sample_weights / sample_weights.sum()
    for epoch in range(1, epochs + 1):
        indices = rng.choice(len(x), size=batch_size, replace=True, p=sampling_p)
        batch = x[indices]
        logits = np.clip(batch @ weights, -30, 30)
        probability = 1 / (1 + np.exp(-logits))
        gradient = -((1 - probability)[:, None] * batch).mean(axis=0) + l2 * weights
        m = .9 * m + .1 * gradient
        v = .999 * v + .001 * gradient * gradient
        step = .045 * math.sqrt(1 - .999 ** epoch) / (1 - .9 ** epoch)
        weights -= step * m / (np.sqrt(v) + 1e-8)
    return weights


def evaluate(examples: list[dict[str, Any]], weights: np.ndarray) -> dict[str, Any]:
    ranks, losses, best, second, discard_actual, discard_predicted = [], [], 0, 0, 0, 0
    by_event: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in examples:
        by_event[(item["game_id"], item["turn"])].append(item)
    event_metrics = []
    for key, particles in by_event.items():
        particle_probabilities = []
        particle_ranks = []
        for item in particles:
            scores = item["features"] @ weights
            shifted = scores - scores.max()
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
            actual_index = item["actual_index"]
            rank = 1 + int(np.sum(scores > scores[actual_index]))
            particle_probabilities.append(float(probabilities[actual_index]))
            particle_ranks.append(rank)
            discard_predicted += int(item["candidates"][int(np.argmax(scores))][0] == "drop")
        probability = max(1e-9, float(np.mean(particle_probabilities)))
        rank = int(round(float(np.mean(particle_ranks))))
        ranks.append(rank)
        losses.append(-math.log(probability))
        best += int(rank == 1)
        second += int(rank <= 2)
        discard_actual += int(particles[0]["actual"][0] == "drop")
        event_metrics.append({"game_id": key[0], "turn": key[1], "rank": rank, "p_actual": probability})
    count = len(ranks)
    return {
        "events": count,
        "negative_log_likelihood": float(np.mean(losses)),
        "top1": best / count,
        "top2": second / count,
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean([1 / rank for rank in ranks])),
        "actual_discard_rate": discard_actual / count,
        "predicted_discard_rate_particle": discard_predicted / max(1, len(examples)),
        "event_metrics": event_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("card_games", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    assignments = {int(key): value for key, value in manifest["assignments"].items()}
    examples, counters = collect_examples(module, catalog, args.card_games, assignments)
    x, sample_weights = pairwise_matrix(examples["train"])
    candidates = []
    for l2 in (0.0005, 0.003, 0.02):
        weights = train_pairwise(x, sample_weights, l2)
        validation = evaluate(examples["validation"], weights)
        candidates.append((validation["negative_log_likelihood"], l2, weights, validation))
    _loss, l2, weights, validation = min(candidates, key=lambda item: item[0])
    training = evaluate(examples["train"], weights)
    # Detailed event rows are useful during development but do not belong in
    # the compact runtime model.
    training.pop("event_metrics")
    validation.pop("event_metrics")
    payload = {
        "version": "opponent-population-ranker-3.9-v1",
        "method": "pairwise conditional logit over compatible pseudo-hands",
        "future_information_used": False,
        "holdout_opened": False,
        "pseudo_hands_per_event": 4,
        "features": list(FEATURES),
        "weights": [float(value) for value in weights],
        "selected_l2": l2,
        "collection": dict(counters),
        "pairwise_rows": len(x),
        "training": training,
        "validation": validation,
        "candidate_l2_validation_nll": {str(item[1]): item[3]["negative_log_likelihood"] for item in candidates},
        "limitations": [
            "Opponent hand is hidden; choice sets are posterior-compatible pseudo-hands, not claimed ground truth.",
            "Model estimates population action preference and is mixed with adversarial tail risk at decision time.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"weights"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
