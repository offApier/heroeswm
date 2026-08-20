from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def load_module(name: str, source: Path):
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location(name, source / "card_game.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def state(module: Any, row: dict[str, Any]):
    raw = row["visible_state"]
    names = ("ore", "mana", "army", "tower", "wall", "mine", "monastery", "barracks")
    make = lambda values: module.PlayerState(**{name: int(values.get(name) or 0) for name in names})
    me, enemy = make(raw["me"]), make(raw["opponent"])
    return module.GameState(
        game_id=int(row["game_id"]), turn=int(raw.get("turn") or 0), is_your_turn=True,
        player_no=1, time_left=40, players={1: me, 2: enemy}, hand=list(row["our_hand"]),
        last_move=str(raw.get("last_move") or ""), table=str(raw.get("table") or ""),
        must_discard=bool(raw.get("must_discard")), first_actor=str(row.get("initiative") or "unknown"),
        reconnect_uncertainty=bool(row.get("reconnect")),
        unknown_transitions=int(row.get("reconnect_unknown_transitions_before") or 0),
    )


def restore(strategy: Any, row: dict[str, Any]) -> None:
    strategy.reset_game(int(row["game_id"]))
    for item in row.get("history") or []:
        turn = int(item.get("turn") or 0)
        card_id = item.get("card_id")
        if card_id is None:
            strategy.belief.unknown_action_indices.add(turn)
            strategy.belief.current_action = max(strategy.belief.current_action, turn)
            continue
        action = str(item.get("action") or "turn")
        move = ("d" if action == "drop" else "t") + str(card_id) + "-0"
        strategy.belief._record(turn, move, int(card_id), str(item.get("actor") or "unknown"), action)


def fixed_ranking(module: Any, strategy: Any, current: Any, particles: int):
    simulate_cache = getattr(strategy, "_simulate_cache", None)
    if simulate_cache is not None:
        simulate_cache.clear()
    for name in (
        "_win_next_cache", "_finisher_pool_cache", "_particle_next_win_cache", "_decision_reply_cache",
        "_decision_extra_reply_cache", "_decision_next_win_cache", "_decision_quantile_cache",
    ):
        cache = getattr(strategy, name, None)
        if cache is not None:
            cache.clear()
    unseen = strategy.belief.unseen_pool(current)
    worlds = strategy.belief.particles(current, particles)
    choices = []
    for action in current_actions(strategy, current):
        sink: dict[str, Any] = {}
        choice = strategy._evaluate_choice(
            action[0], action[1], action[2], current, worlds, unseen, tuple(), sample_sink=sink
        )
        choices.append(strategy._aggregate_samples(choice, sink, current, tuple()))
    choices = strategy._with_policy_scores(current, choices)
    choices.sort(key=strategy._choice_sort_key, reverse=True)
    return choices


def current_actions(strategy: Any, current: Any):
    result = []
    for slot, card_id in enumerate(current.hand):
        card = strategy.catalog.cards.get(card_id)
        if card is None:
            continue
        result.append(("drop", slot, card))
        if not current.must_discard and strategy._affordable(card, current.me):
            result.append(("turn", slot, card))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--particles", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = load_module("policy39_frozen_equivalence", args.frozen)
    new = load_module("policy39_optimized_equivalence", args.optimized)
    old_catalog = old.CardCatalog.load(args.frozen / "cards_catalog.json")
    new_catalog = new.CardCatalog.load(args.optimized / "cards_catalog.json")
    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    rows.sort(key=lambda row: hashlib.sha256(f"equivalence:{row['state_id']}".encode()).hexdigest())
    # Deterministic spread across phase, reconnect and legal-action complexity.
    selected, seen_bands = [], set()
    for row in rows:
        band = (row.get("phase"), bool(row.get("reconnect")), len(row.get("legal_actions") or []))
        if band in seen_bands and len(selected) < args.states // 2:
            continue
        selected.append(row)
        seen_bands.add(band)
        if len(selected) >= args.states:
            break
    details, ranking_mismatches, selection_mismatches = [], 0, 0
    max_score_difference = max_probability_difference = 0.0
    for row in selected:
        old_strategy, new_strategy = old.CardStrategy(old_catalog), new.CardStrategy(new_catalog)
        restore(old_strategy, row)
        restore(new_strategy, row)
        old_rank = fixed_ranking(old, old_strategy, state(old, row), args.particles)
        new_rank = fixed_ranking(new, new_strategy, state(new, row), args.particles)
        old_keys = [(item.action, item.slot, item.card.id) for item in old_rank]
        new_keys = [(item.action, item.slot, item.card.id) for item in new_rank]
        ranking_mismatches += int(old_keys != new_keys)
        selection_mismatches += int(old_keys[0] != new_keys[0])
        old_by_key = {key: item for key, item in zip(old_keys, old_rank)}
        state_score = state_probability = 0.0
        for key, item in zip(new_keys, new_rank):
            baseline = old_by_key[key]
            state_score = max(state_score, abs(float(item.policy_score or 0) - float(baseline.policy_score or 0)))
            state_probability = max(state_probability, abs(float(item.p_win or 0) - float(baseline.p_win or 0)))
        max_score_difference = max(max_score_difference, state_score)
        max_probability_difference = max(max_probability_difference, state_probability)
        details.append({
            "state_id": row["state_id"], "phase": row.get("phase"), "reconnect": row.get("reconnect"),
            "legal_actions": len(row.get("legal_actions") or []), "same_ranking": old_keys == new_keys,
            "same_selection": old_keys[0] == new_keys[0], "max_score_difference": state_score,
            "max_probability_difference": state_probability,
        })
    payload = {
        "states": len(details), "particles": args.particles,
        "ranking_mismatches": ranking_mismatches, "selection_mismatches": selection_mismatches,
        "max_abs_policy_score_difference": max_score_difference,
        "max_abs_pwin_difference": max_probability_difference,
        "pass": not ranking_mismatches and max_score_difference <= 1e-12 and max_probability_difference <= 1e-12,
        "details": details,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
