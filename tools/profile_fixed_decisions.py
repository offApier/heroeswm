from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import statistics
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def load_module(source: Path):
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("policy39_profile_fixed", source / "card_game.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_state(module: Any, row: dict[str, Any]):
    raw = row["visible_state"]
    names = ("ore", "mana", "army", "tower", "wall", "mine", "monastery", "barracks")
    make = lambda values: module.PlayerState(**{name: int(values.get(name) or 0) for name in names})
    return module.GameState(
        game_id=int(row["game_id"]), turn=int(raw.get("turn") or 0), is_your_turn=True,
        player_no=1, time_left=15, players={1: make(raw["me"]), 2: make(raw["opponent"])},
        hand=list(row["our_hand"]), last_move=str(raw.get("last_move") or ""),
        table=str(raw.get("table") or ""), must_discard=bool(raw.get("must_discard")),
        first_actor=str(row.get("initiative") or "unknown"), reconnect_uncertainty=bool(row.get("reconnect")),
        unknown_transitions=int(row.get("reconnect_unknown_transitions_before") or 0),
    )


def restore(strategy: Any, row: dict[str, Any]) -> None:
    strategy.reset_game(int(row["game_id"]))
    for item in row.get("history") or []:
        turn, card_id = int(item.get("turn") or 0), item.get("card_id")
        if card_id is None:
            strategy.belief.unknown_action_indices.add(turn)
            continue
        action = str(item.get("action") or "turn")
        strategy.belief._record(turn, ("d" if action == "drop" else "t") + str(card_id) + "-0", int(card_id), str(item.get("actor") or "unknown"), action)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--particles", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["states"]
    # One deterministic state per category.
    chosen, categories = [], set()
    for item in manifest:
        if item["category"] not in categories:
            chosen.append(item)
            categories.add(item["category"])
    wanted = {item["state_id"] for item in chosen}
    rows = {}
    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["state_id"] in wanted:
                rows[row["state_id"]] = row
    timings: dict[str, list[float]] = defaultdict(list)
    decision_totals = []

    def wrap(obj: Any, name: str, category: str | Callable[[tuple[Any, ...]], str]) -> None:
        original = getattr(obj, name, None)
        if original is None:
            return
        def measured(_self: Any, *call_args: Any, **kwargs: Any):
            label = category(call_args) if callable(category) else category
            started = time.perf_counter()
            try:
                return original(*call_args, **kwargs)
            finally:
                timings[label].append(time.perf_counter() - started)
        setattr(obj, name, types.MethodType(measured, obj))

    for meta in chosen:
        row = rows[meta["state_id"]]
        strategy = module.CardStrategy(catalog)
        started = time.perf_counter(); restore(strategy, row); timings["belief_update_reconstruction"].append(time.perf_counter() - started)
        current = make_state(module, row)
        wrap(strategy, "simulate", "deterministic_card_simulation")
        wrap(strategy, "_counterfactual_replacement", "replacement_draw_simulation")
        wrap(strategy, "_opponent_reply", "opponent_response")
        wrap(strategy, "_our_extra_continuation_with_draws", "terminal_quiescence_extra_turn")
        wrap(strategy, "_opponent_extra_reply", "terminal_quiescence_extra_turn")
        wrap(strategy, "_position_probability", "position_value")
        wrap(strategy, "_evaluate_choice", lambda values: "PLAY_evaluation" if values and values[0] == "turn" else "DISCARD_evaluation")
        wrap(strategy, "_hand_quality_diagnostics", "diagnostics_logging")
        wrap(strategy, "_belief_threats", "diagnostics_logging")
        wrap(strategy.belief, "particles", "belief_particle_generation_and_copy")
        wrap(strategy.belief, "diagnostics", "diagnostics_logging")
        if strategy.policy_runtime is not None:
            wrap(strategy.policy_runtime, "horizon", "Q_value_model")
            wrap(strategy.policy_runtime, "state_pwin", "Q_value_model")
            wrap(strategy.policy_runtime, "action_score", "Q_value_model")
            wrap(strategy.policy_runtime, "_base_features", "feature_extraction")
            wrap(strategy.policy_runtime, "action_feature_vector", "feature_extraction")
        decision_started = time.perf_counter()
        unseen = strategy.belief.unseen_pool(current)
        worlds = strategy.belief.particles(current, args.particles)
        candidate_started = time.perf_counter()
        candidates = []
        for slot, card_id in enumerate(current.hand):
            card = strategy.catalog.cards.get(card_id)
            if card:
                candidates.append(("drop", slot, card))
                if not current.must_discard and strategy._affordable(card, current.me):
                    candidates.append(("turn", slot, card))
        timings["candidate_generation"].append(time.perf_counter() - candidate_started)
        choices = []
        for action, slot, card in candidates:
            sink: dict[str, Any] = {}
            base = strategy._evaluate_choice(action, slot, card, current, worlds, unseen, tuple(), sample_sink=sink)
            choices.append(strategy._aggregate_samples(base, sink, current, tuple()))
        choices = strategy._with_policy_scores(current, choices)
        strategy._belief_threats(current, worlds)
        strategy._hand_quality_diagnostics(current)
        strategy.belief.diagnostics(current, worlds)
        decision_totals.append(time.perf_counter() - decision_started)
    total = sum(decision_totals)
    payload = {"source": str(args.source), "states": len(chosen), "particles_per_state": args.particles, "total_seconds": total, "profile_is_inclusive": True, "categories": {}}
    for name in (
        "candidate_generation", "deterministic_card_simulation", "PLAY_evaluation", "DISCARD_evaluation",
        "replacement_draw_simulation", "opponent_response", "opponent_policy", "terminal_quiescence_extra_turn",
        "Q_value_model", "feature_extraction", "belief_update_reconstruction", "belief_particle_generation_and_copy",
        "diagnostics_logging",
    ):
        values = timings.get(name, [])
        payload["categories"][name] = {
            "total_seconds": sum(values), "inclusive_percent": 100 * sum(values) / max(total, 1e-12),
            "calls": len(values), "mean_seconds": statistics.fmean(values) if values else 0.0,
            "p95_seconds": percentile(values, .95),
        }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
