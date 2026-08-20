from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_module(source: Path):
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("policy39_performance", source / "card_game.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_state(module: Any, row: dict[str, Any], time_left: int):
    raw = row["visible_state"]
    names = ("ore", "mana", "army", "tower", "wall", "mine", "monastery", "barracks")
    make = lambda values: module.PlayerState(**{name: int(values.get(name) or 0) for name in names})
    me, enemy = make(raw["me"]), make(raw["opponent"])
    return module.GameState(
        game_id=int(row["game_id"]), turn=int(raw.get("turn") or 0), is_your_turn=True,
        player_no=1, time_left=time_left, players={1: me, 2: enemy}, hand=list(row["our_hand"]),
        last_move=str(raw.get("last_move") or ""), table=str(raw.get("table") or ""),
        must_discard=bool(raw.get("must_discard")), first_actor=str(row.get("initiative") or "unknown"),
        reconnect_uncertainty=bool(row.get("reconnect")),
        unknown_transitions=int(row.get("reconnect_unknown_transitions_before") or 0),
    )


def restore(strategy: Any, row: dict[str, Any]) -> None:
    strategy.reset_game(int(row["game_id"]))
    for item in row.get("history") or []:
        turn, card_id = int(item.get("turn") or 0), item.get("card_id")
        if card_id is None:
            strategy.belief.unknown_action_indices.add(turn)
            strategy.belief.current_action = max(strategy.belief.current_action, turn)
            continue
        action = str(item.get("action") or "turn")
        move = ("d" if action == "drop" else "t") + str(card_id) + "-0"
        strategy.belief._record(turn, move, int(card_id), str(item.get("actor") or "unknown"), action)


def distribution(values: list[float]) -> dict[str, float]:
    data = np.asarray(values)
    return {name: float(np.quantile(data, q)) for name, q in (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("p95", .95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--time-left", type=int, default=15)
    parser.add_argument("--fixed-particles", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    wanted = {item["state_id"]: item for item in json.loads(args.manifest.read_text(encoding="utf-8"))["states"]}
    rows = {}
    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["state_id"] in wanted:
                rows[row["state_id"]] = row
    results = []
    for identifier, meta in wanted.items():
        row = rows[identifier]
        strategy = module.CardStrategy(catalog)
        if args.fixed_particles:
            strategy.MIN_PARTICLES = args.fixed_particles
            strategy.MAX_PARTICLES = args.fixed_particles
            strategy.PARTICLE_LIMIT_BY_TIMEOUT = {
                15: args.fixed_particles,
                30: args.fixed_particles,
                40: args.fixed_particles,
            }
        restore(strategy, row)
        current = make_state(module, row, args.time_left)
        started = time.perf_counter()
        ranking = strategy.rank_choices(current)
        elapsed = time.perf_counter() - started
        top = ranking[0]
        results.append({
            **meta, "elapsed": elapsed, "particles_requested": top.particles_requested,
            "particles_completed": top.particles_completed, "stopping_reason": top.stopping_reason,
            "deadline_hit": top.analysis_deadline_hit, "deadline_remaining": top.deadline_remaining,
            "decision_margin": top.decision_margin, "se_diff": top.se_diff, "ci_diff": top.ci_diff,
            "selected": [top.action, top.slot, top.card.id],
        })
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    particles = [row["particles_completed"] for row in results]
    elapsed = [row["elapsed"] for row in results]
    fractions = {
        "lt80": sum(value < 80 for value in particles) / len(particles),
        "lt120": sum(value < 120 for value in particles) / len(particles),
        "lt160": sum(value < 160 for value in particles) / len(particles),
        "lt200": sum(value < 200 for value in particles) / len(particles),
        "ge200": sum(value >= 200 for value in particles) / len(particles),
        "ge400": sum(value >= 400 for value in particles) / len(particles),
    }
    payload = {
        "version": "fixed-performance-benchmark-3.9-v1", "time_left": args.time_left,
        "fixed_particles": args.fixed_particles or None,
        "source": str(args.source), "states": len(results), "results": results,
        "runtime_seconds": {**distribution(elapsed), "max": max(elapsed)},
        "particles_completed": {**distribution(particles), "min": min(particles), "max": max(particles), "counts": Counter(particles), "fractions": fractions},
        "deadlines": {
            "deadline_precheck": sum(row["stopping_reason"] == "deadline_precheck" for row in results),
            "deadline_in_batch": sum(row["stopping_reason"] == "deadline_in_batch" for row in results),
            "hard_timeout": 0,
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
