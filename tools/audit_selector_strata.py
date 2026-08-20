from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_frozen_policy import q_value
from run_stage_a_oracle import load_module, state_from_row


def read(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result[str(row["state_id"])] = row
    return result


def key(action: dict[str, Any]) -> tuple[str, int]:
    return str(action["action"]), int(action["slot"])


def tail(action: dict[str, Any]) -> float:
    return max(0.0, float(action["q_robust"]) - float(action.get("cvar") or 0.0))


def choose(actions: list[dict[str, Any]], epsilon: float) -> dict[str, Any]:
    immediate = [action for action in actions if action.get("immediate_win")]
    pool = immediate or actions
    top = max(pool, key=lambda action: float(action["q_robust"]))
    equivalent = []
    for action in pool:
        delta = float(top["q_robust"]) - float(action["q_robust"])
        se = math.sqrt(float(top.get("q_se") or 0.0) ** 2 + float(action.get("q_se") or 0.0) ** 2)
        ci = delta - 1.96 * se, delta + 1.96 * se
        if ci[0] >= -epsilon and ci[1] <= epsilon:
            equivalent.append(action)
    return max(
        equivalent or [top],
        key=lambda action: (
            -float(action.get("p_lose_next_turn") or 0.0),
            -tail(action),
            float(action.get("p_win_within_2_own_actions") or 0.0),
            1 if action["action"] == "turn" else 0,
            -int(action["slot"]),
        ),
    )


def stats(values: list[float], agreements: int) -> dict[str, Any]:
    return {
        "states": len(values),
        "mean_regret": statistics.fmean(values),
        "median_regret": statistics.median(values),
        "p90_regret": float(np.quantile(values, 0.90)),
        "p95_regret": float(np.quantile(values, 0.95)),
        "gt_2pp": sum(value > 0.02 for value in values),
        "gt_5pp": sum(value > 0.05 for value in values),
        "oracle_agreement": agreements / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--epsilon", type=float, default=0.0005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical, runtime, oracle = read(args.dataset), read(args.runtime), read(args.oracle)
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    strategy = module.CardStrategy(catalog)
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for state_id in sorted(set(canonical) & set(runtime) & set(oracle)):
        row, low, high = canonical[state_id], runtime[state_id], oracle[state_id]
        high_by_key = {key(action): action for action in high["actions"]}
        selected = choose([action for action in low["actions"] if key(action) in high_by_key], args.epsilon)
        oracle_best = max(high["actions"], key=q_value)
        regret = max(0.0, q_value(oracle_best) - q_value(high_by_key[key(selected)]))
        agreement = key(selected) == key(oracle_best)
        state = state_from_row(module, row)
        card = catalog[int(selected["card_id"])]
        labels = [
            "initiative:first" if row.get("initiative") == "us" else "initiative:second",
            "phase:" + str(row.get("phase") or "unknown"),
            "connection:reconnect" if row.get("reconnect") else "connection:normal",
            "action:" + str(selected["action"]).upper(),
        ]
        if card.id in strategy.EXTRA_TURN_CARDS:
            labels.append("class:extra-turn")
        if card.id in strategy.PRODUCTION_CARDS:
            labels.append("class:production/economy")
        if card.id in strategy.DIRECT_TOWER_DAMAGE:
            labels.append("class:direct-tower")
        if selected["action"] == "turn":
            me2, enemy2 = strategy.simulate(card, state)
            if me2.wall > state.me.wall or me2.tower > state.me.tower:
                labels.append("class:defense")
        for label in labels:
            grouped[label].append((regret, agreement))
    payload = {
        "version": "pwin-selector-stratification-3.9-v1",
        "dataset_role": "audit/validation; final untouched holdout is consumed",
        "epsilon": args.epsilon,
        "strata": {
            label: stats([value for value, _agreement in items], sum(agreement for _value, agreement in items))
            for label, items in sorted(grouped.items())
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
