from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def rows(paths: list[Path]):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--stage-c-selection", type=Path)
    parser.add_argument("--stage-c-count", type=int, default=100)
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows(args.parts):
        grouped[row["state_id"]].append(row)
    merged = []
    unstable = 0
    for state_id, seeds in grouped.items():
        action_samples: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in seeds:
            for action in row["actions"]:
                action_samples[(action["action"], action["slot"], action["card_id"])].append(action)
        actions = []
        for (action_type, slot, card_id), samples in action_samples.items():
            q_values = [float(item["q_robust"]) for item in samples]
            within_variance = statistics.fmean(float(item["q_se"]) ** 2 for item in samples)
            between_variance = statistics.pvariance(q_values) if len(q_values) > 1 else 0.0
            total_se = math.sqrt(within_variance / max(1, len(samples)) + between_variance / max(1, len(samples)))
            mean = statistics.fmean(q_values)
            actions.append({
                "action": action_type, "slot": slot, "card_id": card_id,
                "q_mean": mean, "q_se": total_se,
                "ci95": [max(0.0, mean - 1.96 * total_se), min(1.0, mean + 1.96 * total_se)],
                "q_empirical": statistics.fmean(float(item["q_empirical"]) for item in samples),
                "q_adversarial": statistics.fmean(float(item["q_adversarial"]) for item in samples),
                "cvar": statistics.fmean(float(item["cvar"]) for item in samples),
                "p_lose_next_turn": statistics.fmean(float(item["p_lose_next_turn"]) for item in samples),
                "p_win_within_2_own_actions": statistics.fmean(float(item["p_win_within_2_own_actions"]) for item in samples),
                "seeds": len(samples), "seed_values": q_values,
            })
        actions.sort(key=lambda item: item["q_mean"], reverse=True)
        first = seeds[0]
        historical = first["historical_choice"]
        chosen = next(item for item in actions if item["action"] == historical["action"] and item["slot"] == historical["slot"])
        margin = actions[0]["q_mean"] - actions[1]["q_mean"] if len(actions) > 1 else 1.0
        delta_se = math.sqrt(actions[0]["q_se"] ** 2 + (actions[1]["q_se"] ** 2 if len(actions) > 1 else 0))
        uncertain = margin - 1.96 * delta_se <= 0
        unstable += int(uncertain)
        merged.append({
            "state_id": state_id, "game_id": first["game_id"], "split": first["split"],
            "decision_index": first["decision_index"], "card_action_index": first["card_action_index"],
            "phase": first["phase"], "outcome": first["outcome"], "initiative": first["initiative"],
            "selection": first.get("selection"), "historical_choice": historical,
            "oracle_best": {key: actions[0][key] for key in ("action", "slot", "card_id")},
            "historical_regret": max(0.0, actions[0]["q_mean"] - chosen["q_mean"]),
            "decision_margin": margin, "se_diff": delta_se,
            "ci_diff": [margin - 1.96 * delta_se, margin + 1.96 * delta_se],
            "uncertain_label": uncertain, "particles_per_seed": first["particles_completed"],
            "seeds": len(seeds), "actions": actions,
        })
    merged.sort(key=lambda row: (row["game_id"], row["decision_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", newline="\n") as output:
        for row in merged:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    regrets = [row["historical_regret"] for row in merged]
    summary = {
        "version": "multiseed-oracle-3.9-v1", "holdout_opened": False,
        "states": len(merged), "random_representative": sum(row["selection"] == "random_representative" for row in merged),
        "suspicious": sum(row["selection"] == "suspicious" for row in merged),
        "uncertain_labels": unstable,
        "regret": {"mean": statistics.fmean(regrets), "median": statistics.median(regrets), "p90": float(np.quantile(regrets, .9)), "p95": float(np.quantile(regrets, .95)), "gt_5pp": sum(value > .05 for value in regrets)},
        "seed_ranking_disagreement": sum(len({max(seed["actions"], key=lambda action: action["q_robust"])["action"] + ":" + str(max(seed["actions"], key=lambda action: action["q_robust"])["slot"]) for seed in grouped[row["state_id"]]}) > 1 for row in merged),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.stage_c_selection:
        # Mix high regret and high uncertainty; deterministic hash breaks ties.
        ordered = sorted(merged, key=lambda row: (row["historical_regret"] + 2 * row["se_diff"], hashlib.sha256(row["state_id"].encode()).hexdigest()), reverse=True)
        selected = [{"state_id": row["state_id"], "game_id": row["game_id"], "card_action_index": row["card_action_index"], "selection": "stage_c_high_regret_or_uncertainty"} for row in ordered[: args.stage_c_count]]
        args.stage_c_selection.write_text(json.dumps({"version": "stage-c-selection-v1", "holdout_opened": False, "states": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
