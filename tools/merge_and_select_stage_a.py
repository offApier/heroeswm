from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    rows.sort(key=lambda row: (row["game_id"], row["decision_index"], row["state_id"]))
    if len({row["state_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate state_id across Stage A shards")
    return rows


def deterministic_order(row: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(f"{salt}:{row['state_id']}".encode()).hexdigest()


def stratified_random(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["phase"], row["outcome"], row["initiative"], row["historical_choice"]["action"])
        strata[key].append(row)
    for values in strata.values():
        values.sort(key=lambda row: deterministic_order(row, "stage-b-random-v1"))
    selected = []
    active = sorted(strata)
    while len(selected) < count and active:
        next_active = []
        for key in active:
            if strata[key] and len(selected) < count:
                selected.append(strata[key].pop())
            if strata[key]:
                next_active.append(key)
        active = next_active
    return selected


def suspicious_priority(row: dict[str, Any]) -> tuple[float, str]:
    actions = row["actions"]
    best = actions[0]
    historical = row["historical_choice"]
    disagreement = best["action"] != historical["action"] or best["slot"] != historical["slot"]
    close = float(row["decision_margin"]) <= .01
    tactical = row["phase"] == "terminal_race" or max(action["p_lose_next_turn"] for action in actions) >= .15
    cross_type = any(action["action"] == "turn" for action in actions) and any(action["action"] == "drop" for action in actions)
    score = (
        10 * float(row["historical_regret"])
        + 1.5 * disagreement + .8 * close + .8 * tactical
        + .5 * cross_type + .4 * (historical["action"] == "drop")
    )
    return score, deterministic_order(row, "stage-b-suspicious-v1")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    regrets = [float(row["historical_regret"]) for row in rows]
    buckets = Counter()
    for value in regrets:
        label = "le_0.25pp" if value <= .0025 else "le_0.5pp" if value <= .005 else "le_1pp" if value <= .01 else "1_2pp" if value <= .02 else "2_5pp" if value <= .05 else "5_10pp" if value <= .10 else "gt_10pp"
        buckets[label] += 1
    strata = defaultdict(list)
    for row in rows:
        for key in (
            f"split:{row['split']}", f"phase:{row['phase']}", f"outcome:{row['outcome']}",
            f"initiative:{row['initiative']}", f"action:{row['historical_choice']['action']}",
        ):
            strata[key].append(float(row["historical_regret"]))
    return {
        "states": len(rows), "legal_actions": sum(len(row["actions"]) for row in rows),
        "disagreements": sum(row["stage_a_best"]["action"] != row["historical_choice"]["action"] or row["stage_a_best"]["slot"] != row["historical_choice"]["slot"] for row in rows),
        "regret": {
            "mean": statistics.fmean(regrets), "median": statistics.median(regrets),
            "p90": float(np.quantile(regrets, .9)), "p95": float(np.quantile(regrets, .95)),
            "buckets": dict(buckets),
        },
        "strata": {
            key: {"count": len(values), "mean": statistics.fmean(values), "p90": float(np.quantile(values, .9))}
            for key, values in sorted(strata.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--random-count", type=int, default=400)
    parser.add_argument("--suspicious-count", type=int, default=400)
    args = parser.parse_args()
    rows = load_rows(args.parts)
    args.merged.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.merged, "wt", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    random_rows = stratified_random(rows, args.random_count)
    random_ids = {row["state_id"] for row in random_rows}
    suspicious = [row for row in sorted(rows, key=suspicious_priority, reverse=True) if row["state_id"] not in random_ids][
        : args.suspicious_count
    ]
    selected = [{"state_id": row["state_id"], "game_id": row["game_id"], "card_action_index": row["card_action_index"], "selection": "random_representative"} for row in random_rows]
    selected += [{"state_id": row["state_id"], "game_id": row["game_id"], "card_action_index": row["card_action_index"], "selection": "suspicious"} for row in suspicious]
    args.selection.write_text(json.dumps({"version": "stage-b-selection-v1", "holdout_opened": False, "states": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"version": "stage-a-merged-summary-v1", "holdout_opened": False, **summarize(rows), "stage_b_selection": Counter(item["selection"] for item in selected)}
    summary["stage_b_selection"] = dict(summary["stage_b_selection"])
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
