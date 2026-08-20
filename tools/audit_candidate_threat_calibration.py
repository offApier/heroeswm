from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_frozen_policy import policy_score
from run_stage_a_oracle import load_module, state_from_row

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from audit_card_dataset import decision_key, load_segments, merge_events, parsed_move  # noqa: E402


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def state_id(game_id: int, event: dict[str, Any]) -> str:
    payload = json.dumps([game_id, decision_key(event)], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def metrics(predictions: list[float], outcomes: list[int]) -> dict[str, Any]:
    p, y = np.asarray(predictions), np.asarray(outcomes)
    bins, ece = [], 0.0
    edges = (0, .05, .10, .20, .40, .60, .80, .95, 1.000001)
    for left, right in zip(edges, edges[1:]):
        mask = (p >= left) & (p < right)
        if not mask.any():
            continue
        predicted, observed, count = float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())
        ece += count / len(p) * abs(predicted - observed)
        bins.append({"left": left, "right": min(1, right), "count": count, "predicted": predicted, "observed": observed})
    exact = p == 1
    return {
        "states": len(p), "brier": float(np.mean((p - y) ** 2)), "ece": ece,
        "exact_100_claims": int(exact.sum()),
        "exact_100_observed_loss_rate": float(y[exact].mean()) if exact.any() else None,
        "buckets": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("card_games", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = {row["state_id"]: row for row in rows(args.dataset)}
    oracle = {row["state_id"]: row for row in rows(args.oracle)}
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    strategy = module.CardStrategy(catalog)
    candidate_prediction = {}
    for identifier, teacher in oracle.items():
        row = canonical[identifier]
        state = state_from_row(module, row)
        selected = max(teacher["actions"], key=lambda action: policy_score(strategy, state, action))
        candidate_prediction[identifier] = float(selected["p_lose_next_turn"])
    candidate_p, baseline_p, actual = [], [], []
    game_labels: list[int] = []
    games = load_segments(args.card_games)
    for game_id, segments in games.items():
        events, _artifacts, _duplicates = merge_events(segments)
        seen = set()
        for index, event in enumerate(events):
            if not event.get("selected"):
                continue
            key = decision_key(event)
            if key in seen:
                continue
            seen.add(key)
            identifier = state_id(game_id, event)
            if identifier not in candidate_prediction:
                continue
            next_opponent = None
            for future in events[index + 1 :]:
                if str(future.get("actor")) == "opponent" and parsed_move(future) is not None:
                    next_opponent = future
                    break
            loss = 0
            if next_opponent is not None:
                after = next_opponent.get("after") or {}
                player_no = int(after.get("player_no") or 1)
                winner = int(after.get("winner") or 0)
                loss = int(bool(int(after.get("finish_reason") or 0) and winner not in {0, player_no}))
            baseline = float((event.get("selected") or {}).get("p_lose_next_turn") or 0.0)
            candidate_p.append(candidate_prediction[identifier])
            baseline_p.append(baseline)
            actual.append(loss)
            game_labels.append(game_id)
    grouped: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    for game_id, candidate, baseline, observed in zip(game_labels, candidate_p, baseline_p, actual):
        grouped[game_id].append((candidate, baseline, observed))
    game_ids = sorted(grouped)
    rng = np.random.default_rng(39002026)
    bootstrap_delta = []
    if game_ids:
        for _ in range(5000):
            sampled = rng.choice(game_ids, size=len(game_ids), replace=True)
            candidate_error = baseline_error = count = 0.0
            for game_id in sampled:
                for candidate, baseline, observed in grouped[int(game_id)]:
                    candidate_error += (candidate - observed) ** 2
                    baseline_error += (baseline - observed) ** 2
                    count += 1
            bootstrap_delta.append((candidate_error - baseline_error) / max(1.0, count))
    payload = {
        "version": "candidate-threat-calibration-3.9-v1",
        "split": sorted({row["split"] for row in oracle.values()}),
        "candidate": metrics(candidate_p, actual), "historical_model": metrics(baseline_p, actual),
        "improvement": {
            "brier": metrics(baseline_p, actual)["brier"] - metrics(candidate_p, actual)["brier"],
            "ece": metrics(baseline_p, actual)["ece"] - metrics(candidate_p, actual)["ece"],
        },
        "game_level_paired_bootstrap": {
            "games": len(game_ids),
            "resamples": len(bootstrap_delta),
            "delta_definition": "candidate_brier_minus_historical_brier",
            "observed_delta": metrics(candidate_p, actual)["brier"] - metrics(baseline_p, actual)["brier"],
            "ci95": [float(np.quantile(bootstrap_delta, .025)), float(np.quantile(bootstrap_delta, .975))] if bootstrap_delta else None,
            "probability_candidate_worse": float(np.mean(np.asarray(bootstrap_delta) > 0)) if bootstrap_delta else None,
            "statistically_significant_at_95_percent": bool(
                bootstrap_delta and (
                    np.quantile(bootstrap_delta, .025) > 0 or np.quantile(bootstrap_delta, .975) < 0
                )
            ),
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
