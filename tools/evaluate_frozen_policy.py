from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_stage_a_oracle import load_module, state_from_row


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def q_value(action: dict[str, Any], name: str = "q_mean") -> float:
    if name in action:
        return float(action[name])
    fallback = {"q_mean": "q_robust", "q_empirical": "q_empirical", "q_adversarial": "q_adversarial"}[name]
    return float(action[fallback])


def policy_score(strategy: Any, state: Any, action: dict[str, Any]) -> float:
    action_type, slot, card_id = action["action"], int(action["slot"]), int(action["card_id"])
    card = strategy.catalog[card_id]
    score = strategy.policy_runtime.action_score(strategy, state, action_type, slot)
    score -= (4.0 if state.me.tower <= 10 else .75) * float(action["p_lose_next_turn"])
    if action_type == "drop":
        retention = strategy._retention_probability_loss(card, state) * 100
        score -= 2 * retention
        score -= .15 * strategy._card_effect_strength(card, state.me, state.opponent, state)
    else:
        me1, enemy1 = strategy.simulate(card, state)
        if strategy._lost(me1, enemy1, state):
            return -1_000_000
        gain = sum(max(0, getattr(me1, field) - getattr(state.me, field)) for field in strategy.PRODUCTION_FIELDS)
        if gain:
            horizon = strategy.policy_runtime.horizon(strategy, state, state.me, state.opponent, state.hand)
            score += .015 * min(40, horizon) * gain * (1 - float(action["p_lose_next_turn"]))
    return score


def metric(regrets: list[float]) -> dict[str, Any]:
    return {
        "states": len(regrets), "mean_regret": statistics.fmean(regrets),
        "median_regret": statistics.median(regrets), "p90_regret": float(np.quantile(regrets, .9)),
        "p95_regret": float(np.quantile(regrets, .95)),
        "gt_2pp": sum(value > .02 for value in regrets),
        "gt_5pp": sum(value > .05 for value in regrets),
        "gt_10pp": sum(value > .10 for value in regrets),
    }


def calibration(probabilities: list[float], outcomes: list[float]) -> dict[str, Any]:
    p, y = np.asarray(probabilities), np.asarray(outcomes)
    bins, ece = [], 0.0
    for left in np.arange(0, 1, .1):
        mask = (p >= left) & ((p < left + .1) if left < .9 else (p <= 1))
        if not mask.any():
            continue
        predicted, observed, count = float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())
        ece += count / len(p) * abs(predicted - observed)
        bins.append({"left": float(left), "right": float(left + .1), "count": count, "predicted": predicted, "observed": observed})
    return {"brier": float(np.mean((p - y) ** 2)), "ece": ece, "reliability": bins}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = {row["state_id"]: row for row in rows(args.dataset)}
    oracle_rows = list(rows(args.oracle))
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    strategy = module.CardStrategy(catalog)
    strategy.MAX_PARTICLES = 200
    candidate_regrets, historical_regrets = [], []
    empirical_regrets, adversarial_regrets = [], []
    predicted, outcomes = [], []
    strata: dict[str, list[float]] = defaultdict(list)
    agreements = Counter()
    per_card: dict[int, dict[str, Any]] = defaultdict(lambda: {"selected": 0, "regrets": []})
    per_type: dict[str, list[float]] = defaultdict(list)
    per_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for oracle in oracle_rows:
        row = canonical[oracle["state_id"]]
        state = state_from_row(module, row)
        scored = [(policy_score(strategy, state, action), action) for action in oracle["actions"]]
        selected = max(scored, key=lambda item: item[0])[1]
        best = max(oracle["actions"], key=lambda action: q_value(action))
        historical = next(action for action in oracle["actions"] if action["action"] == oracle["historical_choice"]["action"] and action["slot"] == oracle["historical_choice"]["slot"])
        empirical = max(oracle["actions"], key=lambda action: q_value(action, "q_empirical"))
        adversarial = max(oracle["actions"], key=lambda action: q_value(action, "q_adversarial"))
        regret = max(0, q_value(best) - q_value(selected))
        historical_regret = max(0, q_value(best) - q_value(historical))
        candidate_regrets.append(regret)
        historical_regrets.append(historical_regret)
        empirical_regrets.append(max(0, q_value(best) - q_value(empirical)))
        adversarial_regrets.append(max(0, q_value(best) - q_value(adversarial)))
        agreements["candidate_oracle"] += int(selected["action"] == best["action"] and selected["slot"] == best["slot"])
        agreements["historical_oracle"] += int(historical["action"] == best["action"] and historical["slot"] == best["slot"])
        outcome = 1.0 if row["outcome"] == "win" else 0.5 if row["outcome"] == "draw" else 0.0
        predicted.append(q_value(selected))
        outcomes.append(outcome)
        for key in (f"phase:{row['phase']}", f"initiative:{row['initiative']}", f"outcome:{row['outcome']}", f"action:{selected['action']}"):
            strata[key].append(regret)
        per_card[int(selected["card_id"])]["selected"] += 1
        per_card[int(selected["card_id"])]["regrets"].append(regret)
        card = catalog[int(selected["card_id"])]
        action_types = []
        if selected["action"] == "drop": action_types.append("discard")
        if card.id in strategy.PRODUCTION_CARDS: action_types.append("production")
        if card.id in strategy.EXTRA_TURN_CARDS: action_types.append("extra_turn")
        if card.id in strategy.DIRECT_TOWER_DAMAGE: action_types.append("direct_tower_damage")
        if card.id in strategy.GENERAL_DAMAGE: action_types.append("normal_damage")
        if not action_types: action_types.append("other")
        for name in action_types:
            per_type[name].append(regret)
        per_game[int(row["game_id"])].append({"regret": regret, "historical_regret": historical_regret, "turn": row["card_action_index"], "outcome": row["outcome"]})
    game_audit = []
    for game_id, values in per_game.items():
        regrets = [item["regret"] for item in values]
        historical = [item["historical_regret"] for item in values]
        maximum = max(regrets)
        game_audit.append({
            "game_id": game_id, "outcome": values[0]["outcome"], "decisions": len(values),
            "candidate_largest_regret": maximum, "candidate_cumulative_regret": sum(regrets),
            "historical_largest_regret": max(historical), "historical_cumulative_regret": sum(historical),
            "classification": "A_likely_avoidable" if maximum > .10 else "B_partially_avoidable" if maximum > .05 else "C_no_major_policy_regret_found",
        })
    payload = {
        "version": "frozen-policy-evaluation-3.9-v1", "states": len(oracle_rows),
        "split": sorted({row["split"] for row in oracle_rows}),
        "candidate": metric(candidate_regrets), "historical": metric(historical_regrets),
        "empirical_only": metric(empirical_regrets), "perfect_adversarial_only": metric(adversarial_regrets),
        "agreements": {key: value / len(oracle_rows) for key, value in agreements.items()},
        "calibration": calibration(predicted, outcomes),
        "strata": {key: metric(values) for key, values in sorted(strata.items())},
        "per_card": {str(card_id): {"selected": item["selected"], "mean_regret": statistics.fmean(item["regrets"]), "p90_regret": float(np.quantile(item["regrets"], .9))} for card_id, item in sorted(per_card.items())},
        "per_action_type": {name: metric(values) for name, values in sorted(per_type.items())},
        "per_game": game_audit,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"per_card", "per_game", "strata", "calibration"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
