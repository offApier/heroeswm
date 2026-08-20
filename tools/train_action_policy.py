from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_stage_a_oracle import Models, load_module, state_from_row, own_state, row_for_position
from train_opponent_policy import FEATURES as ACTION_BASE_NAMES, feature_vector as action_base_features


EXTRA_NAMES = (
    "post_state_pwin", "retained_playable", "retained_eta_best", "retained_eta_mean",
    "retained_producers", "retained_extra_turns", "retained_resource_congestion",
    "discard_finisher", "discard_producer", "discard_extra_turn",
    "play_x_short_horizon", "drop_x_short_horizon", "cost_x_short_horizon",
)
FEATURES = tuple(ACTION_BASE_NAMES) + EXTRA_NAMES


def read_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def eta(strategy: Any, card: Any, me: Any) -> int:
    value = strategy._turns_until_affordable(card, me)
    return 20 if value is None else min(20, value)


def action_features(module: Any, strategy: Any, models: Models, row: dict[str, Any], action: dict[str, Any]) -> np.ndarray:
    state = state_from_row(module, row)
    action_type, slot, card_id = action["action"], int(action["slot"]), int(action["card_id"])
    card = strategy.catalog[card_id]
    me1, enemy1 = (strategy.simulate(card, state) if action_type == "turn" else (state.me, state.opponent))
    retained = [value for index, value in enumerate(state.hand) if index != slot]
    post = own_state(module, state, me1, enemy1, retained)
    retained_cards = [strategy.catalog[value] for value in retained]
    etas = [eta(strategy, candidate, me1) for candidate in retained_cards]
    playable = sum(strategy._affordable(candidate, me1) for candidate in retained_cards)
    resources = Counter("ore" if candidate.ore else "mana" if candidate.mana else "army" if candidate.army else "free" for candidate in retained_cards)
    horizon = models.horizon
    base_row = row_for_position(row, post, strategy)
    # Horizon is already part of state value; expose its short-game interaction
    # explicitly so the ranker can price production tempo rather than learn a
    # fixed card bonus.
    from train_horizon_model import features as horizon_features
    hv = horizon_features(base_row, models.catalog_raw)
    z = (np.asarray(hv) - np.asarray(horizon["center"])) / np.asarray(horizon["scale"])
    predicted_horizon = float(np.r_[1.0, z] @ np.asarray(horizon["coefficients_with_intercept"]))
    if horizon["target_transform"] == "log1p":
        predicted_horizon = math.expm1(predicted_horizon)
    short = math.exp(-max(0, predicted_horizon) / 8)
    base = action_base_features(strategy, state, action_type, card_id).astype(np.float64)
    extra = np.asarray([
        models.pwin(row, post, strategy), playable / 5, min(etas, default=20) / 20,
        (sum(etas) / len(etas) if etas else 20) / 20,
        sum(candidate.id in strategy.PRODUCTION_CARDS for candidate in retained_cards) / 5,
        sum(candidate.id in strategy.EXTRA_TURN_CARDS for candidate in retained_cards) / 5,
        max(resources.values(), default=0) / 5,
        int(action_type == "drop" and (card.id in strategy.DIRECT_TOWER_DAMAGE or card.id in strategy.GENERAL_DAMAGE)),
        int(action_type == "drop" and card.id in strategy.PRODUCTION_CARDS),
        int(action_type == "drop" and card.id in strategy.EXTRA_TURN_CARDS),
        int(action_type == "turn") * short, int(action_type == "drop") * short,
        card.total_cost / 20 * short,
    ], dtype=np.float64)
    return np.r_[base, extra]


def fit_pairwise(x: np.ndarray, weights: np.ndarray, l2: float, epochs: int = 500) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    m, v = np.zeros_like(beta), np.zeros_like(beta)
    rng = np.random.default_rng(39101)
    probability = weights / weights.sum()
    batch = min(8192, len(x))
    for epoch in range(1, epochs + 1):
        idx = rng.choice(len(x), batch, replace=True, p=probability)
        xb = x[idx]
        pred = 1 / (1 + np.exp(-np.clip(xb @ beta, -30, 30)))
        grad = -((1 - pred)[:, None] * xb).mean(axis=0) + l2 * beta
        m, v = .9 * m + .1 * grad, .999 * v + .001 * grad * grad
        step = .03 * math.sqrt(1 - .999 ** epoch) / (1 - .9 ** epoch)
        beta -= step * m / (np.sqrt(v) + 1e-8)
    return beta


def evaluate(states: list[dict[str, Any]], features_by_state: dict[str, dict[tuple[str, int], np.ndarray]], beta: np.ndarray) -> dict[str, Any]:
    learned_regret, historical_regret = [], []
    agreement = 0
    for row in states:
        scores = []
        for action in row["actions"]:
            key = (action["action"], int(action["slot"]))
            scores.append((float(features_by_state[row["state_id"]][key] @ beta), action))
        selected = max(scores, key=lambda item: item[0])[1]
        oracle = max(row["actions"], key=lambda action: action["q_mean"])
        historical = next(action for action in row["actions"] if action["action"] == row["historical_choice"]["action"] and action["slot"] == row["historical_choice"]["slot"])
        learned_regret.append(max(0, oracle["q_mean"] - selected["q_mean"]))
        historical_regret.append(max(0, oracle["q_mean"] - historical["q_mean"]))
        agreement += int(selected["action"] == oracle["action"] and selected["slot"] == oracle["slot"])
    return {
        "states": len(states), "oracle_agreement": agreement / len(states),
        "mean_regret": float(np.mean(learned_regret)), "p90_regret": float(np.quantile(learned_regret, .9)),
        "p95_regret": float(np.quantile(learned_regret, .95)), "gt_5pp": sum(value > .05 for value in learned_regret),
        "historical_mean_regret": float(np.mean(historical_regret)), "historical_p90_regret": float(np.quantile(historical_regret, .9)),
        "historical_p95_regret": float(np.quantile(historical_regret, .95)), "historical_gt_5pp": sum(value > .05 for value in historical_regret),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("horizon_model", type=Path)
    parser.add_argument("value_model", type=Path)
    parser.add_argument("opponent_model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinement-oracle", type=Path)
    args = parser.parse_args()
    oracle_rows = list(read_gzip(args.oracle))
    if args.refinement_oracle:
        refinements = {row["state_id"]: row for row in read_gzip(args.refinement_oracle)}
        oracle_rows = [refinements.get(row["state_id"], row) for row in oracle_rows]
    wanted = {row["state_id"] for row in oracle_rows}
    canonical = {row["state_id"]: row for row in read_gzip(args.dataset) if row["state_id"] in wanted}
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    strategy = module.StrategicCardStrategy(catalog)
    catalog_raw = {int(card["id"]): card for card in json.loads((args.source / "cards_catalog.json").read_text(encoding="utf-8"))}
    models = Models(catalog_raw, json.loads(args.horizon_model.read_text(encoding="utf-8")), json.loads(args.value_model.read_text(encoding="utf-8")), json.loads(args.opponent_model.read_text(encoding="utf-8")))
    features_by_state = {}
    for index, oracle_row in enumerate(oracle_rows):
        row = canonical[oracle_row["state_id"]]
        features_by_state[row["state_id"]] = {
            (action["action"], int(action["slot"])): action_features(module, strategy, models, row, action)
            for action in oracle_row["actions"]
        }
        if (index + 1) % 100 == 0:
            print(json.dumps({"feature_states": index + 1}), flush=True)
    train = [row for row in oracle_rows if row["split"] == "train"]
    validation = [row for row in oracle_rows if row["split"] == "validation"]
    pairs, pair_weights = [], []
    game_counts = Counter(row["game_id"] for row in train)
    for row in train:
        actions = row["actions"]
        for left_index in range(len(actions)):
            for right_index in range(left_index + 1, len(actions)):
                left, right = actions[left_index], actions[right_index]
                delta = float(left["q_mean"] - right["q_mean"])
                if abs(delta) < .001:
                    continue
                winner, loser = (left, right) if delta > 0 else (right, left)
                x = features_by_state[row["state_id"]][(winner["action"], int(winner["slot"]))] - features_by_state[row["state_id"]][(loser["action"], int(loser["slot"]))]
                se = math.sqrt(float(left["q_se"]) ** 2 + float(right["q_se"]) ** 2)
                confidence = min(1.0, abs(delta) / max(.002, 1.96 * se))
                pairs.append(x)
                pair_weights.append(confidence * min(.10, abs(delta)) / game_counts[row["game_id"]])
    x = np.asarray(pairs)
    pair_weights = np.asarray(pair_weights)
    pair_weights /= pair_weights.mean()
    trials = []
    for l2 in (.0003, .001, .003, .01, .03):
        beta = fit_pairwise(x, pair_weights, l2)
        metric = evaluate(validation, features_by_state, beta)
        trials.append((metric["mean_regret"] + .25 * metric["p90_regret"], l2, beta, metric))
    _score, l2, beta, validation_metric = min(trials, key=lambda item: item[0])
    payload = {
        "version": "counterfactual-action-ranker-3.9-v1", "holdout_opened": False,
        "teacher": "three-seed 5000-particle robust oracle",
        "features": list(FEATURES), "weights": beta.tolist(), "l2": l2,
        "train_states": len(train), "validation_states": len(validation), "pairwise_rows": len(x),
        "train": evaluate(train, features_by_state, beta), "validation": validation_metric,
        "l2_trials": [{"l2": item[1], **item[3]} for item in trials],
        "no_hardcoded_card_bonus": True, "uncertain_pair_weighting": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "weights"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
