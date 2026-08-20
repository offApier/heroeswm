from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from run_stage_a_oracle import Models, load_module, own_state, row_for_position, state_from_row
from train_action_policy import FEATURES, action_features
from train_state_value import FEATURES as VALUE_FEATURES, feature_vector as value_features


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("horizon", type=Path)
    parser.add_argument("value", type=Path)
    parser.add_argument("opponent", type=Path)
    parser.add_argument("action", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=300)
    args = parser.parse_args()
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    strategy = module.StrategicCardStrategy(catalog)
    runtime = module.PolicyRuntime.load(args.source / "policy_models.json")
    if runtime is None:
        raise RuntimeError("runtime model bundle did not load")
    catalog_raw = {int(card["id"]): card for card in json.loads((args.source / "cards_catalog.json").read_text(encoding="utf-8"))}
    models = Models(catalog_raw, json.loads(args.horizon.read_text(encoding="utf-8")), json.loads(args.value.read_text(encoding="utf-8")), json.loads(args.opponent.read_text(encoding="utf-8")))
    weights = np.asarray(json.loads(args.action.read_text(encoding="utf-8"))["weights"])
    checked = 0
    differences = []
    ranking_mismatches = 0
    feature_differences = []
    worst_feature = None
    worst_state_feature = None
    for row in rows(args.dataset):
        if row["split"] == "holdout":
            continue
        state = state_from_row(module, row)
        tool_scores, runtime_scores = [], []
        for action in row["legal_actions"]:
            expected_vector = action_features(module, strategy, models, row, action)
            actual_vector = np.asarray(runtime.action_feature_vector(strategy, state, action["action"], int(action["slot"])))
            card = strategy.catalog[action["card_id"]]
            me1, enemy1 = strategy.simulate(card, state) if action["action"] == "turn" else (state.me, state.opponent)
            retained = [card_id for index, card_id in enumerate(state.hand) if index != int(action["slot"])]
            post = own_state(module, state, me1, enemy1, retained)
            expected_state_vector = np.asarray(value_features(row_for_position(row, post, strategy), models.catalog_raw, models.horizon))
            actual_state_vector = np.asarray(runtime.state_feature_vector(strategy, post, me1, enemy1, retained))
            state_delta = np.abs(expected_state_vector - actual_state_vector)
            state_index = int(np.argmax(state_delta))
            if worst_state_feature is None or state_delta[state_index] > worst_state_feature["difference"]:
                worst_state_feature = {"index": state_index, "name": VALUE_FEATURES[state_index], "difference": float(state_delta[state_index]), "expected": float(expected_state_vector[state_index]), "actual": float(actual_state_vector[state_index]), "state_id": row["state_id"], "action": action}
            delta_vector = np.abs(expected_vector - actual_vector)
            feature_differences.extend(delta_vector.tolist())
            feature_index = int(np.argmax(delta_vector))
            if worst_feature is None or delta_vector[feature_index] > worst_feature["difference"]:
                worst_feature = {"index": feature_index, "name": FEATURES[feature_index], "difference": float(delta_vector[feature_index]), "expected": float(expected_vector[feature_index]), "actual": float(actual_vector[feature_index]), "expected_length": len(expected_vector), "actual_length": len(actual_vector), "state_id": row["state_id"], "action": action}
            expected = float(expected_vector @ weights)
            actual = float(actual_vector @ weights)
            differences.append(abs(expected - actual))
            tool_scores.append(expected)
            runtime_scores.append(actual)
        ranking_mismatches += int(int(np.argmax(tool_scores)) != int(np.argmax(runtime_scores)))
        checked += 1
        if checked >= args.states:
            break
    payload = {
        "version": "runtime-feature-parity-3.9-v1", "holdout_opened": False,
        "states": checked, "actions": len(differences), "max_abs_score_difference": max(differences),
        "mean_abs_score_difference": float(np.mean(differences)), "ranking_mismatches": ranking_mismatches,
        "max_abs_feature_difference": max(feature_differences), "worst_feature": worst_feature,
        "worst_state_feature": worst_state_feature,
        "pass": max(differences) < 1e-6 and max(feature_differences) < 1e-6 and ranking_mismatches == 0,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
