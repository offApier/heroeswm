from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


FEATURES = [
    "turn", "our_tower", "enemy_tower", "our_wall", "enemy_wall",
    "tower_distance_ours", "tower_distance_enemy", "destroy_distance_ours", "destroy_distance_enemy",
    "tower_distance_min", "tower_race_diff", "tower_distance_ours_sq", "tower_distance_enemy_sq",
    "our_ore", "our_mana", "our_army", "enemy_ore", "enemy_mana", "enemy_army",
    "resource_sum_ours", "resource_sum_enemy", "resource_min_ours", "resource_min_enemy",
    "resource_diff", "our_mine", "our_monastery", "our_barracks",
    "enemy_mine", "enemy_monastery", "enemy_barracks", "production_sum_ours",
    "production_sum_enemy", "production_diff", "playable_count", "dead_count",
    "hand_eta_mean", "hand_eta_best", "same_resource_max", "must_discard",
    "first_mover", "reconnect", "unknown_transitions", "terminal_pressure",
    "wall_fraction_ours", "wall_fraction_enemy",
]


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def costs(catalog: dict[int, dict[str, Any]], card_id: int) -> tuple[str, int]:
    card = catalog[card_id]
    if int(card.get("ore") or 0):
        return "ore", int(card["ore"])
    if int(card.get("mana") or 0):
        return "mana", int(card["mana"])
    return "army", int(card.get("army") or 0)


def eta(card: dict[str, Any], me: dict[str, Any]) -> int:
    resource, cost = costs({int(card["id"]): card}, int(card["id"]))
    current = int(me.get(resource) or 0)
    if current >= cost:
        return 0
    producer = {"ore": "mine", "mana": "monastery", "army": "barracks"}[resource]
    income = int(me.get(producer) or 0)
    return 99 if income <= 0 else math.ceil((cost - current) / income)


def features(row: dict[str, Any], catalog: dict[int, dict[str, Any]]) -> list[float]:
    state = row["visible_state"]
    me, enemy = state["me"], state["opponent"]
    tower_o, tower_e = int(me["tower"]), int(enemy["tower"])
    td_o, td_e = max(0, 50 - tower_o), max(0, 50 - tower_e)
    resources_o = [int(me[name]) for name in ("ore", "mana", "army")]
    resources_e = [int(enemy[name]) for name in ("ore", "mana", "army")]
    producers_o = [int(me[name]) for name in ("mine", "monastery", "barracks")]
    producers_e = [int(enemy[name]) for name in ("mine", "monastery", "barracks")]
    etas = [min(12, eta(catalog[int(card_id)], me)) for card_id in row["our_hand"]]
    resource_types = Counter(costs(catalog, int(card_id))[0] for card_id in row["our_hand"])
    playable = sum(action["action"] == "turn" for action in row["legal_actions"])
    unknown = int(row.get("reconnect_unknown_transitions_before") or 0)
    values = {
        "turn": min(160, int(state.get("turn") or 0)),
        "our_tower": tower_o, "enemy_tower": tower_e,
        "our_wall": int(me["wall"]), "enemy_wall": int(enemy["wall"]),
        "tower_distance_ours": td_o, "tower_distance_enemy": td_e,
        "destroy_distance_ours": tower_o, "destroy_distance_enemy": tower_e,
        "tower_distance_min": min(td_o, td_e, tower_o, tower_e),
        "tower_race_diff": td_e - td_o,
        "tower_distance_ours_sq": td_o * td_o / 50,
        "tower_distance_enemy_sq": td_e * td_e / 50,
        "our_ore": resources_o[0], "our_mana": resources_o[1], "our_army": resources_o[2],
        "enemy_ore": resources_e[0], "enemy_mana": resources_e[1], "enemy_army": resources_e[2],
        "resource_sum_ours": sum(resources_o), "resource_sum_enemy": sum(resources_e),
        "resource_min_ours": min(resources_o), "resource_min_enemy": min(resources_e),
        "resource_diff": sum(resources_o) - sum(resources_e),
        "our_mine": producers_o[0], "our_monastery": producers_o[1], "our_barracks": producers_o[2],
        "enemy_mine": producers_e[0], "enemy_monastery": producers_e[1], "enemy_barracks": producers_e[2],
        "production_sum_ours": sum(producers_o), "production_sum_enemy": sum(producers_e),
        "production_diff": sum(producers_o) - sum(producers_e),
        "playable_count": playable, "dead_count": 6 - playable,
        "hand_eta_mean": sum(etas) / len(etas), "hand_eta_best": min(etas),
        "same_resource_max": max(resource_types.values()),
        "must_discard": int(bool(state.get("must_discard"))),
        "first_mover": int(row.get("initiative") == "us"),
        "reconnect": int(bool(row.get("reconnect"))),
        "unknown_transitions": min(10, unknown),
        "terminal_pressure": 1.0 / (1.0 + min(td_o, td_e, tower_o, tower_e)),
        "wall_fraction_ours": int(me["wall"]) / max(1, tower_o + int(me["wall"])),
        "wall_fraction_enemy": int(enemy["wall"]) / max(1, tower_e + int(enemy["wall"])),
    }
    return [float(values[name]) for name in FEATURES]


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.abs(prediction - target)
    return {
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "median_ae": float(np.median(error)),
        "p90_ae": float(np.quantile(error, 0.9)),
        "bias": float(np.mean(prediction - target)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("game_index", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = {int(card["id"]): card for card in json.loads(args.catalog.read_text(encoding="utf-8"))}
    game_index = json.loads(args.game_index.read_text(encoding="utf-8"))
    game_info = {int(row["game_id"]): row for row in game_index["game_rows"]}
    data = {"train": [], "validation": []}
    for row in rows(args.dataset):
        split = row["split"]
        if split not in data or row["outcome"] not in {"win", "loss", "draw"}:
            continue
        remaining = max(0, int(game_info[row["game_id"]]["events"]) - int(row["card_action_index"]))
        data[split].append((features(row, catalog), math.log1p(remaining), row["game_id"], remaining))

    x_train = np.asarray([item[0] for item in data["train"]], dtype=np.float64)
    y_train_log = np.asarray([item[1] for item in data["train"]], dtype=np.float64)
    y_train_raw = np.asarray([item[3] for item in data["train"]], dtype=np.float64)
    x_val = np.asarray([item[0] for item in data["validation"]], dtype=np.float64)
    y_val_real = np.asarray([item[3] for item in data["validation"]], dtype=np.float64)
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z_train = (x_train - center) / scale
    z_val = (x_val - center) / scale
    z_train = np.column_stack([np.ones(len(z_train)), z_train])
    z_val = np.column_stack([np.ones(len(z_val)), z_val])
    # Each complete game contributes equal total weight.
    counts = Counter(item[2] for item in data["train"])
    weights = np.asarray([1.0 / counts[item[2]] for item in data["train"]], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    best = None
    trials = []
    for transform, target in (("raw", y_train_raw), ("log1p", y_train_log)):
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            regularizer = np.eye(z_train.shape[1]) * alpha
            regularizer[0, 0] = 0.0
            coef = np.linalg.solve(z_train.T @ (weights[:, None] * z_train) + regularizer, z_train.T @ (weights * target))
            linear = z_val @ coef
            prediction = np.maximum(0.0, np.expm1(linear) if transform == "log1p" else linear)
            result = metrics(prediction, y_val_real)
            trials.append({"transform": transform, "alpha": alpha, **result})
            selection_score = result["mae"] + 0.10 * result["p90_ae"]
            if best is None or selection_score < best[0]:
                best = (selection_score, transform, alpha, coef, result)
    assert best is not None
    _score, transform, alpha, coef, validation = best
    baseline_prediction = np.full_like(y_val_real, np.median([item[3] for item in data["train"]]))
    payload = {
        "version": "horizon-ridge-v2",
        "trained_on": "train game_ids only",
        "selected_on": "validation game_ids only",
        "holdout_opened": False,
        "target": "remaining public card actions; target used only for offline fit",
        "feature_names": FEATURES,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficients_with_intercept": coef.tolist(),
        "target_transform": transform,
        "alpha": alpha,
        "train_rows": len(x_train),
        "validation_rows": len(x_val),
        "validation": validation,
        "constant_median_baseline": metrics(baseline_prediction, y_val_real),
        "alpha_trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("version", "alpha", "train_rows", "validation_rows", "validation", "constant_median_baseline")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
