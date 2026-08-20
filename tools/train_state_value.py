from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from train_horizon_model import FEATURES as BASE_FEATURES, features as base_features


EXTRA_FEATURES = (
    "horizon_prediction", "horizon_short", "horizon_long",
    "our_tower_finish_pressure", "enemy_tower_finish_pressure",
    "our_destruction_pressure", "enemy_destruction_pressure",
    "our_resource_pressure", "enemy_resource_pressure",
    "tower_gap_x_short", "production_diff_x_horizon",
    "wall_value_normal_threat", "initiative_x_terminal", "turn_sqrt",
)
FEATURES = tuple(BASE_FEATURES) + EXTRA_FEATURES


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            # Never deserialize/use holdout state payloads during development.
            if row.get("split") != "holdout":
                yield row


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def horizon_prediction(base: list[float], model: dict[str, Any]) -> float:
    x = np.asarray(base, dtype=np.float64)
    z = (x - np.asarray(model["center"])) / np.asarray(model["scale"])
    coef = np.asarray(model["coefficients_with_intercept"])
    value = float(np.r_[1.0, z] @ coef)
    if model["target_transform"] == "log1p":
        value = math.expm1(value)
    return max(0.0, min(180.0, value))


def feature_vector(row: dict[str, Any], catalog: dict[int, dict[str, Any]], horizon: dict[str, Any]) -> list[float]:
    base = base_features(row, catalog)
    state = row["visible_state"]
    me, enemy = state["me"], state["opponent"]
    h = horizon_prediction(base, horizon)
    td_us, td_en = max(0, 50 - int(me["tower"])), max(0, 50 - int(enemy["tower"]))
    hp_us = int(me["tower"]) + 0.65 * min(12, int(me["wall"]))
    hp_en = int(enemy["tower"]) + 0.65 * min(12, int(enemy["wall"]))
    r_us = min(int(me[name]) for name in ("ore", "mana", "army"))
    r_en = min(int(enemy[name]) for name in ("ore", "mana", "army"))
    production_diff = sum(int(me[name]) for name in ("mine", "monastery", "barracks")) - sum(
        int(enemy[name]) for name in ("mine", "monastery", "barracks")
    )
    short = math.exp(-h / 8)
    terminal = 1 / (1 + min(td_us, td_en, int(me["tower"]), int(enemy["tower"])))
    initiative = 1 if row.get("initiative") == "us" else -1
    ordinary_threat_proxy = min(1.0, int(enemy.get("army") or 0) / 18)
    extra = [
        h / 80, short, 1 - short,
        math.exp(-td_us / 6), math.exp(-td_en / 6),
        math.exp(-hp_en / 11), math.exp(-hp_us / 11),
        math.exp(-(150 - min(150, r_us)) / 30), math.exp(-(150 - min(150, r_en)) / 30),
        (td_en - td_us) * short / 20,
        production_diff * min(h, 60) / 180,
        min(12, int(me["wall"])) * ordinary_threat_proxy / 12,
        initiative * terminal,
        math.sqrt(max(0, int(state.get("turn") or 0))) / 12,
    ]
    return base + extra


def fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray, l2: float, *, epochs: int = 360) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=np.float64)
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    rng = np.random.default_rng(39031)
    probability = weights / weights.sum()
    batch_size = min(8192, len(x))
    for epoch in range(1, epochs + 1):
        idx = rng.choice(len(x), size=batch_size, replace=True, p=probability)
        xb, yb = x[idx], y[idx]
        pred = sigmoid(xb @ beta)
        grad = xb.T @ (pred - yb) / len(idx)
        grad[1:] += l2 * beta[1:]
        m = .9 * m + .1 * grad
        v = .999 * v + .001 * grad * grad
        lr = .025 * math.sqrt(1 - .999 ** epoch) / (1 - .9 ** epoch)
        beta -= lr * m / (np.sqrt(v) + 1e-8)
    return beta


def fit_platt(logits: np.ndarray, y: np.ndarray, game_ids: list[int]) -> tuple[float, float]:
    counts = Counter(game_ids)
    weights = np.asarray([1 / counts[game_id] for game_id in game_ids], dtype=np.float64)
    weights /= weights.mean()
    design = np.column_stack([np.ones(len(logits)), logits])
    beta = np.asarray([0.0, 1.0])
    for _ in range(80):
        p = sigmoid(design @ beta)
        gradient = design.T @ (weights * (p - y))
        hessian = design.T @ ((weights * p * (1 - p))[:, None] * design) + np.eye(2) * 1e-5
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return float(beta[0]), float(beta[1])


def metrics(p: np.ndarray, y: np.ndarray, game_ids: list[int]) -> dict[str, Any]:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    bins = []
    ece = 0.0
    for left in np.linspace(0, 1, 11)[:-1]:
        right = left + .1
        mask = (p >= left) & ((p < right) if right < 1 else (p <= right))
        if not np.any(mask):
            continue
        predicted, observed, count = float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())
        ece += count / len(p) * abs(predicted - observed)
        bins.append({"left": left, "right": right, "count": count, "predicted": predicted, "observed": observed})
    per_game = defaultdict(list)
    for probability, target, game_id in zip(p, y, game_ids):
        per_game[game_id].append((float(probability), float(target)))
    game_mean = np.asarray([np.mean([item[0] for item in values]) for values in per_game.values()])
    game_target = np.asarray([values[0][1] for values in per_game.values()])
    return {
        "rows": len(p), "games": len(per_game),
        "brier_state": float(np.mean((p - y) ** 2)),
        "log_loss_state": float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))),
        "ece_state": ece,
        "brier_game_mean": float(np.mean((game_mean - game_target) ** 2)),
        "reliability": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("horizon_model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = {int(card["id"]): card for card in json.loads(args.catalog.read_text(encoding="utf-8"))}
    horizon = json.loads(args.horizon_model.read_text(encoding="utf-8"))
    data: dict[str, list[tuple[list[float], float, int]]] = defaultdict(list)
    for row in rows(args.dataset):
        if row.get("outcome") not in {"win", "loss"}:
            continue
        data[row["split"]].append((feature_vector(row, catalog, horizon), float(row["outcome"] == "win"), int(row["game_id"])))
    center = np.mean([item[0] for item in data["train"]], axis=0)
    scale = np.std([item[0] for item in data["train"]], axis=0)
    scale[scale < 1e-8] = 1.0
    def matrix(items):
        x = (np.asarray([item[0] for item in items]) - center) / scale
        return np.column_stack([np.ones(len(x)), x]), np.asarray([item[1] for item in items]), [item[2] for item in items]
    x_train, y_train, g_train = matrix(data["train"])
    val_cal_items, val_eval_items = [], []
    for item in data["validation"]:
        digest = hashlib.sha256(f"value-cal:{item[2]}".encode()).digest()[0]
        (val_cal_items if digest < 128 else val_eval_items).append(item)
    x_cal, y_cal, g_cal = matrix(val_cal_items)
    x_eval, y_eval, g_eval = matrix(val_eval_items)
    counts = Counter(g_train)
    game_weights = np.asarray([1 / counts[game_id] for game_id in g_train])
    game_weights /= game_weights.mean()
    trials = []
    for l2 in (0.0003, 0.001, 0.003, 0.01, 0.03):
        beta = fit_logistic(x_train, y_train, game_weights, l2)
        p_cal = sigmoid(x_cal @ beta)
        trial = metrics(p_cal, y_cal, g_cal)
        trials.append((trial["log_loss_state"], l2, beta, trial))
    _score, l2, beta, calibration_selection = min(trials, key=lambda item: item[0])
    platt_intercept, platt_slope = fit_platt(x_cal @ beta, y_cal, g_cal)
    def predict(x):
        return sigmoid(platt_intercept + platt_slope * (x @ beta))
    payload = {
        "version": "state-value-population-3.9-v1",
        "architecture": "game-balanced nonlinear logistic value with separately learned horizon",
        "future_information_used": False,
        "holdout_opened": False,
        "features": list(FEATURES),
        "center": center.tolist(), "scale": scale.tolist(),
        "coefficients_with_intercept": beta.tolist(),
        "l2": l2,
        "platt_intercept": platt_intercept, "platt_slope": platt_slope,
        "train": metrics(predict(x_train), y_train, g_train),
        "validation_calibration_half": metrics(predict(x_cal), y_cal, g_cal),
        "validation_assessment_half": metrics(predict(x_eval), y_eval, g_eval),
        "l2_trials_on_calibration_half": [
            {"l2": item[1], **{key: value for key, value in item[3].items() if key != "reliability"}}
            for item in trials
        ],
        "notes": [
            "Final game outcome is an offline target only and never an online feature.",
            "Each training game has equal total weight; validation is split by game_id into calibration and assessment halves.",
            "Holdout state rows are skipped before feature extraction.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"center", "scale", "coefficients_with_intercept"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
