from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np

from evaluate_frozen_policy import policy_score, q_value
from run_stage_a_oracle import load_module, state_from_row


def rows(path: Path) -> dict[str, dict[str, Any]]:
    sources = sorted(path.glob("*.jsonl.gz")) if path.is_dir() else [path]
    result: dict[str, dict[str, Any]] = {}
    for source in sources:
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                result[str(row["state_id"])] = row
    return result


def action_key(action: dict[str, Any]) -> tuple[str, int]:
    return str(action["action"]), int(action["slot"])


def tail(action: dict[str, Any]) -> float:
    return max(0.0, float(action["q_robust"]) - float(action.get("cvar") or 0.0))


def metric(values: list[float], agreements: int, runtimes: list[float]) -> dict[str, Any]:
    return {
        "states": len(values),
        "mean_regret": statistics.fmean(values),
        "median_regret": statistics.median(values),
        "p90_regret": float(np.quantile(values, 0.90)),
        "p95_regret": float(np.quantile(values, 0.95)),
        "gt_2pp": sum(value > 0.02 for value in values),
        "gt_5pp": sum(value > 0.05 for value in values),
        "oracle_agreement": agreements / len(values),
        "runtime_p50": statistics.median(runtimes),
        "runtime_p95": float(np.quantile(runtimes, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    canonical, runtime, oracle = rows(args.dataset), rows(args.runtime), rows(args.oracle)
    common = sorted(set(canonical) & set(runtime) & set(oracle))
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    strategy = module.CardStrategy(catalog)
    epsilons = (0.0005, 0.0010, 0.0020, 0.0030)
    tie_modes: dict[str, Callable[[dict[str, Any], Any], tuple[Any, ...]]] = {
        "risk_tail_tactical_policy": lambda action, state: (
            -float(action.get("p_lose_next_turn") or 0.0),
            -tail(action),
            float(action.get("p_win_within_2_own_actions") or 0.0),
            policy_score(strategy, state, action),
            -int(action["slot"]),
            1 if action["action"] == "turn" else 0,
        ),
        "risk_tail_tactical_stable": lambda action, _state: (
            -float(action.get("p_lose_next_turn") or 0.0),
            -tail(action),
            float(action.get("p_win_within_2_own_actions") or 0.0),
            -int(action["slot"]),
            1 if action["action"] == "turn" else 0,
        ),
        "policy_only": lambda action, state: (
            policy_score(strategy, state, action),
            -int(action["slot"]),
            1 if action["action"] == "turn" else 0,
        ),
    }
    results: dict[str, Any] = {}
    for epsilon in epsilons:
        for mode, tie_key in tie_modes.items():
            regrets: list[float] = []
            runtimes: list[float] = []
            agreements = equivalents = changed = 0
            for state_id in common:
                low, high, row = runtime[state_id], oracle[state_id], canonical[state_id]
                state = state_from_row(module, row)
                high_by_key = {action_key(action): action for action in high["actions"]}
                actions = [action for action in low["actions"] if action_key(action) in high_by_key]
                immediate = [action for action in actions if action.get("immediate_win")]
                pool = immediate or actions
                pwin_best = max(pool, key=lambda action: float(action["q_robust"]))
                equivalent = []
                for action in pool:
                    delta = float(pwin_best["q_robust"]) - float(action["q_robust"])
                    # q_se values share common random numbers, but saved Stage-A
                    # rows do not retain covariance.  This deliberately uses a
                    # conservative upper bound and is labelled as such.
                    se = math.sqrt(float(pwin_best.get("q_se") or 0.0) ** 2 + float(action.get("q_se") or 0.0) ** 2)
                    lower, upper = delta - 1.96 * se, delta + 1.96 * se
                    if lower >= -epsilon and upper <= epsilon:
                        equivalent.append(action)
                selected = max(equivalent or [pwin_best], key=lambda action: tie_key(action, state))
                equivalents += int(len(equivalent) > 1)
                changed += int(action_key(selected) != action_key(pwin_best))
                oracle_best = max(high["actions"], key=q_value)
                oracle_value = q_value(oracle_best)
                selected_value = q_value(high_by_key[action_key(selected)])
                regrets.append(max(0.0, oracle_value - selected_value))
                agreements += int(action_key(selected) == action_key(oracle_best))
                runtimes.append(float(low.get("runtime_seconds") or 0.0))
            key = f"epsilon_{epsilon * 100:.2f}pp__{mode}"
            results[key] = {
                **metric(regrets, agreements, runtimes),
                "practical_equivalence_states": equivalents,
                "selection_changed_vs_argmax_pwin": changed,
            }

    seed_domains = {
        "runtime_seed_formula": "sha256('stage-a:' + state_id + ':validation-runtime200')",
        "oracle_seed_formulas": [
            "sha256('stage-a:' + state_id + ':fv1')",
            "sha256('stage-a:' + state_id + ':fv2')",
            "sha256('stage-a:' + state_id + ':fv3')",
        ],
        "shared_samples": False,
        "runtime_estimate_reused_by_oracle": False,
        "future_real_cards_used": False,
        "oracle_definition": "mean of three independent 5000-particle estimates of the same expected-Pwin estimator",
    }
    payload = {
        "version": "practical-equivalence-validation-3.9-v1",
        "dataset_role": "audit/validation; not untouched holdout",
        "states": len(common),
        "ci_note": "conservative SE bound because saved Stage-A rows omit paired covariance",
        "independence": seed_domains,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
