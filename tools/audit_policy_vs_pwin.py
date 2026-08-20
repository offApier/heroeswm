from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_frozen_policy import policy_score, q_value
from run_stage_a_oracle import load_module, state_from_row


def read_rows(path: Path) -> list[dict[str, Any]]:
    paths = (
        sorted(path.glob("final_holdout_fh1_part_*.jsonl.gz"))
        if path.is_dir()
        else [path]
    )
    result: list[dict[str, Any]] = []
    for source in paths:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            result.extend(json.loads(line) for line in handle)
    return result


def action_key(action: dict[str, Any]) -> tuple[str, int]:
    return str(action["action"]), int(action["slot"])


def metric(regrets: list[float], agreements: int) -> dict[str, Any]:
    if not regrets:
        return {
            "states": 0,
            "mean_oracle_regret": None,
            "p95_oracle_regret": None,
            "oracle_agreement": None,
        }
    return {
        "states": len(regrets),
        "mean_oracle_regret": statistics.fmean(regrets),
        "median_oracle_regret": statistics.median(regrets),
        "p90_oracle_regret": float(np.quantile(regrets, 0.90)),
        "p95_oracle_regret": float(np.quantile(regrets, 0.95)),
        "gt_2pp": sum(value > 0.02 for value in regrets),
        "gt_5pp": sum(value > 0.05 for value in regrets),
        "oracle_agreement": agreements / len(regrets),
    }


def evaluate(
    state_ids: list[str],
    canonical: dict[str, dict[str, Any]],
    stage: dict[str, dict[str, Any]],
    oracle: dict[str, dict[str, Any]],
    module: Any,
    strategy: Any,
) -> dict[str, Any]:
    policy_regrets: list[float] = []
    pwin_regrets: list[float] = []
    policy_agreement = pwin_agreement = 0
    disagreement_policy: list[float] = []
    disagreement_pwin: list[float] = []
    disagreement_policy_agreement = disagreement_pwin_agreement = 0
    examples: list[dict[str, Any]] = []

    for state_id in state_ids:
        if state_id not in canonical or state_id not in stage or state_id not in oracle:
            continue
        row = canonical[state_id]
        current = state_from_row(module, row)
        low_actions = stage[state_id]["actions"]
        high_actions = oracle[state_id]["actions"]
        high_by_key = {action_key(action): action for action in high_actions}
        comparable = [action for action in low_actions if action_key(action) in high_by_key]
        if not comparable:
            continue

        def tail(action: dict[str, Any]) -> float:
            return max(0.0, float(action["q_robust"]) - float(action.get("cvar") or 0.0))

        def final_key(action: dict[str, Any]) -> tuple[float, float, float, float, float]:
            return (
                1.0 if action.get("immediate_win") else 0.0,
                policy_score(strategy, current, action),
                float(action["q_robust"]),
                -float(action.get("p_lose_next_turn") or 0.0),
                -tail(action),
            )

        def pwin_key(action: dict[str, Any]) -> tuple[float, float, float, float]:
            return (
                1.0 if action.get("immediate_win") else 0.0,
                float(action["q_robust"]),
                -float(action.get("p_lose_next_turn") or 0.0),
                -tail(action),
            )

        selected_policy = max(comparable, key=final_key)
        selected_pwin = max(comparable, key=pwin_key)
        oracle_best = max(high_actions, key=q_value)
        oracle_value = q_value(oracle_best)
        policy_value = q_value(high_by_key[action_key(selected_policy)])
        pwin_value = q_value(high_by_key[action_key(selected_pwin)])
        policy_regret = max(0.0, oracle_value - policy_value)
        pwin_regret = max(0.0, oracle_value - pwin_value)
        policy_regrets.append(policy_regret)
        pwin_regrets.append(pwin_regret)
        oracle_key = action_key(oracle_best)
        policy_agreement += int(action_key(selected_policy) == oracle_key)
        pwin_agreement += int(action_key(selected_pwin) == oracle_key)

        if action_key(selected_policy) != action_key(selected_pwin):
            disagreement_policy.append(policy_regret)
            disagreement_pwin.append(pwin_regret)
            disagreement_policy_agreement += int(action_key(selected_policy) == oracle_key)
            disagreement_pwin_agreement += int(action_key(selected_pwin) == oracle_key)
            if len(examples) < 25:
                examples.append(
                    {
                        "state_id": state_id,
                        "game_id": row["game_id"],
                        "turn": row["card_action_index"],
                        "policy_choice": {
                            **{key: selected_policy[key] for key in ("action", "slot", "card_id")},
                            "policy_score": policy_score(strategy, current, selected_policy),
                            "displayed_pwin": selected_policy["q_robust"],
                            "oracle_value": policy_value,
                        },
                        "pwin_choice": {
                            **{key: selected_pwin[key] for key in ("action", "slot", "card_id")},
                            "policy_score": policy_score(strategy, current, selected_pwin),
                            "displayed_pwin": selected_pwin["q_robust"],
                            "oracle_value": pwin_value,
                        },
                        "oracle_best": {
                            **{key: oracle_best[key] for key in ("action", "slot", "card_id")},
                            "oracle_value": oracle_value,
                        },
                    }
                )

    return {
        "states": len(policy_regrets),
        "policy_score_first": metric(policy_regrets, policy_agreement),
        "displayed_pwin_first": metric(pwin_regrets, pwin_agreement),
        "rank_disagreement": {
            "count": len(disagreement_policy),
            "fraction": len(disagreement_policy) / max(1, len(policy_regrets)),
            "policy_score_first": metric(disagreement_policy, disagreement_policy_agreement),
            "displayed_pwin_first": metric(disagreement_pwin, disagreement_pwin_agreement),
        },
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("stage_a", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--random-size", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    canonical = {row["state_id"]: row for row in read_rows(args.dataset)}
    stage = {row["state_id"]: row for row in read_rows(args.stage_a)}
    oracle = {row["state_id"]: row for row in read_rows(args.oracle)}
    common = sorted(set(canonical) & set(stage) & set(oracle))
    random_ids = sorted(
        common,
        key=lambda state_id: hashlib.sha256(f"policy-vs-pwin:{state_id}".encode()).hexdigest(),
    )[: min(args.random_size, len(common))]

    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    strategy = module.CardStrategy(catalog)
    payload = {
        "version": "policy-vs-displayed-pwin-audit-3.9-v1",
        "low_budget_source": "matching first-seed q_robust/p_lose_next_turn",
        "high_budget_judge": "multiseed oracle q_mean",
        "selection_objective": [
            "immediate_terminal_win",
            "policy_score",
            "displayed_pwin",
            "negative_p_lose_next_turn",
            "negative_tail_risk",
        ],
        "full_holdout": evaluate(common, canonical, stage, oracle, module, strategy),
        "deterministic_random_holdout": evaluate(
            random_ids, canonical, stage, oracle, module, strategy
        ),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        "version": payload["version"],
        "full_holdout": {key: value for key, value in payload["full_holdout"].items() if key != "examples"},
        "deterministic_random_holdout": {
            key: value
            for key, value in payload["deterministic_random_holdout"].items()
            if key != "examples"
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
