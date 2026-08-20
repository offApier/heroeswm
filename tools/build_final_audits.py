from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_frozen_policy import policy_score
from run_stage_a_oracle import load_module, state_from_row

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from audit_card_dataset import load_segments, merge_events, parsed_move  # noqa: E402


def jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def q_value(action: dict[str, Any]) -> float:
    return float(action.get("q_mean", action.get("q_robust", 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("card_games", type=Path)
    parser.add_argument("simulator", type=Path)
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--oracle", action="append", type=Path, required=True)
    parser.add_argument("--policy", action="append", type=Path, required=True)
    parser.add_argument("--per-card-output", type=Path, required=True)
    parser.add_argument("--per-game-output", type=Path, required=True)
    args = parser.parse_args()

    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    strategy = module.CardStrategy(catalog)
    states: dict[str, dict[str, Any]] = {}
    for path in args.dataset:
        states.update({row["state_id"]: row for row in jsonl(path)})
    teachers: dict[str, dict[str, Any]] = {}
    for path in args.oracle:
        teachers.update({row["state_id"]: row for row in jsonl(path)})

    card_stats: dict[int, dict[str, Any]] = {
        card_id: {
            "card_id": card_id,
            "name": card.name,
            "our_historical_play": 0,
            "our_historical_discard": 0,
            "opponent_play": 0,
            "opponent_discard": 0,
            "candidate_play": 0,
            "candidate_discard": 0,
            "historical_regrets": [],
            "candidate_regrets": [],
            "candidate_by_phase": defaultdict(Counter),
        }
        for card_id, card in catalog.cards.items()
    }

    raw_games = load_segments(args.card_games)
    for game_id, segments in raw_games.items():
        events, _artifacts, _duplicates = merge_events(segments)
        for event in events:
            move = parsed_move(event)
            if move is None:
                continue
            action, card_id = move[:2]
            if card_id not in card_stats:
                continue
            who = "opponent" if str(event.get("actor")) == "opponent" else "our_historical"
            card_stats[card_id][f"{who}_{'discard' if action == 'drop' else 'play'}"] += 1

    for identifier, teacher in teachers.items():
        row = states.get(identifier)
        if row is None:
            continue
        best = max(q_value(action) for action in teacher["actions"])
        historical = row["historical_choice"]
        hmatch = next((action for action in teacher["actions"] if action["action"] == historical["action"] and int(action["slot"]) == int(historical["slot"])), None)
        if hmatch is not None and int(historical["card_id"]) in card_stats:
            card_stats[int(historical["card_id"])]["historical_regrets"].append(max(0.0, best - q_value(hmatch)))
        state = state_from_row(module, row)
        selected = max(teacher["actions"], key=lambda action: policy_score(strategy, state, action))
        card_id = int(selected["card_id"])
        regret = max(0.0, best - q_value(selected))
        card_stats[card_id]["candidate_regrets"].append(regret)
        action_name = "discard" if selected["action"] == "drop" else "play"
        card_stats[card_id][f"candidate_{action_name}"] += 1
        card_stats[card_id]["candidate_by_phase"][str(row.get("phase") or "unknown")][action_name] += 1

    simulator = json.loads(args.simulator.read_text(encoding="utf-8"))
    simulator_by_card = {int(item["card_id"]): item for item in simulator.get("per_card", [])}
    per_card = []
    for card_id, item in sorted(card_stats.items()):
        sim = simulator_by_card.get(card_id, {})
        item["historical_average_oracle_regret"] = mean(item.pop("historical_regrets"))
        item["candidate_average_oracle_regret"] = mean(item.pop("candidate_regrets"))
        item["candidate_by_phase"] = {phase: dict(counts) for phase, counts in item["candidate_by_phase"].items()}
        item["direct_effect_comparisons"] = int(sim.get("observations") or sim.get("comparable") or 0)
        item["direct_effect_exact"] = int(sim.get("exact") or 0)
        item["direct_effect_accuracy"] = float(sim.get("exact_rate")) if sim.get("exact_rate") is not None else (
            item["direct_effect_exact"] / item["direct_effect_comparisons"] if item["direct_effect_comparisons"] else None
        )
        per_card.append(item)
    args.per_card_output.write_text(json.dumps({"cards": per_card}, ensure_ascii=False, indent=2), encoding="utf-8")

    all_games: dict[str, Any] = {}
    classifications = Counter()
    for path in args.policy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        policy_games = payload.get("per_game", [])
        audits = policy_games if isinstance(policy_games, list) else [dict(audit, game_id=game_id) for game_id, audit in policy_games.items()]
        for audit in audits:
            game_id = audit["game_id"]
            largest_historical = float(audit.get("historical_largest_regret") or 0.0)
            largest_candidate = float(audit.get("candidate_largest_regret") or 0.0)
            outcome = str(audit.get("outcome") or "unknown")
            if outcome == "loss":
                improvement = largest_historical - largest_candidate
                if largest_historical > 0.10 and improvement > 0.05:
                    classification = "RESCUABLE"
                elif largest_historical > 0.05 and improvement > 0.02:
                    classification = "POSSIBLY_RESCUABLE"
                else:
                    classification = "NO_CLEAR_RESCUE"
                classifications[classification] += 1
            elif outcome == "win":
                classification = "WIN_AUDIT"
                classifications[classification] += 1
            else:
                classification = "INSUFFICIENT_INFORMATION"
                classifications[classification] += 1
            all_games[str(game_id)] = {**audit, "classification": classification}
    for game_id in sorted(set(raw_games) - {int(value) for value in all_games}):
        all_games[str(game_id)] = {
            "game_id": game_id,
            "outcome": "unknown",
            "classification": "INSUFFICIENT_INFORMATION",
            "reason": "excluded from final evaluation because this game was inspected before the final freeze",
        }
        classifications["INSUFFICIENT_INFORMATION"] += 1
    args.per_game_output.write_text(
        json.dumps({"games_audited": len(all_games), "classification_counts": classifications, "games": all_games}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"cards": len(per_card), "games": len(all_games), "classifications": classifications}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
