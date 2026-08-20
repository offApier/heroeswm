from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from audit_card_dataset import game_result, load_segments, merge_events, outcome_label  # noqa: E402


SALT = "policy-improvement-3.9-locked-split-2026-08-20-v1"


def prior_oracle_games(results: Path) -> set[int]:
    game_ids: set[int] = set()
    if not results.exists():
        return game_ids
    for path in results.glob("*.json*"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in (payload.get("rows") or []) if isinstance(payload, dict) else []:
            if isinstance(row, dict) and row.get("game_id") is not None:
                game_ids.add(int(row["game_id"]))
    return game_ids


def dataset_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.json"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest().upper()


def stable_key(game_id: int) -> str:
    return hashlib.sha256(f"{SALT}:{game_id}".encode("ascii")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("card_games", type=Path)
    parser.add_argument("--prior-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    games = load_segments(args.card_games)
    contaminated = prior_oracle_games(args.prior_results)
    records: list[dict[str, Any]] = []
    for game_id, segments in sorted(games.items()):
        events, artifacts, duplicates = merge_events(segments)
        result = game_result(segments)
        label = outcome_label(result, segments)
        first_actor = str(events[0].get("actor") or "unknown") if events else "unknown"
        length = len(events)
        length_bucket = "short" if length <= 32 else "medium" if length <= 64 else "long"
        records.append({
            "game_id": game_id,
            "outcome": label,
            "initiative": first_actor,
            "length_bucket": length_bucket,
            "reconnect": len(segments) > 1,
            "segments": len(segments),
            "events": length,
            "terminal_artifacts": artifacts,
            "duplicates": duplicates,
            "prior_oracle_label": game_id in contaminated,
        })

    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    assignments: dict[int, str] = {}
    for record in records:
        if record["prior_oracle_label"]:
            assignments[record["game_id"]] = "train"
        else:
            strata[(record["outcome"], record["initiative"], record["length_bucket"])].append(record)
    for selected in strata.values():
        selected.sort(key=lambda record: stable_key(record["game_id"]))
        n = len(selected)
        holdout_count = round(n * 0.15) if n >= 7 else 0
        validation_count = round(n * 0.15) if n >= 7 else (1 if n >= 3 else 0)
        for index, record in enumerate(selected):
            if index < holdout_count:
                split = "holdout"
            elif index < holdout_count + validation_count:
                split = "validation"
            else:
                split = "train"
            assignments[record["game_id"]] = split

    split_counts = Counter(assignments.values())
    stratified_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        split = assignments[record["game_id"]]
        key = f"{record['outcome']}|{record['initiative']}|{record['length_bucket']}|reconnect={record['reconnect']}"
        stratified_counts[key][split] += 1
    payload = {
        "version": "policy-3.9-split-v1",
        "created_date": "2026-08-20",
        "salt": SALT,
        "dataset_fingerprint_sha256": dataset_fingerprint(args.card_games),
        "rules": {
            "unit": "whole game_id",
            "target": "70/15/15 stratified by outcome, initiative and length",
            "prior_3_8_oracle_games_forced_to_train": True,
            "holdout_access_policy": "do not calculate or inspect policy/oracle metrics until architecture is frozen",
        },
        "counts": dict(split_counts),
        "prior_oracle_labeled_games": sorted(contaminated),
        "prior_oracle_labeled_count": len(contaminated),
        "stratified_counts": {key: dict(value) for key, value in sorted(stratified_counts.items())},
        "assignments": {str(game_id): assignments[game_id] for game_id in sorted(assignments)},
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "dataset_fingerprint_sha256": payload["dataset_fingerprint_sha256"],
        "counts": payload["counts"],
        "prior_oracle_labeled_count": payload["prior_oracle_labeled_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
