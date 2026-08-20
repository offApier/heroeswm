from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("development_split", type=Path)
    parser.add_argument("game_index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contaminated-game", type=int, action="append", default=[])
    args = parser.parse_args()
    development = json.loads(args.development_split.read_text(encoding="utf-8"))
    assignments = {int(key): value for key, value in development["assignments"].items()}
    index = json.loads(args.game_index.read_text(encoding="utf-8"))["game_rows"]
    info = {int(row["game_id"]): row for row in index}
    candidates = [game_id for game_id, split in assignments.items() if split == "holdout" and game_id not in set(args.contaminated_game)]
    strata = defaultdict(list)
    for game_id in candidates:
        row = info[game_id]
        strata[(row["outcome"], row["initiative"], bool(row["reconnect"]))].append(game_id)
    final = {}
    for game_id, split in assignments.items():
        final[game_id] = "development" if split in {"train", "validation"} else "reserved"
    for game_id in args.contaminated_game:
        final[game_id] = "contaminated_structural_only"
    # Approximately half of every stratum goes to freeze-validation and half
    # to the one-time final holdout.  The salt was fixed before either metric
    # was computed.
    for stratum, game_ids in strata.items():
        ordered = sorted(game_ids, key=lambda game_id: hashlib.sha256(repr(("policy39-final-freeze-v1", stratum, game_id)).encode()).hexdigest())
        for index_in_stratum, game_id in enumerate(ordered):
            final[game_id] = "freeze_validation" if index_in_stratum % 2 == 0 else "final_holdout"
    payload = {
        "version": "policy-3.9-final-freeze-split-v1",
        "created_before_freeze_validation_metrics": True,
        "source_development_split": str(args.development_split),
        "dataset_fingerprint_sha256": development["dataset_fingerprint_sha256"],
        "contaminated_structural_only": args.contaminated_game,
        "assignments": {str(key): value for key, value in sorted(final.items())},
        "counts": dict(Counter(final.values())),
        "policy": "No model/threshold/architecture changes after freeze-validation. Final holdout opens once.",
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "assignments"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
