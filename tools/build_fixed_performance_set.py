from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


EXTRA = {1, 2, 12, 13, 34, 35, 68, 73, 100, 101}


def stable(row: dict) -> str:
    return hashlib.sha256(f"perf39:{row['state_id']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()
    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    categories = {
        "normal": lambda r: r.get("phase") == "midgame" and not r.get("reconnect") and 8 <= len(r.get("legal_actions") or []) <= 10,
        "many_legal_actions": lambda r: len(r.get("legal_actions") or []) >= 11,
        "forced_or_many_discards": lambda r: bool(r["visible_state"].get("must_discard")) or sum(a["action"] == "drop" for a in r.get("legal_actions") or []) >= 6,
        "extra_turn": lambda r: bool(EXTRA.intersection(r.get("our_hand") or [])),
        "terminal_race": lambda r: r.get("phase") == "terminal_race",
        "complex_belief": lambda r: int(r.get("card_action_index") or 0) >= 75 and len(r.get("history") or []) >= 60,
        "reconnect_latent": lambda r: bool(r.get("reconnect")) or int(r.get("reconnect_unknown_transitions_before") or 0) > 0,
        "late_high_resources": lambda r: r.get("phase") in {"late", "terminal_race"} and sum(r["visible_state"]["me"].get(x, 0) for x in ("ore", "mana", "army")) >= 45,
    }
    selected, used = [], set()
    for category, predicate in categories.items():
        candidates = sorted((row for row in rows if predicate(row) and row["state_id"] not in used), key=stable)
        for row in candidates[: args.per_category]:
            used.add(row["state_id"])
            selected.append({
                "category": category, "state_id": row["state_id"], "game_id": row["game_id"],
                "turn": row["visible_state"]["turn"], "phase": row.get("phase"),
                "legal_actions": len(row.get("legal_actions") or []), "reconnect": bool(row.get("reconnect")),
            })
    payload = {"version": "performance-set-3.9-v1", "states": selected, "count": len(selected)}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(selected), "categories": {name: sum(x["category"] == name for x in selected) for name in categories}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
