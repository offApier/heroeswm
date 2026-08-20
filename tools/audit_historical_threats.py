from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from audit_card_dataset import decision_key, load_segments, merge_events, parsed_move  # noqa: E402


BUCKETS = ((0, .05), (.05, .10), (.10, .20), (.20, .40), (.40, .60), (.60, .80), (.80, .95), (.95, 1.000001))


def bucket(value: float) -> str:
    if value == 1.0:
        return "exactly_100"
    for left, right in BUCKETS:
        if left <= value < right:
            return f"{left:.2f}_{min(1, right):.2f}"
    return "unknown"


def threat_rows(selected: dict[str, Any]) -> list[tuple[str, float]]:
    rows = []
    for text in selected.get("opponent_belief_top_threats") or []:
        match = re.match(r"(.+):\s*([0-9.,]+)%$", str(text))
        if match:
            rows.append((match.group(1), float(match.group(2).replace(",", ".")) / 100))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("card_games", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    assignments = {int(key): value for key, value in manifest["assignments"].items()}
    games = load_segments(args.card_games)
    calibration: dict[str, Counter[str]] = defaultdict(Counter)
    exact_claims = []
    totals = Counter()
    for game_id, segments in sorted(games.items()):
        split = assignments[game_id]
        if split == "holdout":
            continue
        events, artifacts, _duplicates = merge_events(segments)
        totals["terminal_artifacts_removed"] += artifacts
        seen = set()
        for index, event in enumerate(events):
            selected = event.get("selected")
            if not isinstance(selected, dict):
                continue
            key = decision_key(event)
            if key in seen:
                continue
            seen.add(key)
            probability = selected.get("p_lose_next_turn")
            if probability is None:
                continue
            probability = float(probability)
            next_opponent = None
            # Extra-turn chains remain our action; the next opposing card action
            # is the observable target for next-opponent-action calibration.
            for future in events[index + 1 :]:
                if str(future.get("actor")) == "opponent" and parsed_move(future) is not None:
                    next_opponent = future
                    break
                if future.get("selected") and str(future.get("actor")) == "us" and int((future.get("before") or {}).get("turn") or 0) > int(event.get("turn") or 0) + 3:
                    break
            actual_loss = False
            next_card = None
            if next_opponent is not None:
                move = parsed_move(next_opponent)
                next_card = int(move[1]) if move else None
                after = next_opponent.get("after") or {}
                player_no = int(after.get("player_no") or 1)
                winner = int(after.get("winner") or 0)
                actual_loss = bool(int(after.get("finish_reason") or 0) and winner not in {0, player_no})
            name = bucket(probability)
            calibration[name]["states"] += 1
            calibration[name]["actual_next_losses"] += int(actual_loss)
            calibration[name]["predicted_sum_pp"] += round(probability * 1_000_000)
            totals["states_with_p_lose"] += 1
            for card_name, threat_probability in threat_rows(selected):
                totals["threat_entries"] += 1
                if threat_probability >= .999999:
                    confirmed = next_opponent is not None and str(next_opponent.get("card_name") or "") == card_name
                    exact_claims.append({
                        "game_id": game_id, "turn": int((event.get("before") or {}).get("turn") or 0),
                        "card_name": card_name, "next_opponent_card_id": next_card,
                        "observable_immediate_confirmation": confirmed,
                    })
    calibration_payload = {}
    for name, values in calibration.items():
        count = values["states"]
        calibration_payload[name] = {
            "states": count,
            "mean_predicted": values["predicted_sum_pp"] / 1_000_000 / count,
            "observed_next_loss_rate": values["actual_next_losses"] / count,
        }
    payload = {
        "version": "historical-threat-audit-3.9-v1",
        "splits": ["train", "validation"], "holdout_opened": False,
        "totals": dict(totals), "immediate_loss_calibration": calibration_payload,
        "exact_100_claims": {
            "count": len(exact_claims),
            "observable_immediate_confirmations": sum(item["observable_immediate_confirmation"] for item in exact_claims),
            "rows": exact_claims,
            "interpretation": "Non-confirmation is diagnostic evidence, not proof the card was absent; finite-particle certainty is removed in candidate.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "exact_100_claims": {key: value for key, value in payload["exact_100_claims"].items() if key != "rows"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
