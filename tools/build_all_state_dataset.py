from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from audit_card_dataset import (  # noqa: E402
    decision_key,
    game_result,
    load_segments,
    merge_events,
    outcome_label,
    parsed_move,
)


def load_module(source: Path):
    spec = importlib.util.spec_from_file_location("policy39_dataset_card_game", source / "card_game.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load card_game.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def player(module: Any, values: dict[str, Any]):
    return module.PlayerState(**{name: int(values.get(name) or 0) for name in (
        "ore", "mana", "army", "tower", "wall", "mine", "monastery", "barracks"
    )})


def game_state(module: Any, game_id: int, snapshot: dict[str, Any]):
    user_no = int(snapshot.get("player_no") or 1)
    enemy_no = 3 - user_no
    return module.GameState(
        game_id=game_id,
        turn=int(snapshot.get("turn") or 0),
        is_your_turn=bool(snapshot.get("is_your_turn")),
        player_no=user_no,
        time_left=int(snapshot.get("time_left") or 0),
        players={
            user_no: player(module, snapshot.get("me") or {}),
            enemy_no: player(module, snapshot.get("opponent") or {}),
        },
        hand=[int(card_id) for card_id in snapshot.get("hand") or []],
        winner=int(snapshot.get("winner") or 0),
        finish_reason=int(snapshot.get("finish_reason") or 0),
        last_move=str(snapshot.get("last_move") or ""),
        now_player=int(snapshot.get("now_player") or 0),
        table=str(snapshot.get("table") or ""),
        must_discard=bool(snapshot.get("must_discard")),
    )


def phase(snapshot: dict[str, Any]) -> str:
    turn = int(snapshot.get("turn") or 0)
    me, enemy = snapshot.get("me") or {}, snapshot.get("opponent") or {}
    distance = min(
        50 - int(me.get("tower") or 0),
        50 - int(enemy.get("tower") or 0),
        int(me.get("tower") or 0),
        int(enemy.get("tower") or 0),
    )
    if distance <= 10:
        return "terminal_race"
    if turn <= 10:
        return "opening"
    if turn <= 25:
        return "early"
    if turn <= 50:
        return "midgame"
    return "late"


def legal_actions(strategy: Any, state: Any) -> list[dict[str, int | str]]:
    actions: list[dict[str, int | str]] = []
    for slot, card_id in enumerate(state.hand):
        actions.append({"action": "drop", "slot": slot, "card_id": card_id})
        card = strategy.catalog.cards.get(card_id)
        if not state.must_discard and card is not None and strategy._affordable(card, state.me):
            actions.append({"action": "turn", "slot": slot, "card_id": card_id})
    return actions


def signature(game_id: int, event: dict[str, Any]) -> str:
    payload = json.dumps([game_id, decision_key(event)], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("card_games", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--game-index", type=Path, required=True)
    args = parser.parse_args()

    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    games = load_segments(args.card_games)
    split_payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    assignments = {int(key): value for key, value in split_payload["assignments"].items()}
    totals: Counter[str] = Counter()
    game_rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", newline="\n") as output:
        for game_id, segments in sorted(games.items()):
            events, artifacts, duplicates = merge_events(segments)
            result = game_result(segments)
            label = outcome_label(result, segments)
            strategy = module.CardStrategy(catalog)
            strategy.reset_game(game_id)
            public_history: list[dict[str, Any]] = []
            seen_decisions: set[tuple[Any, ...]] = set()
            decisions = unknown_transitions = 0
            first_actor = str(events[0].get("actor") or "unknown") if events else "unknown"
            for event in events:
                selected = event.get("selected")
                key = decision_key(event) if isinstance(selected, dict) else None
                if key is not None and key not in seen_decisions:
                    seen_decisions.add(key)
                    before_raw = event.get("before") or {}
                    state = game_state(module, game_id, before_raw)
                    actions = legal_actions(strategy, state)
                    hist_action = str(event.get("action") or selected.get("action") or "")
                    hist_slot = int(event.get("hand_slot") if event.get("hand_slot") is not None else selected.get("slot") or 0)
                    hist_card = int(event.get("card_id") if event.get("card_id") is not None else state.hand[hist_slot])
                    row = {
                        "state_id": signature(game_id, event),
                        "game_id": game_id,
                        "split": assignments[game_id],
                        "outcome": label,
                        "result": result,
                        "initiative": first_actor,
                        "reconnect": len(segments) > 1,
                        "segment_count": len(segments),
                        "decision_index": decisions,
                        "card_action_index": int(before_raw.get("turn") or 0),
                        "phase": phase(before_raw),
                        "visible_state": before_raw,
                        "our_hand": state.hand,
                        "legal_actions": actions,
                        "historical_choice": {"action": hist_action, "slot": hist_slot, "card_id": hist_card},
                        "history": list(public_history),
                        "history_sha256": hashlib.sha256(json.dumps(public_history, sort_keys=True).encode("utf-8")).hexdigest(),
                        "cooldown_last_seen": {
                            str(item["card_id"]): int(before_raw.get("turn") or 0) - int(item["turn"])
                            for item in public_history
                            if item.get("card_id") is not None
                        },
                        "reconnect_unknown_transitions_before": unknown_transitions,
                        "belief_reconstruction": {
                            "algorithm": "persistent-particle-filter-v2",
                            "seed": f"policy39:{game_id}:{int(before_raw.get('turn') or 0)}",
                            "input_is_prefix_only": True,
                            "full_particles_materialized_during_oracle": True,
                        },
                    }
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    decisions += 1
                    totals[f"decisions_{assignments[game_id]}"] += 1
                    totals["decisions"] += 1
                    totals[f"historical_{hist_action}"] += 1
                    totals["legal_actions"] += len(actions)
                before_turn = int((event.get("before") or {}).get("turn") or 0)
                event_turn = int(event.get("turn") or 0)
                gap = max(0, event_turn - before_turn - 1)
                for offset in range(gap):
                    public_history.append({
                        "turn": before_turn + 1 + offset,
                        "actor": "unknown",
                        "action": "unknown",
                        "card_id": None,
                        "latent_reconnect_transition": True,
                    })
                    unknown_transitions += 1
                parsed = parsed_move(event)
                if parsed is not None:
                    public_history.append({
                        "turn": event_turn,
                        "actor": str(event.get("actor") or "unknown"),
                        "action": parsed[0],
                        "card_id": parsed[1],
                        "latent_reconnect_transition": False,
                    })
            game_rows.append({
                "game_id": game_id,
                "split": assignments[game_id],
                "outcome": label,
                "result": result,
                "initiative": first_actor,
                "reconnect": len(segments) > 1,
                "segments": len(segments),
                "events": len(events),
                "decisions": decisions,
                "unknown_transitions": unknown_transitions,
                "terminal_artifacts_removed": artifacts,
                "duplicates_removed": duplicates,
            })

    summary = {
        "version": "all-state-dataset-3.9-v1",
        "future_information_in_policy_features": False,
        "games": len(game_rows),
        "totals": dict(totals),
        "splits": dict(Counter(row["split"] for row in game_rows)),
        "outcomes": dict(Counter(row["outcome"] for row in game_rows)),
        "reconnect_games": sum(row["reconnect"] for row in game_rows),
        "unknown_reconnect_transitions": sum(row["unknown_transitions"] for row in game_rows),
        "game_rows": game_rows,
    }
    args.game_index.parent.mkdir(parents=True, exist_ok=True)
    args.game_index.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("games", "totals", "splits", "outcomes", "reconnect_games", "unknown_reconnect_transitions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
