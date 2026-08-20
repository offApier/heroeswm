from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from audit_card_dataset import load_segments, merge_events, parsed_move  # noqa: E402


def load_module(source: Path):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("policy39_card_game", source / "card_game.py")
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


def state_for_actor(module: Any, game_id: int, snapshot: dict[str, Any], actor_no: int):
    user_no = int(snapshot.get("player_no") or 1)
    enemy_no = 2 if user_no == 1 else 1
    players = {
        user_no: player(module, snapshot.get("me") or {}),
        enemy_no: player(module, snapshot.get("opponent") or {}),
    }
    return module.GameState(
        game_id=game_id,
        turn=int(snapshot.get("turn") or 0),
        is_your_turn=True,
        player_no=actor_no,
        time_left=int(snapshot.get("time_left") or 0),
        players=players,
        hand=[int(card_id) for card_id in snapshot.get("hand") or []],
        winner=int(snapshot.get("winner") or 0),
        finish_reason=int(snapshot.get("finish_reason") or 0),
        last_move=str(snapshot.get("last_move") or ""),
        now_player=int(snapshot.get("now_player") or 0),
        table=str(snapshot.get("table") or ""),
        must_discard=bool(snapshot.get("must_discard")),
    )


def observed_players(module: Any, snapshot: dict[str, Any]) -> dict[int, Any]:
    user_no = int(snapshot.get("player_no") or 1)
    enemy_no = 2 if user_no == 1 else 1
    return {
        user_no: player(module, snapshot.get("me") or {}),
        enemy_no: player(module, snapshot.get("opponent") or {}),
    }


def remove_next_turn_income(module: Any, value: Any) -> Any:
    return module.PlayerState(
        ore=max(0, value.ore - value.mine),
        mana=max(0, value.mana - value.monastery),
        army=max(0, value.army - value.barracks),
        tower=value.tower,
        wall=value.wall,
        mine=value.mine,
        monastery=value.monastery,
        barracks=value.barracks,
    )


def normalize_server_terminal_tower(module: Any, value: Any) -> Any:
    # The server keeps damage overshoot as a negative tower value in the final
    # response, while the search state deliberately clamps terminal health to
    # zero. Those states are strategically equivalent and are not a simulator
    # effect mismatch.
    if value.tower >= 0:
        return value
    return module.PlayerState(**{**asdict(value), "tower": 0})


def delta(before: Any, after: Any) -> dict[str, int]:
    left, right = asdict(before), asdict(after)
    return {key: right[key] - left[key] for key in left if right[key] != left[key]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("card_games", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    engine = module.CardStrategy(catalog)
    games = load_segments(args.card_games)
    per_card: dict[int, Counter[str]] = defaultdict(Counter)
    mismatch_examples: list[dict[str, Any]] = []
    total = exact = actor_exact = defender_exact = skipped = 0
    initial_opponent_unsynchronized = terminal_tower_overshoots = 0
    discontinuous_transitions_excluded = 0
    field_mismatches: Counter[str] = Counter()

    for game_id, segments in sorted(games.items()):
        events, _artifacts, _duplicates = merge_events(segments)
        for event in events:
            parsed = parsed_move(event)
            if parsed is None or parsed[0] != "turn":
                continue
            before_raw, after_raw = event.get("before") or {}, event.get("after") or {}
            if not before_raw or not after_raw:
                skipped += 1
                continue
            card_id = int(event.get("card_id") if event.get("card_id") is not None else parsed[1])
            if card_id not in catalog.cards:
                skipped += 1
                continue
            actor_no = int(event.get("actor_no") or (before_raw.get("player_no") if event.get("actor") == "us" else 0) or 0)
            if actor_no not in (1, 2):
                user_no = int(before_raw.get("player_no") or 1)
                actor_no = user_no if event.get("actor") == "us" else 3 - user_no
            # Before the first observed opponent action, old logs populated the
            # hidden opponent state from the symmetric default. The response
            # reveals the actual initial modifiers, so this delta cannot test a
            # card effect without pretending the hidden values were known.
            if event.get("actor") == "opponent" and not str(before_raw.get("last_move") or ""):
                initial_opponent_unsynchronized += 1
                continue
            if int(event.get("turn") or 0) != int(before_raw.get("turn") or 0) + 1:
                # A reconnect/poll gap can fold one or more unobserved actions
                # into the before->after delta. It is useful for latent belief
                # reconstruction but cannot validate one card's simulator.
                discontinuous_transitions_excluded += 1
                continue
            before = state_for_actor(module, game_id, before_raw, actor_no)
            predicted_actor, predicted_defender = engine.simulate(catalog.cards[card_id], before)
            observed = observed_players(module, after_raw)
            observed_actor = observed[actor_no]
            defender_no = 3 - actor_no
            observed_defender = observed[defender_no]

            # When the move passes initiative, the server response already
            # contains the next player's production income. Remove that income
            # before comparing the card's deterministic effect.
            user_no = int(after_raw.get("player_no") or 1)
            after_your_turn = bool(after_raw.get("is_your_turn"))
            next_no = user_no if after_your_turn else 3 - user_no
            if not int(after_raw.get("winner") or 0) and next_no == defender_no:
                observed_defender = remove_next_turn_income(module, observed_defender)
            terminal_tower_overshoots += int(observed_actor.tower < 0) + int(observed_defender.tower < 0)
            observed_actor = normalize_server_terminal_tower(module, observed_actor)
            observed_defender = normalize_server_terminal_tower(module, observed_defender)

            total += 1
            per_card[card_id]["observations"] += 1
            actor_ok = predicted_actor == observed_actor
            defender_ok = predicted_defender == observed_defender
            actor_exact += int(actor_ok)
            defender_exact += int(defender_ok)
            if actor_ok and defender_ok:
                exact += 1
                per_card[card_id]["exact"] += 1
                continue
            per_card[card_id]["mismatch"] += 1
            pred = {"actor": asdict(predicted_actor), "defender": asdict(predicted_defender)}
            obs = {"actor": asdict(observed_actor), "defender": asdict(observed_defender)}
            for side in ("actor", "defender"):
                for field in pred[side]:
                    if pred[side][field] != obs[side][field]:
                        field_mismatches[f"{side}.{field}"] += 1
            if len(mismatch_examples) < 200:
                mismatch_examples.append({
                    "game_id": game_id,
                    "turn": int(event.get("turn") or 0),
                    "actor": event.get("actor"),
                    "card_id": card_id,
                    "card_name": catalog.cards[card_id].name,
                    "before_actor": asdict(before.me),
                    "before_defender": asdict(before.opponent),
                    "predicted_actor": pred["actor"],
                    "observed_actor_normalized": obs["actor"],
                    "predicted_defender": pred["defender"],
                    "observed_defender_normalized": obs["defender"],
                    "predicted_actor_delta": delta(before.me, predicted_actor),
                    "observed_actor_delta": delta(before.me, observed_actor),
                    "predicted_defender_delta": delta(before.opponent, predicted_defender),
                    "observed_defender_delta": delta(before.opponent, observed_defender),
                    "after_is_your_turn": after_your_turn,
                })

    card_rows = []
    for card_id, card in sorted(catalog.cards.items()):
        counts = per_card[card_id]
        observations = counts["observations"]
        card_rows.append({
            "card_id": card_id,
            "name": card.name,
            "cost": {"ore": card.ore, "mana": card.mana, "army": card.army},
            "effect": card.effect,
            "observations": observations,
            "exact": counts["exact"],
            "mismatches": counts["mismatch"],
            "exact_rate": counts["exact"] / observations if observations else None,
        })
    payload = {
        "method": {
            "future_information_used": False,
            "next_turn_income_normalized": True,
            "comparison": "simulator(before, card) vs observed after player states",
        },
        "summary": {
            "catalog_cards": len(catalog.cards),
            "cards_observed": sum(row["observations"] > 0 for row in card_rows),
            "play_events": total,
            "exact_events": exact,
            "exact_rate": exact / total if total else None,
            "actor_exact_rate": actor_exact / total if total else None,
            "defender_exact_rate": defender_exact / total if total else None,
            "skipped": skipped,
            "initial_opponent_states_excluded": initial_opponent_unsynchronized,
            "discontinuous_transitions_excluded": discontinuous_transitions_excluded,
            "terminal_tower_overshoots_normalized": terminal_tower_overshoots,
            "cards_with_mismatches": sum(row["mismatches"] > 0 for row in card_rows),
        },
        "field_mismatches": dict(field_mismatches.most_common()),
        "per_card": card_rows,
        "mismatch_examples": mismatch_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
