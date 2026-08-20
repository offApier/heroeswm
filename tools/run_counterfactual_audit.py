from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[3]
AUDIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "tools"))
sys.path.insert(0, str(AUDIT_ROOT / "oracle"))
from audit_card_dataset import decision_key, game_result, load_segments, merge_events  # noqa: E402
from discard_oracle import DiscardSearchOracle, hand_diagnostics  # noqa: E402
from fast_discard_oracle import FastDiscardOracle, PersistentParticleFilter  # noqa: E402


def load_card_game(source: Path) -> Any:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("card_game_discard_audit", source / "card_game.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("card_game.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def game_state(module: Any, game_id: int, snapshot: dict[str, Any]) -> Any:
    player_no = int(snapshot.get("player_no") or 1)
    opponent_no = 2 if player_no == 1 else 1
    return module.GameState(
        game_id=game_id,
        turn=int(snapshot.get("turn") or 0),
        is_your_turn=bool(snapshot.get("is_your_turn")),
        player_no=player_no,
        time_left=max(40, int(snapshot.get("time_left") or 0)),
        players={
            player_no: module.PlayerState(**snapshot["me"]),
            opponent_no: module.PlayerState(**snapshot["opponent"]),
        },
        hand=[int(card_id) for card_id in snapshot.get("hand") or []],
        winner=int(snapshot.get("winner") or 0),
        finish_reason=int(snapshot.get("finish_reason") or 0),
        last_move=str(snapshot.get("last_move") or ""),
        now_player=int(snapshot.get("now_player") or 0),
        table=str(snapshot.get("table") or ""),
        must_discard=bool(snapshot.get("must_discard")),
    )


def first_actor(events: list[dict[str, Any]]) -> str:
    return str(events[0].get("actor") or "unknown") if events else "unknown"


def split_games(games: dict[int, list[Any]]) -> dict[int, str]:
    strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for game_id, segments in games.items():
        events, _artifacts, _duplicates = merge_events(segments)
        strata[(game_result(segments), first_actor(events))].append(game_id)
    result: dict[int, str] = {}
    for stratum, game_ids in strata.items():
        ordered = sorted(
            game_ids,
            key=lambda game_id: hashlib.sha256(repr(("discard-audit-v1", stratum, game_id)).encode()).hexdigest(),
        )
        for index, game_id in enumerate(ordered):
            fraction = (index + 0.5) / len(ordered)
            result[game_id] = "train" if fraction < 0.70 else "validation" if fraction < 0.85 else "holdout"
    return result


def phase(snapshot: dict[str, Any]) -> str:
    turn = int(snapshot.get("turn") or 0)
    me, opponent = snapshot.get("me") or {}, snapshot.get("opponent") or {}
    distance = min(
        50 - int(me.get("tower") or 0),
        50 - int(opponent.get("tower") or 0),
        int(me.get("tower") or 0),
        int(opponent.get("tower") or 0),
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


def candidate_priority(event: dict[str, Any], catalog: Any) -> float:
    candidates = event.get("candidates") or []
    action = str(event.get("action") or "")
    before = event.get("before") or {}
    hand = [int(card_id) for card_id in before.get("hand") or []]
    play = [float(row["p_win"]) for row in candidates if row.get("action") == "turn" and row.get("p_win") is not None]
    drop = [float(row["p_win"]) for row in candidates if row.get("action") == "drop" and row.get("p_win") is not None]
    priority = 4.0 if action == "drop" else 0.0
    priority += 4.0 if not play else 2.0 if len(play) == 1 else 0.0
    if play and drop:
        priority += max(0.0, 4.0 - 40.0 * abs(max(play) - max(drop)))
    priority += 2.0 if phase(before) == "terminal_race" else 0.0
    priority += 2.0 if any(card_id in {1, 2, 12, 13, 34, 35, 68, 73, 100, 101} for card_id in hand) else 0.0
    resource_types = Counter(
        "ore" if catalog[card_id].ore else "mana" if catalog[card_id].mana else "army" if catalog[card_id].army else "free"
        for card_id in hand if card_id in catalog.cards
    )
    priority += max(resource_types.values(), default=0) / 3.0
    digest = int(hashlib.sha256(repr((event.get("turn"), hand)).encode()).hexdigest()[:8], 16)
    return priority + digest / 2**36


def build_targets(
    games: dict[int, list[Any]],
    catalog: Any,
    assignments: dict[int, str],
    split: str,
    limit: int,
    target_kind: str,
) -> dict[int, set[tuple[Any, ...]]]:
    rows: list[tuple[float, int, tuple[Any, ...]]] = []
    for game_id, segments in games.items():
        if split != "all" and assignments[game_id] != split:
            continue
        events, _artifacts, _duplicates = merge_events(segments)
        seen: set[tuple[Any, ...]] = set()
        for event in events:
            if not event.get("selected"):
                continue
            if target_kind in {"historical-play", "historical-play-no-extra", "historical-play-interior"} and event.get("action") != "turn":
                continue
            if target_kind == "historical-discard" and event.get("action") != "drop":
                continue
            if target_kind == "extra-turn":
                hand = set((event.get("before") or {}).get("hand") or [])
                if not hand.intersection({1, 2, 12, 13, 34, 35, 68, 73, 100, 101}):
                    continue
            if target_kind in {"historical-play-no-extra", "historical-play-interior"}:
                hand = set((event.get("before") or {}).get("hand") or [])
                if hand.intersection({1, 2, 12, 13, 34, 35, 68, 73, 100, 101}):
                    continue
            if target_kind == "historical-play-interior":
                selected_pwin = (event.get("selected") or {}).get("p_win")
                if selected_pwin is None or not 0.05 <= float(selected_pwin) <= 0.95:
                    continue
            key = decision_key(event)
            if key in seen:
                continue
            seen.add(key)
            rows.append((candidate_priority(event, catalog), game_id, key))
    rows.sort(reverse=True)
    selected = rows[:limit] if limit > 0 else rows
    targets: dict[int, set[tuple[Any, ...]]] = defaultdict(set)
    for _priority, game_id, key in selected:
        targets[game_id].add(key)
    return targets


def targets_from_positions(games: dict[int, list[Any]], path: Path) -> dict[int, set[tuple[Any, ...]]]:
    requested = json.loads(path.read_text(encoding="utf-8"))
    wanted = {
        (int(row["game_id"]), int(row["turn"]))
        for row in requested
    }
    targets: dict[int, set[tuple[Any, ...]]] = defaultdict(set)
    for game_id, segments in games.items():
        requested_turns = {turn for candidate_game, turn in wanted if candidate_game == game_id}
        if not requested_turns:
            continue
        events, _artifacts, _duplicates = merge_events(segments)
        for event in events:
            before_turn = int((event.get("before") or {}).get("turn") or -1)
            if event.get("selected") and before_turn in requested_turns:
                targets[game_id].add(decision_key(event))
    missing = wanted - {
        (game_id, turn)
        for game_id in targets
        for turn in requested_turns_for_game(games, game_id, targets[game_id])
    }
    if missing:
        raise ValueError(f"target positions not found: {sorted(missing)}")
    return targets


def requested_turns_for_game(
    games: dict[int, list[Any]], game_id: int, keys: set[tuple[Any, ...]]
) -> set[int]:
    events, _artifacts, _duplicates = merge_events(games[game_id])
    return {
        int((event.get("before") or {}).get("turn") or -1)
        for event in events
        if event.get("selected") and decision_key(event) in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("card_games", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "train", "validation", "holdout"), default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--particles", type=int, default=200)
    parser.add_argument("--screen-particles", type=int, default=200)
    parser.add_argument("--deep-candidates", type=int, default=5)
    parser.add_argument("--oracle", choices=("fast", "deep"), default="fast")
    parser.add_argument(
        "--target-kind",
        choices=("all", "historical-play", "historical-play-no-extra", "historical-play-interior", "historical-discard", "extra-turn"),
        default="all",
    )
    parser.add_argument("--target-positions", type=Path)
    args = parser.parse_args()

    module = load_card_game(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    games = load_segments(args.card_games)
    assignments = split_games(games)
    targets = (
        targets_from_positions(games, args.target_positions)
        if args.target_positions
        else build_targets(games, catalog, assignments, args.split, args.limit, args.target_kind)
    )
    rows: list[dict[str, Any]] = []
    started_all = time.monotonic()

    for game_id in sorted(targets):
        segments = games[game_id]
        events, _artifacts, _duplicates = merge_events(segments)
        strategy = module.CardStrategy(catalog)
        strategy.reset_game(game_id)
        first_before = game_state(module, game_id, events[0]["before"])
        particle_filter = PersistentParticleFilter(
            strategy,
            first_before,
            args.particles,
            seed=f"{game_id}:persistent-filter-v1",
        )
        seen: set[tuple[Any, ...]] = set()
        last_replayed_turn = -1
        for event in events:
            before = game_state(module, game_id, event["before"])
            after = game_state(module, game_id, event["after"])
            key = decision_key(event) if event.get("selected") else None
            duplicate_decision = bool(key is not None and key in seen)
            if key is not None:
                seen.add(key)
            is_target = bool(key is not None and not duplicate_decision and key in targets[game_id])
            if is_target:
                evaluation_started = time.monotonic()
                state = game_state(module, game_id, event["before"])
                # A transition ending at state.turn is exactly the public history
                # available before this decision.  Only a later transition would
                # leak future information into the belief state.
                if last_replayed_turn > state.turn:
                    raise RuntimeError(
                        f"future leakage detected: replayed through {last_replayed_turn}, "
                        f"decision state is {state.turn}"
                    )
                if args.oracle == "fast":
                    filtered_particles, public_last_seen = particle_filter.snapshot(state)
                    screen_count = min(len(filtered_particles), max(1, args.screen_particles))
                    screen_oracle = FastDiscardOracle(
                        module,
                        strategy,
                        state,
                        filtered_particles[:screen_count],
                        public_last_seen,
                        seed=f"{game_id}:{state.turn}:discard-oracle-fast-v1",
                    )
                    screen_started = time.monotonic()
                    screen_ranking = screen_oracle.evaluate()
                    screen_elapsed = time.monotonic() - screen_started
                    allowed = {
                        (choice.action, choice.slot)
                        for choice in screen_ranking[: max(1, args.deep_candidates)]
                    }
                    historical_key = (
                        str(event.get("action")),
                        int(event["hand_slot"]) if event.get("hand_slot") is not None else -1,
                    )
                    allowed.add(historical_key)
                    best_screen_play = next(
                        (choice for choice in screen_ranking if choice.action == "turn"), None
                    )
                    best_screen_drop = next(
                        (choice for choice in screen_ranking if choice.action == "drop"), None
                    )
                    if best_screen_play is not None:
                        allowed.add((best_screen_play.action, best_screen_play.slot))
                    if best_screen_drop is not None:
                        allowed.add((best_screen_drop.action, best_screen_drop.slot))
                    oracle = FastDiscardOracle(
                        module,
                        strategy,
                        state,
                        filtered_particles,
                        public_last_seen,
                        seed=f"{game_id}:{state.turn}:discard-oracle-fast-v1",
                        allowed_actions=allowed,
                    )
                else:
                    screen_count = None
                    screen_ranking = []
                    screen_elapsed = 0.0
                    oracle = DiscardSearchOracle(
                        module,
                        strategy,
                        state,
                        particles=args.particles,
                        seed=f"{game_id}:{state.turn}:discard-oracle-deep-v1",
                    )
                ranking = oracle.evaluate()
                elapsed = time.monotonic() - evaluation_started
                historical = next(
                    (
                        choice for choice in ranking
                        if choice.action == str(event.get("action"))
                        and choice.card_id == (
                            int(event["card_id"]) if event.get("card_id") is not None else -1
                        )
                        and choice.slot == (
                            int(event["hand_slot"]) if event.get("hand_slot") is not None else -1
                        )
                    ),
                    None,
                )
                best_play = next((choice for choice in ranking if choice.action == "turn"), None)
                best_drop = next((choice for choice in ranking if choice.action == "drop"), None)
                best = ranking[0]
                regret_se = None
                regret_ci = None
                if historical is not None and len(best.outcomes) >= 2:
                    differences = [
                        left - right for left, right in zip(best.outcomes, historical.outcomes)
                    ]
                    if len(differences) >= 2:
                        regret_se = statistics.pstdev(differences) / math.sqrt(len(differences))
                        regret_value = best.p_win - historical.p_win
                        regret_ci = [
                            regret_value - 1.96 * regret_se,
                            regret_value + 1.96 * regret_se,
                        ]
                rows.append(
                    {
                        "game_id": game_id,
                        "split": assignments[game_id],
                        "turn": state.turn,
                        "phase": phase(event.get("before") or {}),
                        "game_result": game_result(segments),
                        "hand": state.hand,
                        "state": event.get("before"),
                        "history_max_turn_used": last_replayed_turn,
                        "future_leakage": last_replayed_turn > state.turn,
                        "historical": {
                            "action": event.get("action"),
                            "slot": event.get("hand_slot"),
                            "card_id": event.get("card_id"),
                            "recorded_pwin": (event.get("selected") or {}).get("p_win"),
                            "oracle_pwin": historical.p_win if historical else None,
                        },
                        "oracle_best": {
                            "action": best.action,
                            "slot": best.slot,
                            "card_id": best.card_id,
                            "p_win": best.p_win,
                            "paired_se": best.paired_se,
                            "ci_diff": best.ci_diff,
                            "replacement_distribution": best.replacement_distribution,
                        },
                        "oracle_ranking": [
                            {
                                "action": choice.action,
                                "slot": choice.slot,
                                "card_id": choice.card_id,
                                "p_win": choice.p_win,
                                "tail_survival": getattr(choice, "tail_survival", None),
                                "replacement_distribution": choice.replacement_distribution,
                            }
                            for choice in ranking
                        ],
                        "best_play": None if best_play is None else {"card_id": best_play.card_id, "slot": best_play.slot, "p_win": best_play.p_win},
                        "best_discard": None if best_drop is None else {"card_id": best_drop.card_id, "slot": best_drop.slot, "p_win": best_drop.p_win, "replacement_distribution": best_drop.replacement_distribution},
                        "regret": None if historical is None else best.p_win - historical.p_win,
                        "regret_paired_se": regret_se,
                        "regret_ci": regret_ci,
                        "play_minus_discard": None if best_play is None or best_drop is None else best_play.p_win - best_drop.p_win,
                        "hand_diagnostics": hand_diagnostics(strategy, state, state.hand),
                        "particles": args.particles,
                        "screen_particles": screen_count,
                        "screen_runtime_seconds": screen_elapsed,
                        "screen_best": None if not screen_ranking else {
                            "action": screen_ranking[0].action,
                            "slot": screen_ranking[0].slot,
                            "card_id": screen_ranking[0].card_id,
                            "p_win": screen_ranking[0].p_win,
                        },
                        "runtime_seconds": elapsed,
                        "particle_filter": {
                            "conditioned_actions": particle_filter.conditioned_actions,
                            "rejuvenated_particles": particle_filter.rejuvenated_particles,
                            "unknown_transitions": particle_filter.unknown_transitions,
                        },
                    }
                )
            strategy.observe_transition(before, after)
            if not duplicate_decision:
                particle_filter.advance(
                    before,
                    after,
                    actor=str(event.get("actor") or "unknown"),
                    action=str(event.get("action") or "drop"),
                    card_id=int(event["card_id"]) if event.get("card_id") is not None else -1,
                    action_index=int(event.get("turn") or after.turn),
                )
            last_replayed_turn = max(last_replayed_turn, int(event.get("turn") or -1))
            if is_target:
                print(
                    json.dumps(
                        {"done": len(rows), "game_id": game_id, "turn": int(event.get("turn") or 0)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                checkpoint = args.output.with_suffix(args.output.suffix + ".partial")
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(
                    json.dumps(
                        {
                            "method": f"offline-discard-oracle-{args.oracle}-v1",
                            "split": args.split,
                            "particles": args.particles,
                            "completed": len(rows),
                            "rows": rows,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    regrets = [float(row["regret"]) for row in rows if row.get("regret") is not None]
    play_loses = [row for row in rows if row["historical"]["action"] == "turn" and row["oracle_best"]["action"] == "drop"]
    drop_loses = [row for row in rows if row["historical"]["action"] == "drop" and row["oracle_best"]["action"] == "turn"]
    payload = {
        "method": {
            "version": f"offline-discard-oracle-{args.oracle}-v1",
            "split": args.split,
            "particles": args.particles,
            "screen_particles": args.screen_particles if args.oracle == "fast" else None,
            "deep_candidates": args.deep_candidates if args.oracle == "fast" else None,
            "target_kind": args.target_kind,
            "future_events_used_for_belief": False,
            "opponent_hand_note": "persistent sequential particle filter; revealed opponent actions condition/resample incompatible particles",
        },
        "dataset_split": Counter(assignments.values()),
        "audited_decisions": len(rows),
        "historical_play_oracle_discard": len(play_loses),
        "historical_discard_oracle_play": len(drop_loses),
        "regret_thresholds": {
            ">=1pp": sum(value >= 0.01 for value in regrets),
            ">=2pp": sum(value >= 0.02 for value in regrets),
            ">=5pp": sum(value >= 0.05 for value in regrets),
            ">=10pp": sum(value >= 0.10 for value in regrets),
        },
        "regret_mean": statistics.fmean(regrets) if regrets else None,
        "runtime_seconds": time.monotonic() - started_all,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "rows"}, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
