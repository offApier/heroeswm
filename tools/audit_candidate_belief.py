from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from run_stage_a_oracle import load_module, state_from_row

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from audit_card_dataset import decision_key, load_segments, merge_events  # noqa: E402


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") in {"train", "validation"}:
                yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--particles", type=int, default=200)
    parser.add_argument("--card-games", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    module = load_module(args.source)
    catalog = module.CardCatalog.load(args.source / "cards_catalog.json")
    selected = sorted(rows(args.dataset), key=lambda row: hashlib.sha256(f"belief-audit:{row['state_id']}".encode()).hexdigest())[: args.states]
    diagnostics = []
    false_certainties = structural_certainties = 0
    if args.card_games and args.split_manifest:
        assignments = {int(key): value for key, value in json.loads(args.split_manifest.read_text(encoding="utf-8"))["assignments"].items()}
        games = load_segments(args.card_games)

        def raw_state(game_id: int, raw: dict):
            player_no = int(raw.get("player_no") or 1)
            enemy_no = 3 - player_no
            names = ("ore", "mana", "army", "tower", "wall", "mine", "monastery", "barracks")
            player = lambda values: module.PlayerState(**{name: int(values.get(name) or 0) for name in names})
            return module.GameState(
                game_id=game_id, turn=int(raw.get("turn") or 0), is_your_turn=bool(raw.get("is_your_turn")),
                player_no=player_no, time_left=40,
                players={player_no: player(raw.get("me") or {}), enemy_no: player(raw.get("opponent") or {})},
                hand=[int(card_id) for card_id in raw.get("hand") or []], last_move=str(raw.get("last_move") or ""),
                must_discard=bool(raw.get("must_discard")),
            )

        for game_id in sorted(games, key=lambda value: hashlib.sha256(f"belief-game:{value}".encode()).hexdigest()):
            if assignments[game_id] not in {"train", "validation"} or len(diagnostics) >= args.states:
                continue
            events, _artifacts, _duplicates = merge_events(games[game_id])
            belief = module.OpponentBelief(catalog, particle_count=args.particles)
            belief.reset(game_id)
            seen = set()
            for event in events:
                before = raw_state(game_id, event.get("before") or {})
                after = raw_state(game_id, event.get("after") or {})
                if event.get("selected"):
                    key = decision_key(event)
                    if key not in seen and len(diagnostics) < args.states:
                        seen.add(key)
                        belief.synchronize_state(before)
                        particles = belief.particles(before, args.particles)
                        diagnostic = belief.diagnostics(before, particles)
                        probabilities = belief.probabilities(particles, set(diagnostic["structurally_certain_cards"]))
                        false_certainties += sum(probability == 1.0 and card_id not in set(diagnostic["structurally_certain_cards"]) for card_id, probability in probabilities.items())
                        structural_certainties += len(diagnostic["structurally_certain_cards"])
                        diagnostics.append(diagnostic)
                belief.observe_transition(before, after)
    else:
        for row in selected:
            state = state_from_row(module, row)
            belief = module.OpponentBelief(catalog, particle_count=args.particles)
            belief.game_id = int(row["game_id"])
            for item in row.get("history") or []:
                action_index = int(item.get("turn") or 0)
                if item.get("card_id") is None:
                    belief.unknown_action_indices.add(action_index)
                    belief.current_action = max(belief.current_action, action_index)
                    continue
                move = ("d" if item.get("action") == "drop" else "t") + str(item["card_id"]) + "-0"
                belief._record(action_index, move, int(item["card_id"]), str(item.get("actor") or "unknown"), str(item.get("action") or "unknown"))
            particles = belief.particles(state, args.particles)
            diagnostic = belief.diagnostics(state, particles)
            probabilities = belief.probabilities(particles, set(diagnostic["structurally_certain_cards"]))
            false_certainties += sum(probability == 1.0 and card_id not in set(diagnostic["structurally_certain_cards"]) for card_id, probability in probabilities.items())
            structural_certainties += len(diagnostic["structurally_certain_cards"])
            diagnostics.append(diagnostic)
    ess = [item["effective_sample_size"] for item in diagnostics]
    unique = [item["unique_opponent_hands"] for item in diagnostics]
    entropy = [item["opponent_hand_entropy"] for item in diagnostics]
    payload = {
        "version": "candidate-belief-audit-3.9-v1", "holdout_opened": False,
        "states": len(diagnostics), "particles_per_state": args.particles,
        "effective_sample_size": {"p5": float(np.quantile(ess, .05)), "median": statistics.median(ess), "p95": float(np.quantile(ess, .95)), "min": min(ess)},
        "unique_opponent_hands": {"p5": float(np.quantile(unique, .05)), "median": statistics.median(unique), "min": min(unique)},
        "opponent_hand_entropy": {"p5": float(np.quantile(entropy, .05)), "median": statistics.median(entropy)},
        "false_100_percent_claims": false_certainties,
        "structurally_proven_100_percent_claims": structural_certainties,
        "resample_count": sum(item["resample_count"] for item in diagnostics),
        "rejuvenation_count": sum(item["rejuvenation_count"] for item in diagnostics),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
