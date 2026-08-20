from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def read_oracles(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("actions") and "q_mean" in row["actions"][0]:
                    result[str(row["state_id"])] = row
    return result


def metrics(
    regrets: list[float],
    agreements: int,
    rows: list[dict[str, Any]],
    server_timeout: int,
) -> dict[str, Any]:
    compute_cap = {15: 11.0, 30: 26.0, 40: 36.0}[server_timeout]
    return {
        "states": len(regrets),
        "mean_regret": statistics.fmean(regrets),
        "median_regret": statistics.median(regrets),
        "p90_regret": float(np.quantile(regrets, 0.90)),
        "p95_regret": float(np.quantile(regrets, 0.95)),
        "gt_2pp": sum(value > 0.02 for value in regrets),
        "gt_5pp": sum(value > 0.05 for value in regrets),
        "oracle_agreement": agreements / len(regrets),
        "runtime_median": statistics.median(float(row["elapsed"]) for row in rows),
        "runtime_p95": float(np.quantile([float(row["elapsed"]) for row in rows], 0.95)),
        "runtime_max": max(float(row["elapsed"]) for row in rows),
        "particles_median": statistics.median(int(row["particles_completed"]) for row in rows),
        "particles_p95": float(np.quantile([int(row["particles_completed"]) for row in rows], 0.95)),
        "deadline_limited_best_so_far": sum(bool(row["deadline_hit"]) for row in rows),
        "hard_timeouts": sum(float(row["elapsed"]) > compute_cap + 0.25 for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="+", type=Path)
    parser.add_argument("--oracle", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    oracles = read_oracles(args.oracle)
    modes = {}
    for path in args.benchmarks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        regrets, agreements, covered = [], 0, []
        for row in payload["results"]:
            oracle = oracles.get(str(row["state_id"]))
            if oracle is None:
                continue
            by_key = {(a["action"], int(a["slot"]), int(a["card_id"])): a for a in oracle["actions"]}
            selected = tuple(row["selected"])
            if selected not in by_key:
                continue
            best = max(oracle["actions"], key=lambda action: float(action["q_mean"]))
            best_key = (best["action"], int(best["slot"]), int(best["card_id"]))
            regret = max(0.0, float(best["q_mean"]) - float(by_key[selected]["q_mean"]))
            regrets.append(regret)
            agreements += int(selected == best_key)
            covered.append(row)
        label = (
            f"fixed-{payload['fixed_particles']}"
            if payload.get("fixed_particles")
            else f"adaptive-{payload['time_left']}s"
        )
        modes[label] = metrics(
            regrets, agreements, covered, int(payload["time_left"])
        )
    result = {
        "version": "adaptive-mode-oracle-audit-3.9-v1",
        "oracle": "independent multiseed 5000x3 q_mean where available",
        "modes": modes,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
