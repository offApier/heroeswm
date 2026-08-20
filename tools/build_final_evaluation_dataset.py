from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("final_split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assignments = {int(key): value for key, value in json.loads(args.final_split.read_text(encoding="utf-8"))["assignments"].items()}
    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.dataset, "rt", encoding="utf-8") as source, gzip.open(args.output, "wt", encoding="utf-8", newline="\n") as output:
        for line in source:
            row = json.loads(line)
            split = assignments[int(row["game_id"])]
            if split not in {"freeze_validation", "final_holdout"}:
                continue
            row["split"] = split
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[f"states_{split}"] += 1
            counts[f"actions_{split}"] += len(row["legal_actions"])
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
