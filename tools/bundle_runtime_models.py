from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("horizon", type=Path)
    parser.add_argument("state_value", type=Path)
    parser.add_argument("action_policy", type=Path)
    parser.add_argument("opponent_policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "version": "policy-model-bundle-3.9-v2",
        "horizon": json.loads(args.horizon.read_text(encoding="utf-8")),
        "state_value": json.loads(args.state_value.read_text(encoding="utf-8")),
        "action_policy": json.loads(args.action_policy.read_text(encoding="utf-8")),
        "opponent_policy": json.loads(args.opponent_policy.read_text(encoding="utf-8")),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
