from __future__ import annotations

import math

from opponent_policy_runtime import OpponentAction, OpponentPolicyRuntime, _stable_uniform


def test_discard_rate_offset_matches_validation_odds() -> None:
    payload = {
        "weights": [0.0] * 31,
        "validation": {
            "actual_discard_rate": 0.12,
            "predicted_discard_rate_particle": 0.02,
        },
    }
    runtime = OpponentPolicyRuntime(payload)
    assert runtime.discard_logit_offset > 1.5
    assert runtime.score([0.0] * 31, action="drop") > runtime.score([0.0] * 31, action="turn")


def test_lethal_action_is_never_sampled_away() -> None:
    actions = [
        OpponentAction("turn", 0, 10, -10.0, True),
        OpponentAction("turn", 1, 11, 10.0, False),
        OpponentAction("drop", 2, 12, 12.0, False),
    ]
    for uniform in (0.0, 0.1, 0.5, 0.999999):
        assert OpponentPolicyRuntime.choose(actions, uniform).card_id == 10


def test_softmax_sampling_prefers_high_score_without_becoming_minimax() -> None:
    actions = [
        OpponentAction("turn", 0, 10, 2.0, False),
        OpponentAction("turn", 1, 11, 0.0, False),
    ]
    # Softmax P(first) = e^2 / (e^2 + 1) ~= 0.881.
    assert OpponentPolicyRuntime.choose(actions, 0.50).card_id == 10
    assert OpponentPolicyRuntime.choose(actions, 0.95).card_id == 11


def test_stable_uniform_is_reproducible_and_bounded() -> None:
    first = _stable_uniform("game", 1, (2, 3, 4))
    second = _stable_uniform("game", 1, (2, 3, 4))
    other = _stable_uniform("game", 2, (2, 3, 4))
    assert first == second
    assert 0.0 < first < 1.0
    assert first != other
