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

from types import SimpleNamespace

from opponent_policy_runtime import _select_survival_choice, _selector_risk_weight


def _choice(
    action: str,
    slot: int,
    pwin: float,
    risk: float,
    policy: float,
    *,
    win: bool = False,
):
    return SimpleNamespace(
        action=action,
        slot=slot,
        p_win=pwin,
        p_lose_next_turn=risk,
        policy_score=policy,
        immediate_terminal_win=win,
    )


def test_selector_never_chooses_known_self_terminal_action_when_safe_action_exists() -> None:
    berserk = _choice("turn", 5, 0.0084, 1.0, -1_000_000.0)
    slave_labor = _choice("turn", 1, 0.0, 0.567, -1.721)
    selected, reason, _utility, _weight = _select_survival_choice(
        [berserk, slave_labor],
        tower=2,
        unsafe_keys={("turn", 5)},
        policy_available=True,
        uncertain=True,
    )
    assert selected is slave_labor
    assert reason != "exact_terminal_win"


def test_selector_rejects_live_log_survival_dominated_discord_choice() -> None:
    discord = _choice("turn", 3, 0.0517, 0.368, 0.1094)
    dwarf = _choice("turn", 2, 0.0003, 0.005, 0.6774)
    selected, _reason, _utility, weight = _select_survival_choice(
        [discord, dwarf],
        tower=11,
        policy_available=True,
    )
    assert selected is dwarf
    assert math.isclose(weight, 0.60)


def test_selector_rejects_live_log_58pct_loss_for_small_pwin_uplift() -> None:
    scouts = _choice("turn", 3, 0.0376, 0.581, 1.5793)
    unicorn = _choice("turn", 5, 0.0012, 0.0, 1.3240)
    selected, _reason, _utility, _weight = _select_survival_choice(
        [scouts, unicorn],
        tower=32,
        policy_available=True,
    )
    assert selected is unicorn


def test_selector_keeps_large_pwin_edge_when_risk_is_not_survival_dominated() -> None:
    aggressive = _choice("turn", 0, 0.70, 0.30, 0.5)
    safe = _choice("turn", 1, 0.50, 0.0, 1.0)
    selected, _reason, _utility, weight = _select_survival_choice(
        [aggressive, safe],
        tower=30,
        policy_available=True,
    )
    assert selected is aggressive
    assert math.isclose(weight, 0.20)


def test_policy_breaks_safe_tie_only_in_saturated_zone() -> None:
    first = _choice("turn", 0, 0.0010, 0.02, 0.1)
    second = _choice("turn", 1, 0.0009, 0.02, 1.5)
    selected, reason, _utility, _weight = _select_survival_choice(
        [first, second],
        tower=20,
        policy_available=True,
    )
    assert selected is second
    assert reason == "policy_tiebreak_in_saturated_or_uncertain_zone"


def test_policy_cannot_override_material_risk_gap() -> None:
    risky_policy_favorite = _choice("turn", 0, 0.010, 0.20, 100.0)
    safe = _choice("turn", 1, 0.009, 0.0, 0.0)
    selected, _reason, _utility, _weight = _select_survival_choice(
        [risky_policy_favorite, safe],
        tower=20,
        policy_available=True,
    )
    assert selected is safe


def test_exact_terminal_win_still_has_absolute_priority() -> None:
    win = _choice("turn", 0, 1.0, 0.0, 0.0, win=True)
    other = _choice("turn", 1, 0.99, 0.0, 100.0)
    selected, reason, _utility, _weight = _select_survival_choice(
        [other, win],
        tower=1,
        policy_available=True,
        uncertain=True,
    )
    assert selected is win
    assert reason == "exact_terminal_win"


def test_uncertain_decision_increases_survival_weight() -> None:
    assert _selector_risk_weight(tower=20, top_pwin=0.5, uncertain=True) > _selector_risk_weight(
        tower=20, top_pwin=0.5, uncertain=False
    )


def test_apply_patch_reorders_engine_result_and_clamps_self_terminal(monkeypatch) -> None:
    import sys
    import types
    from dataclasses import dataclass
    import opponent_policy_runtime as runtime_module

    @dataclass(frozen=True)
    class FakeDecision:
        action: str
        slot: int
        card: object
        score: float
        p_win: float
        p_lose_next_turn: float
        expected_reply_value: float
        policy_score: float
        final_rank_reason: str
        reasons: tuple[str, ...]
        immediate_terminal_win: bool = False
        decision_uncertain: bool = True
        p_win_next_action: float = 0.0
        p_win_within_2_own_actions: float = 0.0
        p_opponent_win_within_2_actions: float = 0.0

    class FakeStrategy:
        STRATEGY_VERSION = "3.9.1"
        OOS_SERIES_ID = "old"

        def __init__(self) -> None:
            self.policy_runtime = object()
            self.last_sampling = {}

        def _opponent_reply(self, *args, **kwargs):
            return 0.5, False, False, 0.0, None

        def _aggregate_samples(self, base, samples, state, top_threats):
            return base

        def rank_choices(self, state):
            berserk = FakeDecision(
                "turn",
                5,
                types.SimpleNamespace(name="berserk"),
                0.84,
                0.0084,
                1.0,
                0.1,
                -1_000_000.0,
                "old",
                ("old",),
            )
            safe = FakeDecision(
                "turn",
                1,
                types.SimpleNamespace(name="safe"),
                0.0,
                0.0,
                0.567,
                0.0,
                -1.721,
                "old",
                ("old",),
            )
            return [berserk, safe]

        def metadata(self):
            return {"version": self.STRATEGY_VERSION}

        def simulate(self, card, state):
            if card.name == "berserk":
                return types.SimpleNamespace(tower=-1), types.SimpleNamespace(tower=11)
            return types.SimpleNamespace(tower=2), types.SimpleNamespace(tower=11)

        def _lost(self, me, enemy, state):
            return me.tower <= 0

    fake_card_game = types.ModuleType("card_game")
    fake_card_game.ProbabilisticCardStrategy = FakeStrategy
    monkeypatch.setitem(sys.modules, "card_game", fake_card_game)

    old_patched = runtime_module._PATCHED
    old_original = runtime_module._ORIGINAL_OPPONENT_REPLY
    runtime_module._PATCHED = False
    runtime_module._ORIGINAL_OPPONENT_REPLY = None
    try:
        assert runtime_module.apply_runtime_patch() is True
        strategy = FakeStrategy()
        state = types.SimpleNamespace(me=types.SimpleNamespace(tower=2))
        aggregate_base = FakeDecision(
            "turn",
            5,
            types.SimpleNamespace(name="berserk"),
            0.84,
            0.0084,
            1.0,
            0.1,
            -1_000_000.0,
            "old",
            ("old",),
        )
        clamped = strategy._aggregate_samples(aggregate_base, {}, state, tuple())
        assert clamped.p_win == 0.0
        assert clamped.p_lose_next_turn == 1.0

        ranked = strategy.rank_choices(state)
        assert ranked[0].card.name == "safe"
        berserk = next(item for item in ranked if item.card.name == "berserk")
        assert berserk.p_win == 0.0
        assert berserk.p_lose_next_turn == 1.0
        assert strategy.STRATEGY_VERSION == "3.9.3-survival-selector"
        metadata = strategy.metadata()
        assert metadata["runtime_patch_active"] is True
        assert strategy.last_sampling["selector_version"] == "3.9.3-survival-guard"
    finally:
        runtime_module._PATCHED = old_patched
        runtime_module._ORIGINAL_OPPONENT_REPLY = old_original
