from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import requests
from bs4 import BeautifulSoup

from card_game import (
    CardCatalog,
    CardDefinition,
    CardDecision,
    CardGameLearner,
    CardGameBot,
    CardGameStopped,
    CardGameStakeUnavailable,
    CardGameRecorder,
    CardGameProtocol,
    CardStrategy,
    GameState,
    OpponentBelief,
    PlayerState,
    parse_game_id,
    parse_last_move,
)
from policy_runtime import PolicyRuntime


ROOT = Path(__file__).resolve().parents[1]


def catalog() -> CardCatalog:
    return CardCatalog.load(ROOT / "cards_catalog.json")


def make_state(
    hand: list[int],
    *,
    me: PlayerState | None = None,
    enemy: PlayerState | None = None,
    turn: int = 10,
    must_discard: bool = False,
) -> GameState:
    return GameState(
        game_id=1,
        turn=turn,
        is_your_turn=True,
        player_no=2,
        # Unit positions use the shortest supported room by default.  The
        # 15/30/40 budget scaling itself is covered by dedicated tests below.
        time_left=15,
        players={
            1: enemy or PlayerState(10, 10, 10, 20, 5, 2, 2, 2),
            2: me or PlayerState(10, 10, 10, 20, 5, 2, 2, 2),
        },
        hand=hand,
        must_discard=must_discard,
    )


def test_catalog_contains_all_102_cards_and_known_ids() -> None:
    cards = catalog()
    assert len(cards.cards) == 102
    assert cards[10].name == "Новшества"
    assert cards[58].name == "Монастырь"
    assert cards[101].name == "Эльфы-скауты"


def test_parse_initial_live_response() -> None:
    payload = (
        "Opponent#1111111|TestPlayer#2222222|50|150|0|"
        "1|0|2|129|2|2|2|2|2|2|7|5|7|5|7|5|5|5|20|20|"
        "6-10-101-30-7-27|0|0||0||0|0||0||0"
    )
    state = CardGameProtocol(121005261).parse(payload)
    assert state.turn == 1
    assert state.player_no == 2
    assert state.hand == [6, 10, 101, 30, 7, 27]
    assert state.me == PlayerState(5, 5, 5, 20, 5, 2, 2, 2)
    assert state.opponent == PlayerState(7, 7, 7, 20, 5, 2, 2, 2)
    assert state.nicknames[2] == "TestPlayer"
    assert state.tower_goal == 50
    assert state.resource_goal == 150


def test_parse_followup_response_keeps_metadata() -> None:
    protocol = CardGameProtocol(1)
    protocol.parse(
        "A#1|B#2|50|150|0|1|0|2|40|2|2|2|2|2|2|7|5|7|5|7|5|5|5|20|20|6-10|0|0||0||0|0||0||0"
    )
    state = protocol.parse(
        "3|0|2|40|3|3|2|2|2|2|10|5|9|11|9|7|5|5|20|20|6-37|0|0|t10-1|2|t10-|0|0||0||0"
    )
    assert state.turn == 3
    assert state.nicknames == {1: "A", 2: "B"}
    assert state.me.ore == 5
    assert state.opponent.ore == 10


def test_strategy_chooses_immediate_tower_kill() -> None:
    cards = catalog()
    strategy = CardStrategy(cards)
    state = make_state(
        [3, 8, 99],
        me=PlayerState(20, 20, 20, 20, 5, 2, 2, 2),
        enemy=PlayerState(10, 10, 10, 10, 30, 2, 2, 2),
    )
    decision = strategy.choose(state)
    assert decision.action == "turn"
    assert decision.card.id == 99


def test_strategy_chooses_immediate_tower_build_win() -> None:
    cards = catalog()
    strategy = CardStrategy(cards)
    state = make_state(
        [3, 57, 80],
        me=PlayerState(20, 20, 20, 40, 5, 2, 2, 2),
    )
    decision = strategy.choose(state)
    assert decision.card.id == 57


def test_strategy_prefers_early_production_over_small_wall() -> None:
    strategy = CardStrategy(catalog())
    state = make_state([3, 8], turn=4)
    assert strategy.choose(state).card.id == 3


def test_strategy_discards_unaffordable_low_value_card() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [27, 54],
        me=PlayerState(0, 0, 0, 20, 5, 2, 2, 2),
    )
    decision = strategy.choose(state)
    assert decision.action == "drop"
    assert decision.card.id in {27, 54}


def test_must_discard_never_plays_an_extra_card() -> None:
    strategy = CardStrategy(catalog())
    decision = strategy.choose(make_state([3, 8], must_discard=True))
    assert decision.action == "drop"


def test_learning_keeps_neutral_counts_without_false_credit(tmp_path: Path) -> None:
    path = tmp_path / "learning.json"
    learner = CardGameLearner(path)
    learner.record([("turn", 10), ("drop", 27)], won=True)
    loaded = CardGameLearner(path)
    assert loaded.preference(10) == 0
    assert loaded.preference(27) == 0
    assert loaded.data["cards"]["10"]["plays"] == 1
    assert loaded.data["cards"]["27"]["drops"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))["games"] == 1


def test_parse_game_id_from_redirect_or_html() -> None:
    assert parse_game_id("https://www.heroeswm.ru/cgame.php?gameid=123") == 123
    assert parse_game_id("https://www.heroeswm.ru/tavern.php", '<a href="cgame.php?gameid=456">') == 456


def test_parse_server_move_includes_opponent_card_and_action() -> None:
    assert parse_last_move("t37-1") == ("turn", 37, 1)
    assert parse_last_move("d83-0") == ("drop", 83, 0)
    assert parse_last_move("") is None


def test_strategy_defends_critical_tower_instead_of_small_damage() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [36, 41],
        me=PlayerState(10, 10, 10, 1, 0, 2, 2, 2),
        enemy=PlayerState(10, 10, 10, 16, 0, 2, 2, 2),
        turn=15,
    )
    decision = strategy.choose(state)
    assert decision.action == "turn"
    assert decision.card.id == 36


def test_strategy_can_discard_affordable_self_kill_card() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [97, 27],
        me=PlayerState(0, 0, 4, 2, 0, 2, 2, 2),
        enemy=PlayerState(5, 5, 5, 20, 5, 2, 2, 2),
    )
    decision = strategy.choose(state)
    assert decision.action == "drop"
    assert decision.card.id == 97


def test_recorder_saves_opponent_card_raw_response_and_candidates(tmp_path: Path) -> None:
    cards = catalog()
    recorder = CardGameRecorder(tmp_path, 77, cards)
    before = make_state([3, 8])
    before.game_id = 77
    before.is_your_turn = False
    recorder.begin(before)
    after = make_state([3, 8], enemy=PlayerState(10, 10, 10, 23, 5, 2, 3, 2), turn=11)
    after.game_id = 77
    after.last_move = "t37-0"
    after.raw = "raw|server|payload"
    event = recorder.observe(before, after)
    assert event is not None
    assert event["actor"] == "opponent"
    assert event["actor_no"] == after.opponent_no
    assert event["hand_slot"] == 0
    assert event["card_id"] == 37
    assert event["card_name"] == cards[37].name
    saved = json.loads(recorder.json_path.read_text(encoding="utf-8"))
    assert saved["events"][0]["after"]["raw_response"] == "raw|server|payload"


def test_recorder_uses_pre_move_turn_to_identify_actor(tmp_path: Path) -> None:
    cards = catalog()
    recorder = CardGameRecorder(tmp_path, 78, cards)
    before = make_state([61, 3, 8, 9, 10, 11], turn=72)
    before.is_your_turn = True
    recorder.begin(before)
    after = make_state([37, 3, 8, 9, 10, 11], turn=73)
    after.is_your_turn = False
    after.last_move = "t61-5"
    event = recorder.observe(before, after)
    assert event is not None
    assert event["actor"] == "us"
    assert event["actor_no"] == after.player_no
    assert event["hand_slot"] == 5

    opponent_before = after
    opponent_after = make_state([37, 3, 8, 9, 10, 11], turn=74)
    opponent_after.is_your_turn = True
    opponent_after.last_move = "t58-0"
    event = recorder.observe(opponent_before, opponent_after)
    assert event is not None
    assert event["actor"] == "opponent"
    assert event["actor_no"] == opponent_after.opponent_no
    assert event["hand_slot"] == 0


def test_recorder_saves_two_ply_strategy_and_candidate_diagnostics(tmp_path: Path) -> None:
    cards = catalog()
    strategy = CardStrategy(cards)
    before = make_state([3, 8])
    before.is_your_turn = True
    rankings = strategy.rank_choices(before)
    selected = rankings[0]
    recorder = CardGameRecorder(tmp_path, 79, cards, strategy.metadata())
    recorder.begin(before)
    after = make_state([3, 8], turn=before.turn + 1)
    after.is_your_turn = False
    prefix = "t" if selected.action == "turn" else "d"
    after.last_move = f"{prefix}{selected.card.id}-{selected.slot}"
    recorder.observe(before, after, selected=selected, rankings=rankings)

    saved = json.loads(recorder.json_path.read_text(encoding="utf-8"))
    assert saved["version"] == 3
    assert saved["strategy"]["version"] == "3.9.1-pwin-objective-candidate"
    assert saved["strategy"]["lookahead_plies"] == 2
    assert saved["events"][0]["selected"]["response_value"] is not None
    assert saved["events"][0]["selected"]["predicted_me"] is not None
    assert all(item["immediate_score"] is not None for item in saved["events"][0]["candidates"])
    assert all(item["response_value"] is not None for item in saved["events"][0]["candidates"])
    assert all(item["p_win"] is not None for item in saved["events"][0]["candidates"])
    assert all(item["tail_risk"] is not None for item in saved["events"][0]["candidates"])
    assert saved["events"][0]["selected"]["particle_count"] >= 1
    assert saved["events"][0]["selected"]["decision_margin"] is not None
    assert saved["events"][0]["selected"]["random_seed"]


def test_every_catalog_card_can_be_simulated_and_scored() -> None:
    cards = catalog()
    strategy = CardStrategy(cards)
    state = make_state(
        list(range(6)),
        me=PlayerState(200, 200, 200, 20, 12, 4, 4, 4),
        enemy=PlayerState(200, 200, 200, 20, 12, 4, 4, 4),
    )
    for card in cards.cards.values():
        me, enemy = strategy.simulate(card, state)
        score, reasons = strategy.score(card, state)
        assert isinstance(me, PlayerState)
        assert isinstance(enemy, PlayerState)
        assert isinstance(score, float)
        assert reasons


def test_gremlin_in_tower_deals_two_damage_and_builds_defense() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [78],
        me=PlayerState(10, 10, 10, 20, 5, 2, 2, 2),
        enemy=PlayerState(10, 10, 10, 20, 6, 2, 2, 2),
    )
    me, enemy = strategy.simulate(catalog()[78], state)
    assert (me.tower, me.wall) == (22, 9)
    assert (enemy.tower, enemy.wall) == (20, 4)


def test_strategy_blocks_conditional_direct_tower_kill() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [6, 9, 94, 5, 89, 32],
        me=PlayerState(10, 14, 13, 6, 1, 2, 3, 2),
        enemy=PlayerState(10, 0, 13, 16, 6, 2, 2, 2),
        turn=23,
    )
    rankings = strategy.rank_choices(state)
    decision = rankings[0]
    assert decision.action == "turn"
    assert decision.card.id == 6
    alternative = next(choice for choice in rankings if choice.action == "turn" and choice.card.id == 5)
    assert decision.p_lose_next_turn <= alternative.p_lose_next_turn


def test_rankings_include_play_and_discard_for_affordable_cards() -> None:
    strategy = CardStrategy(catalog())
    ranked = strategy.rank_choices(make_state([3, 8]))
    pairs = {(choice.action, choice.card.id) for choice in ranked}
    assert {("turn", 3), ("drop", 3), ("turn", 8), ("drop", 8)} <= pairs


def test_two_ply_rankings_include_training_diagnostics() -> None:
    strategy = CardStrategy(catalog())
    ranked = strategy.rank_choices(make_state([3, 8]))
    assert strategy.metadata()["lookahead_plies"] == 2
    assert strategy.metadata()["response_weight"] == 1.0
    for choice in ranked:
        assert choice.immediate_score is not None
        assert choice.response_value is not None
        assert choice.predicted_me is not None
        assert choice.predicted_opponent is not None
        assert choice.winning_replies_before is not None
        assert choice.winning_replies_after is not None
        assert choice.discard_retention >= 0


def test_preserves_matrix_when_monastery_is_far_behind() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [31, 94, 40, 21, 64, 39],
        me=PlayerState(7, 2, 12, 19, 17, 4, 1, 6),
        enemy=PlayerState(5, 35, 2, 22, 16, 3, 5, 2),
        turn=45,
    )
    rankings = strategy.rank_choices(state)
    decision = rankings[0]
    matrix_drop = next(choice for choice in rankings if choice.action == "drop" and choice.card.id == 40)
    assert decision.action == "drop"
    assert decision.card.id != 40
    assert matrix_drop.discard_retention > decision.discard_retention
    assert any("retention diagnostics" in reason for reason in matrix_drop.reasons)


def test_equal_board_discards_choose_least_valuable_card_not_first_slot() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [27, 14, 58, 30, 25, 44],
        me=PlayerState(15, 2, 12, 22, 0, 2, 2, 3),
        enemy=PlayerState(15, 8, 10, 21, 0, 2, 3, 2),
        turn=9,
        must_discard=True,
    )
    decision = strategy.choose(state)
    assert decision.action == "drop"
    assert decision.card.id == 14  # Землетрясение, not first-slot Сердце дракона


def test_unreachable_production_card_does_not_lock_the_hand() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [40, 94],
        me=PlayerState(5, 0, 0, 20, 5, 2, 0, 2),
        enemy=PlayerState(5, 5, 5, 20, 5, 2, 2, 2),
        must_discard=True,
    )
    matrix_drop = next(
        choice for choice in strategy.rank_choices(state)
        if choice.action == "drop" and choice.card.id == 40
    )
    assert matrix_drop.discard_retention < 40


def test_thief_steals_half_of_resources_that_enemy_actually_has() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [91],
        me=PlayerState(10, 10, 12, 20, 5, 2, 2, 2),
        enemy=PlayerState(1, 5, 10, 20, 5, 2, 2, 2),
    )
    me, enemy = strategy.simulate(catalog()[91], state)
    assert (me.ore, me.mana, me.army) == (11, 13, 0)
    assert (enemy.ore, enemy.mana) == (0, 0)


def test_thief_caps_stolen_resources_and_rounds_halves_up() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [91],
        me=PlayerState(10, 10, 12, 20, 5, 2, 2, 2),
        enemy=PlayerState(20, 20, 10, 20, 5, 2, 2, 2),
    )
    me, enemy = strategy.simulate(catalog()[91], state)
    assert (me.ore, me.mana) == (13, 15)
    assert (enemy.ore, enemy.mana) == (15, 10)


def test_move_delay_is_three_to_ten_seconds_with_normal_timer() -> None:
    assert CardGameBot._move_delay(40, lambda low, high: low) == 3.0
    assert CardGameBot._move_delay(40, lambda low, high: high) == 10.0


def test_move_delay_keeps_timer_safety_margin() -> None:
    assert CardGameBot._move_delay(7, lambda low, high: high) == 4.5
    assert CardGameBot._move_delay(2, lambda low, high: high) == 0.15


def test_state_delta_explains_visible_board_changes() -> None:
    before = make_state([3], me=PlayerState(5, 5, 5, 20, 5, 2, 2, 2))
    after = make_state([3], me=PlayerState(5, 5, 5, 23, 8, 2, 3, 2))
    text = CardGameBot._state_delta(before, after)
    assert "наша башня +3" in text
    assert "наша стена +3" in text
    assert "наш монастырь +1" in text


def test_strategy_prefers_pressure_when_enemy_can_build_win_next_turn() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [3, 41],
        me=PlayerState(10, 10, 10, 25, 5, 2, 2, 2),
        enemy=PlayerState(10, 16, 10, 40, 5, 2, 2, 2),
        turn=20,
    )
    assert strategy.choose(state).card.id == 41


def test_regression_a_symmetric_army_loss_uses_real_clamped_losses() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [67, 41],
        me=PlayerState(5, 2, 9, 20, 0, 2, 2, 2),
        enemy=PlayerState(5, 5, 2, 20, 0, 2, 2, 2),
    )
    cow_me, cow_enemy = strategy.simulate(catalog()[67], state)
    assert cow_me.army - state.me.army == -6
    assert cow_enemy.army - state.opponent.army == -2
    ranked = strategy.rank_choices(state)
    cow = next(choice for choice in ranked if choice.action == "turn" and choice.card.id == 67)
    crack = next(choice for choice in ranked if choice.action == "turn" and choice.card.id == 41)
    assert crack.p_win >= cow.p_win


def test_regression_b_symmetric_ore_loss_changes_sign_with_position() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [0],
        me=PlayerState(4, 5, 5, 20, 5, 2, 2, 2),
        enemy=PlayerState(11, 5, 5, 20, 5, 2, 2, 2),
    )
    me, enemy = strategy.simulate(catalog()[0], state)
    assert me.ore - state.me.ore == -4
    assert enemy.ore - state.opponent.ore == -8


def test_regression_c_near_future_stone_eaters_is_not_minimum_value_discard() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [90, 60, 27],
        me=PlayerState(0, 0, 9, 20, 5, 2, 2, 2),
        enemy=PlayerState(10, 10, 10, 20, 5, 3, 3, 3),
        must_discard=True,
    )
    ranked = strategy.rank_choices(state)
    decision = ranked[0]
    stone = next(choice for choice in ranked if choice.card.id == 90)
    assert decision.card.id != 90
    assert stone.eta_key_hand_cards is not None
    assert strategy._turns_until_affordable(catalog()[90], state.me) == 1
    assert stone.discard_retention > decision.discard_retention


def test_regression_d_meditation_is_survival_move_at_tower_eight() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [66, 8, 41],
        me=PlayerState(10, 18, 10, 8, 0, 2, 3, 2),
        enemy=PlayerState(10, 10, 14, 22, 0, 2, 2, 4),
    )
    assert strategy.choose(state).card.id == 66


def test_regression_e_fairy_searches_pegasus_extra_turn_finish() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [68, 99, 8],
        me=PlayerState(5, 5, 19, 20, 0, 2, 2, 2),
        enemy=PlayerState(5, 5, 5, 14, 0, 2, 2, 2),
    )
    decision = strategy.choose(state)
    assert decision.card.id == 68
    assert "Всадник" in decision.extra_turn_continuation
    assert decision.p_win == 1.0


def test_regression_f_snakes_reserve_mana_for_empathy_instead_of_matrix_rule() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [40, 76, 56],
        me=PlayerState(8, 10, 6, 38, 3, 2, 4, 2),
        enemy=PlayerState(8, 21, 8, 28, 3, 2, 4, 2),
        turn=28,
    )
    decision = strategy.choose(state)
    assert decision.card.id == 76
    assert "Эмпатия" in decision.cards_unlocked_next_turn


def test_regression_g_hidden_dragon_eye_is_probability_not_certainty() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [8, 9, 3],
        me=PlayerState(8, 8, 8, 46, 5, 2, 2, 2),
        enemy=PlayerState(8, 21, 8, 35, 5, 2, 4, 2),
    )
    particles = strategy.belief.particles(state)
    probability = strategy.belief.probabilities(particles)[60]
    assert 0.0 < probability < 0.20
    decision = strategy.choose(state)
    assert decision.p_lose_next_turn is not None
    assert decision.p_lose_next_turn < 0.50
    assert decision.p_win is not None and decision.p_win > 0.05


def test_regression_h_preserves_three_mana_to_unlock_hardening() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [61, 37, 3],
        me=PlayerState(3, 3, 5, 25, 5, 2, 5, 2),
        enemy=PlayerState(8, 8, 8, 24, 5, 2, 3, 2),
    )
    rankings = strategy.rank_choices(state)
    preserve = next(choice for choice in rankings if choice.action == "turn" and choice.card.id == 3)
    spend = next(choice for choice in rankings if choice.action == "turn" and choice.card.id == 37)
    assert "Отвердение" in preserve.cards_unlocked_next_turn
    assert preserve.p_win > spend.p_win


def test_regression_i_rainbow_beats_discord_without_destruction_finish() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [52, 63],
        me=PlayerState(5, 5, 5, 44, 5, 2, 3, 2),
        enemy=PlayerState(5, 5, 5, 24, 5, 2, 3, 2),
    )
    decision = strategy.choose(state)
    assert decision.card.id == 63
    discord_me, discord_enemy = strategy.simulate(catalog()[52], state)
    rainbow_me, rainbow_enemy = strategy.simulate(catalog()[63], state)
    assert (discord_me.tower, discord_enemy.tower) == (37, 17)
    assert (rainbow_me.tower, rainbow_enemy.tower) == (45, 25)


def test_regression_j_wall_is_valuable_against_probable_physical_damage() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [32, 3, 41],
        me=PlayerState(14, 2, 5, 32, 0, 2, 2, 2),
        enemy=PlayerState(8, 2, 22, 20, 0, 2, 2, 5),
        turn=24,
    )
    decision = strategy.choose(state)
    assert decision.card.id == 32


def test_belief_particles_exclude_our_hand_and_observed_cards() -> None:
    strategy = CardStrategy(catalog())
    before = make_state([60, 61, 62])
    before.is_your_turn = False
    after = make_state([60, 61, 62], turn=before.turn + 1)
    after.last_move = "t90-0"
    strategy.observe_transition(before, after)
    particles = strategy.belief.particles(after)
    assert particles
    assert all(not ({60, 61, 62, 90} & set(hand)) for hand in particles)


def test_regression_n_played_card_is_impossible_during_empirical_cooldown() -> None:
    strategy = CardStrategy(catalog())
    strategy.reset_game(701)
    before = make_state([1, 2, 3], turn=10)
    before.is_your_turn = False
    after = make_state([1, 2, 3], turn=11)
    after.last_move = "t90-0"
    strategy.observe_transition(before, after)
    during_cooldown = make_state([1, 2, 3], turn=55)
    assert strategy.belief.card_age(90, during_cooldown) == 44
    assert strategy.belief.return_weight(90, during_cooldown) == 0.0
    assert all(90 not in hand for hand in strategy.belief.particles(during_cooldown, 200))


def test_regression_o_discarded_card_uses_the_same_cooldown() -> None:
    strategy = CardStrategy(catalog())
    strategy.reset_game(702)
    before = make_state([1, 2, 3], turn=20)
    after = make_state([4, 2, 3], turn=21)
    after.last_move = "d1-0"
    strategy.observe_transition(before, after)
    assert strategy.belief.return_weight(1, make_state([4, 2, 3], turn=65)) == 0.0


def test_regression_p_card_returns_gradually_after_full_cycle() -> None:
    strategy = CardStrategy(catalog())
    strategy.reset_game(703)
    before = make_state([1, 2, 3], turn=10)
    after = make_state([4, 2, 3], turn=11)
    after.last_move = "t1-0"
    strategy.observe_transition(before, after)
    eligible = make_state([4, 2, 3], turn=56)
    early_weight = strategy.belief.return_weight(1, eligible)
    mature_weight = strategy.belief.return_weight(1, make_state([4, 2, 3], turn=76))
    assert 0.0 < early_weight < mature_weight <= 1.0
    assert 1 in strategy.belief.unseen_pool(eligible)


def test_empirical_return_curve_uses_age_weights_not_uniform_reentry() -> None:
    strategy = CardStrategy(catalog())
    strategy.reset_game(7031)
    before = make_state([1, 2, 3], turn=10)
    after = make_state([4, 2, 3], turn=11)
    after.last_move = "t1-0"
    strategy.observe_transition(before, after)
    assert strategy.belief.return_weight(1, make_state([4, 2, 3], turn=55)) == 0.0
    assert strategy.belief.return_weight(1, make_state([4, 2, 3], turn=56)) == pytest.approx(0.06)
    assert strategy.belief.return_weight(1, make_state([4, 2, 3], turn=57)) == pytest.approx(0.12)
    assert strategy.belief.return_weight(1, make_state([4, 2, 3], turn=66)) == pytest.approx(0.53)


def test_regression_q_card_can_switch_owner_after_cycle() -> None:
    strategy = CardStrategy(catalog())
    strategy.reset_game(704)
    before = make_state([1, 2, 3], turn=30)
    before.is_your_turn = False
    after = make_state([1, 2, 3], turn=31)
    after.last_move = "t90-0"
    strategy.observe_transition(before, after)
    later = make_state([1, 2, 3], turn=90)
    assert strategy.belief.last_seen[90]["owner"] == "opponent"
    assert strategy.belief.return_weight(90, later) > 0.0
    assert any(90 in hand for hand in strategy.belief.particles(later, 300))


def test_regression_r_server_action_counter_survives_extra_turn_sequence() -> None:
    strategy = CardStrategy(catalog())
    strategy.reset_game(705)
    before = make_state([68, 1, 2], turn=10)
    after = make_state([3, 1, 2], turn=11)
    after.last_move = "t68-0"
    strategy.observe_transition(before, after)
    extra_before = make_state([3, 1, 2], turn=11)
    extra_after = make_state([4, 1, 2], turn=12)
    extra_after.last_move = "t3-0"
    strategy.observe_transition(extra_before, extra_after)
    assert strategy.belief.current_action == 12
    assert strategy.belief.card_age(68, make_state([4, 1, 2], turn=55)) == 44


def test_regression_s_near_tie_increases_sampling_and_reports_uncertainty() -> None:
    strategy = CardStrategy(catalog())
    strategy.MAX_PARTICLES = 400
    state = make_state([3, 3], must_discard=True)
    ranking = strategy.rank_choices(state)
    assert 200 <= ranking[0].particle_count <= 400
    assert ranking[0].sampling_batches >= 1
    assert not ranking[0].decision_uncertain
    assert ranking[0].stopping_reason == "practical_equivalence"
    assert ranking[0].decision_margin == pytest.approx(0.0)
    assert not ranking[0].analysis_deadline_hit


def candidate_strategy(*, particles: int = 40) -> CardStrategy:
    result = CardStrategy(catalog())
    result.PARTICLE_COUNT = particles
    result.MIN_PARTICLES = particles
    result.PARTICLE_BATCH = particles
    result.MAX_PARTICLES = particles
    result.MAX_ANALYSIS_SECONDS = 30.0
    return result


def test_regression_u_dead_hand_cycling_is_a_full_action() -> None:
    strategy = candidate_strategy()
    state = make_state(
        [84, 85, 86, 89, 94, 99],
        me=PlayerState(4, 4, 0, 20, 3, 2, 2, 1),
    )
    ranking = strategy.rank_choices(state)
    assert ranking[0].action == "drop"
    assert ranking[0].replacement_distribution
    assert ranking[0].hand_diagnostics["hand_playable_count"] == 0


def test_regression_v_playable_symmetric_loss_can_be_worse_than_discard() -> None:
    strategy = candidate_strategy()
    state = make_state(
        [0, 84, 85, 86, 89, 94],
        me=PlayerState(9, 0, 0, 20, 3, 2, 2, 1),
        enemy=PlayerState(2, 8, 8, 20, 3, 2, 2, 2),
    )
    ranking = strategy.rank_choices(state)
    play = next(choice for choice in ranking if choice.action == "turn" and choice.card.id == 0)
    drop = next(choice for choice in ranking if choice.action == "drop" and choice.card.id == 0)
    assert drop.p_win > play.p_win


def test_regression_w_eta_one_finisher_has_retention_diagnostics() -> None:
    strategy = candidate_strategy()
    state = make_state(
        [90, 60, 27],
        me=PlayerState(0, 0, 9, 20, 5, 2, 2, 2),
        enemy=PlayerState(10, 10, 10, 20, 5, 3, 3, 3),
        must_discard=True,
    )
    ranking = strategy.rank_choices(state)
    stone = next(choice for choice in ranking if choice.card.id == 90)
    assert strategy._turns_until_affordable(catalog()[90], state.me) == 1
    assert stone.discard_retention > 0.0
    assert ranking[0].card.id != 90


def test_regression_x_resource_congestion_is_reported_without_hard_rule() -> None:
    strategy = candidate_strategy()
    state = make_state(
        [84, 85, 86, 87, 88, 89],
        me=PlayerState(20, 20, 0, 22, 4, 3, 3, 1),
    )
    diagnostic = strategy._hand_quality_diagnostics(state)
    assert diagnostic["hand_resource_congestion"] == 1.0
    assert diagnostic["hand_same_resource_congestion"] == {"army": 6}
    assert diagnostic["hand_cycle_value"] > 0.5


def test_regression_y_terminal_cycling_keeps_stochastic_distribution() -> None:
    strategy = candidate_strategy()
    state = make_state(
        [37, 84, 85, 86, 89, 94],
        me=PlayerState(3, 3, 1, 9, 0, 1, 1, 1),
        enemy=PlayerState(15, 15, 15, 49, 2, 3, 3, 3),
    )
    ranking = strategy.rank_choices(state)
    producer_drop = next(choice for choice in ranking if choice.action == "drop" and choice.card.id == 37)
    assert producer_drop.replacement_distribution
    assert producer_drop.p_lose_next_turn is not None


def test_regression_z_discarded_card_is_in_immediate_cooldown() -> None:
    strategy = candidate_strategy()
    state = make_state([0, 1, 2, 3, 4, 5], turn=10)
    opponent = (90, 91, 92, 93, 94, 95)
    next_hand, next_seen, replacement = strategy._transition_hand_with_draw(
        state,
        tuple(state.hand),
        opponent,
        0,
        0,
        {},
        11,
        0,
        ("test-root",),
    )
    assert next_seen[0] == 11
    assert replacement != 0
    assert 0 not in next_hand


def test_regression_aa_replacement_is_conditioned_on_particle() -> None:
    strategy = candidate_strategy()
    state = make_state([0, 1, 2, 3, 4, 5], turn=10)
    first = strategy._counterfactual_replacement(
        state, (0, 1, 2, 3, 4), (6, 7, 8, 9, 10, 11), {5: 11}, 11, 0, ("same",)
    )
    assert first is not None
    second = strategy._counterfactual_replacement(
        state, (0, 1, 2, 3, 4), (first, 7, 8, 9, 10, 11), {5: 11}, 11, 0, ("same",)
    )
    assert second is not None
    assert first != second


def test_regression_ab_only_selected_slot_changes() -> None:
    strategy = candidate_strategy()
    state = make_state([0, 1, 2, 3, 4, 5], turn=10)
    next_hand, _seen, replacement = strategy._transition_hand_with_draw(
        state,
        tuple(state.hand),
        (90, 91, 92, 93, 94, 95),
        3,
        3,
        {},
        11,
        0,
        ("slot",),
    )
    assert {0, 1, 2, 4, 5}.issubset(next_hand)
    assert next_hand[3] == replacement


def test_regression_ac_scouts_searches_forced_discard_then_continuation() -> None:
    strategy = candidate_strategy()
    state = make_state(
        [101, 3, 8, 84, 85, 86],
        me=PlayerState(10, 10, 8, 20, 5, 2, 2, 2),
    )
    first_me, first_enemy = strategy.simulate(catalog()[101], state)
    hand, seen, _replacement = strategy._transition_hand_with_draw(
        state,
        tuple(state.hand),
        (90, 91, 92, 93, 94, 95),
        0,
        101,
        {},
        state.turn + 1,
        0,
        ("scouts",),
    )
    _me, _enemy, _hand, sequence = strategy._our_extra_continuation_with_draws(
        first_me,
        first_enemy,
        hand,
        (90, 91, 92, 93, 94, 95),
        state,
        0,
        state.turn + 1,
        seen,
        3,
        forced_discard=True,
    )
    assert sequence
    assert sequence[0].startswith("СБРОС ")
    assert len(sequence) >= 2


def test_regression_ad_uncertain_tie_has_no_play_type_preference() -> None:
    strategy = candidate_strategy(particles=80)
    state = make_state([3, 3], must_discard=True)
    ranking = strategy.rank_choices(state)
    assert ranking[0].decision_margin == pytest.approx(0.0)
    assert not ranking[0].decision_uncertain
    assert ranking[0].stopping_reason == "practical_equivalence"
    assert ranking[0].slot == 0


def test_regression_ae_tail_survival_participates_in_tie_break() -> None:
    strategy = candidate_strategy()
    state = make_state([3, 8])
    card = catalog()[3]
    safer = CardDecision("drop", 0, card, 50.0, tuple(), p_win=0.5, p_lose_next_turn=0.1, tail_risk=0.1)
    riskier = CardDecision("turn", 0, card, 50.0, tuple(), p_win=0.5, p_lose_next_turn=0.2, tail_risk=0.2)
    assert strategy._choice_sort_key(safer) > strategy._choice_sort_key(riskier)


def test_final_objective_does_not_let_policy_override_material_pwin_gap() -> None:
    strategy = candidate_strategy()
    card = catalog()[3]
    policy_favorite = CardDecision(
        "drop", 0, card, 25.0, tuple(), p_win=0.25, policy_score=10.0
    )
    pwin_favorite = CardDecision(
        "turn", 0, card, 47.0, tuple(), p_win=0.47, policy_score=-10.0
    )
    assert strategy._choice_sort_key(pwin_favorite) > strategy._choice_sort_key(policy_favorite)
    ranked = strategy._with_diagnostic_ranks([policy_favorite, pwin_favorite])
    assert ranked[0].action == "turn"
    assert ranked[0].pwin_rank == 1
    assert ranked[0].policy_rank == 2


def test_regression_af_synchronization_never_advances_beyond_visible_state() -> None:
    strategy = candidate_strategy()
    state = make_state([1, 2, 3], turn=15)
    strategy.synchronize_state(state)
    assert strategy.belief.current_action == state.turn


def test_regression_t_same_state_and_seed_is_reproducible() -> None:
    state = make_state([3, 8, 41])
    first = CardStrategy(catalog())
    second = CardStrategy(catalog())
    first.MAX_PARTICLES = second.MAX_PARTICLES = 400
    first_ranking = first.rank_choices(state)
    second_ranking = second.rank_choices(state)
    assert [(item.action, item.card.id) for item in first_ranking] == [
        (item.action, item.card.id) for item in second_ranking
    ]
    assert [item.p_win for item in first_ranking] == pytest.approx([item.p_win for item in second_ranking])
    assert first_ranking[0].random_seed == second_ranking[0].random_seed


def test_mc_prefix_is_invariant_to_server_mode_batch_size() -> None:
    state = make_state([3, 8, 41])
    state.time_left = 40
    short = candidate_strategy(particles=40)
    long = candidate_strategy(particles=40)
    short.configure_move_timeout(15)
    long.configure_move_timeout(30)
    short_ranking = short.rank_choices(state)
    long_ranking = long.rank_choices(state)
    assert [(item.action, item.slot) for item in short_ranking] == [
        (item.action, item.slot) for item in long_ranking
    ]
    assert [item.p_win for item in short_ranking] == pytest.approx(
        [item.p_win for item in long_ranking], abs=1e-12
    )


def test_expired_wall_clock_budget_returns_best_so_far_before_move_timeout() -> None:
    strategy = CardStrategy(catalog())
    state = make_state([3, 8, 41])
    state.time_left = 4
    started = time.monotonic()
    ranking = strategy.rank_choices(state)
    elapsed = time.monotonic() - started
    assert ranking
    assert ranking[0].analysis_deadline_hit
    assert ranking[0].particle_count <= strategy.PARTICLE_BATCH
    assert ranking[0].particles_requested >= ranking[0].particles_completed >= 1
    assert ranking[0].stopping_reason in {"deadline_precheck", "deadline_in_batch", "emergency_fallback"}
    assert ranking[0].deadline_remaining is not None
    assert elapsed < 1.5


@pytest.mark.parametrize(
    ("server_timeout", "expected_budget", "expected_particle_limit"),
    [(15, 11.0, 1600), (30, 26.0, 4000), (40, 36.0, 6000)],
)
def test_monte_carlo_budget_scales_with_selected_server_timeout(
    server_timeout: int,
    expected_budget: float,
    expected_particle_limit: int,
) -> None:
    strategy = CardStrategy(catalog())
    strategy.configure_move_timeout(server_timeout)
    state = make_state([3, 8, 41])
    # The first server response can expose the total-game timer (119/121).
    # It must never enlarge the selected per-move mode.
    state.time_left = 121
    budget, particle_limit, effective_timeout = strategy._analysis_limits(state)
    assert effective_timeout == server_timeout
    assert budget == expected_budget
    assert particle_limit == expected_particle_limit


def test_monte_carlo_budget_shrinks_when_actual_turn_time_is_depleted() -> None:
    strategy = CardStrategy(catalog())
    strategy.configure_move_timeout(40)
    state = make_state([3, 8, 41])
    state.time_left = 9
    budget, particle_limit, effective_timeout = strategy._analysis_limits(state)
    assert effective_timeout == 40
    assert budget == 5.0
    assert particle_limit == 1000


def test_card_bot_propagates_selected_server_timeout_to_strategy() -> None:
    bot = CardGameBot(
        requests.Session(),
        login_name="tester",
        ensure_login=lambda: None,
        timeout=30,
    )
    assert bot.timeout == 30
    assert bot.strategy.configured_move_timeout == 30


def test_reconnect_keeps_and_reloads_persisted_deck_history(tmp_path: Path) -> None:
    belief_file = tmp_path / "belief.json"
    first = CardStrategy(catalog(), belief_state_path=belief_file)
    first.reset_game(706)
    before = make_state([1, 2, 3], turn=10)
    before.game_id = 706
    after = make_state([1, 2, 3], turn=11)
    after.game_id = 706
    after.last_move = "t90-0"
    first.observe_transition(before, after)
    first.reset_game(706)
    assert first.belief.last_seen[90]["action"] == 11

    restored = CardStrategy(catalog(), belief_state_path=belief_file)
    restored.reset_game(706)
    assert restored.belief.last_seen[90]["action"] == 11
    assert restored.belief.return_weight(90, make_state([1, 2, 3], turn=20)) == 0.0


def test_reconnect_reconstructs_segments_and_ignores_terminal_poll_artifact(tmp_path: Path) -> None:
    history_root = tmp_path / "card_games"
    history_root.mkdir()
    valid_before = make_state([1, 2, 3], turn=10)
    valid_before.is_your_turn = False
    valid_after = make_state([1, 2, 3], turn=11)
    valid_after.last_move = "t90-0"
    artifact_before = make_state([1, 2, 3], turn=11)
    artifact_before.last_move = "t91-0"
    artifact_after = make_state([1, 2, 3], turn=12)
    artifact_after.last_move = "t91-0"
    recorder = CardGameRecorder(tmp_path, 707, catalog())
    payload = {
        "version": 3,
        "game_id": 707,
        "started_at": "2026-08-20T10:00:00",
        "events": [
            {
                "turn": 11,
                "actor": "opponent",
                "server_move": "t90-0",
                "before": recorder._state(valid_before, include_raw=False),
                "after": recorder._state(valid_after, include_raw=False),
            },
            {
                "turn": 12,
                "actor": "opponent",
                "server_move": "t91-0",
                "before": recorder._state(artifact_before, include_raw=False),
                "after": recorder._state(artifact_after, include_raw=False),
            },
        ],
    }
    (history_root / "2026-08-20_10-00-00_game_707.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    restored = CardStrategy(catalog(), history_root=history_root)
    restored.reset_game(707)
    assert restored.belief.last_seen[90]["action"] == 11
    assert 91 not in restored.belief.last_seen


def test_reconnect_turn_jump_resamples_missing_play_discard_transitions(tmp_path: Path) -> None:
    belief_file = tmp_path / "belief.json"
    strategy = CardStrategy(catalog(), belief_state_path=belief_file)
    strategy.reset_game(708)

    before = make_state([1, 2, 3], turn=10)
    before.game_id = 708
    after = make_state([1, 2, 3], turn=11)
    after.game_id = 708
    after.last_move = "t90-0"
    strategy.observe_transition(before, after)

    # The client is disconnected while actions 12, 13 and 14 happen.  The
    # compact reconnect response exposes only the tail DISCARD at action 15.
    reconnect = make_state([1, 2, 3], turn=15)
    reconnect.game_id = 708
    reconnect.last_move = "d92-4"
    resync = strategy.synchronize_state(reconnect)
    assert resync["missing_actions"] == 3
    assert strategy.belief.unknown_action_indices.issuperset({12, 13, 14})
    assert strategy.belief.last_seen[92]["action"] == 15
    assert strategy.belief.last_seen[92]["event"] == "drop"

    latent_cards: set[int] = set()

    def deterministic_latent_choice(_rng, pool):
        chosen = pool[0][0]
        latent_cards.add(chosen)
        return chosen

    strategy.belief._weighted_choice = deterministic_latent_choice
    particles = strategy.belief.particles(reconnect, 20)
    assert latent_cards
    # Every sampled missing action puts that card back onto cooldown, so it
    # cannot simultaneously occur in the reconstructed current hidden hand.
    assert all(not (set(hand) & latent_cards) for hand in particles)

    restored = CardStrategy(catalog(), belief_state_path=belief_file)
    restored.reset_game(708)
    assert restored.belief.unknown_action_indices.issuperset({12, 13, 14})
    assert restored.belief.last_seen[92]["action"] == 15


def test_initial_turn_one_is_not_reported_as_a_missing_reconnect_action() -> None:
    strategy = CardStrategy(catalog())
    initial = make_state([1, 2, 3], turn=1)
    initial.game_id = 709
    resync = strategy.synchronize_state(initial)
    assert resync["missing_actions"] == 0
    assert not strategy.belief.unknown_action_indices


def test_particle_diagnostics_distinguish_requested_completed_and_ci() -> None:
    strategy = CardStrategy(catalog())
    strategy.MAX_PARTICLES = 400
    ranked = strategy.rank_choices(make_state([3, 8, 41]))
    selected = ranked[0]
    assert selected.particles_requested >= selected.particles_completed >= 1
    assert selected.particle_count == selected.particles_completed
    assert selected.particle_limit >= selected.particle_count
    assert selected.stopping_reason in {
        "statistical_convergence",
        "terminal",
        "practical_equivalence",
        "max_particles",
        "deadline_precheck",
        "deadline_in_batch",
        "emergency_fallback",
    }
    assert selected.deadline_remaining is not None and selected.deadline_remaining >= 0.0
    assert selected.decision_margin is not None
    if selected.se_diff is not None:
        assert selected.se_diff >= 0.0
        assert selected.ci_diff is not None
        assert selected.ci_diff[0] <= selected.decision_margin <= selected.ci_diff[1]


def test_probability_diagnostics_are_calibrated_and_complete() -> None:
    strategy = CardStrategy(catalog())
    ranked = strategy.rank_choices(make_state([3, 8, 41]))
    assert ranked
    for choice in ranked:
        assert 0.0 <= choice.score <= 100.0
        assert choice.p_win is not None and 0.0 <= choice.p_win <= 1.0
        assert choice.p_win_next_action is not None and 0.0 <= choice.p_win_next_action <= 1.0
        assert choice.p_lose_next_turn is not None and 0.0 <= choice.p_lose_next_turn <= 1.0
        assert choice.p_win_within_2_own_actions is not None
        assert choice.p_opponent_win_within_2_actions is not None
        assert choice.expected_reply_value is not None
        assert choice.tail_risk is not None and 0.0 <= choice.tail_risk <= 1.0
        assert choice.immediate_state_delta is not None
        assert choice.eta_key_hand_cards is not None
        assert choice.particles_requested >= choice.particles_completed >= 1
        assert choice.stopping_reason
        assert choice.policy_rank >= 1
        assert choice.pwin_rank >= 1
        assert choice.stopping_objective.startswith("risk-adjusted P(win)")
        assert choice.model_policy_uncertainty.startswith("not estimated")


def make_bot(*, max_stake: int = 0, work_pause_callback=None) -> CardGameBot:
    return CardGameBot(
        requests.Session(),
        login_name="tester",
        ensure_login=lambda: None,
        max_stake=max_stake,
        work_pause_callback=work_pause_callback,
    )


def test_live_tavern_form_selects_highest_stake_within_user_cap() -> None:
    html = """
    <form action="create_card_game.php">
      <input type="hidden" name="sign" value="abc">
      <select name="timeout"><option value="15">15</option><option value="40">40</option></select>
      <select name="ktype"><option value="1">one</option><option value="2">infinite</option></select>
      <label>Ставка <select name="bet"><option value="5">5</option><option value="10">10</option></select></label>
    </form>
    """
    action, params, stake = make_bot(max_stake=10)._create_game_request(
        "https://www.heroeswm.ru/tavern.php?form=1", html
    )
    assert action == "https://www.heroeswm.ru/create_card_game.php"
    assert params == {"sign": "abc", "timeout": "40", "ktype": "1", "bet": "10"}
    assert stake == 10


def test_live_tavern_form_distinguishes_internal_gold_code_from_real_stake() -> None:
    html = """
    <form action="create_card_game.php">
      <select name="gold">
        <option value="1">40 золота</option>
        <option value="2">100 золота</option>
        <option value="3">300 золота</option>
        <option value="4">600 золота</option>
        <option value="5">1000 золота</option>
      </select>
    </form>
    """
    _action, params, stake = make_bot(max_stake=40)._create_game_request(
        "https://www.heroeswm.ru/tavern.php?form=1", html
    )
    assert params["gold"] == "1"
    assert stake == 40


def test_nonzero_stake_is_never_selected_above_user_cap() -> None:
    html = """
    <form action="create_card_game.php">
      <label>Ставка <select name="bet"><option value="5">5</option><option value="10">10</option></select></label>
    </form>
    """
    with pytest.raises(CardGameStakeUnavailable, match="минимум 5"):
        make_bot(max_stake=0)._create_game_request("https://www.heroeswm.ru/tavern.php?form=1", html)


def test_coded_stake_reports_real_minimum() -> None:
    html = """
    <form action="create_card_game.php">
      <select name="gold"><option value="1">40 золота</option></select>
    </form>
    """
    with pytest.raises(CardGameStakeUnavailable, match="минимум 40"):
        make_bot(max_stake=0)._create_game_request("https://www.heroeswm.ru/tavern.php?form=1", html)


def test_invitation_stake_is_read_before_accepting() -> None:
    soup = BeautifulSoup(
        '<table><tr><td>Ставка: 25</td><td><a href="acard_game.php?id=7">Принять</a></td></tr></table>',
        "html.parser",
    )
    assert CardGameBot._offer_stake(soup.a) == 25


def test_invitation_stake_is_read_from_gold_icon_cell() -> None:
    soup = BeautifulSoup(
        '<table><tr><td><img src="i/gold.gif" alt="Золото"> 100</td>'
        '<td><a href="acard_game.php?id=7">Принять</a></td></tr></table>',
        "html.parser",
    )
    assert CardGameBot._offer_stake(soup.a) == 100


def test_work_priority_callback_is_observed() -> None:
    bot = make_bot(work_pause_callback=lambda: True)
    assert bot._work_pause_requested()


def test_work_priority_cancels_waiting_application_before_new_game(monkeypatch) -> None:
    pauses = iter((True, False))
    bot = make_bot(work_pause_callback=lambda: next(pauses, False))
    calls: list[str] = []
    tavern_reads = 0

    class FakeResponse:
        def __init__(self, url: str, html: str = "") -> None:
            self.url = url
            self.content = html.encode("cp1251")

    def fake_get(url: str, **_kwargs):
        nonlocal tavern_reads
        calls.append(url)
        if url.endswith("cancel_card_game.php"):
            return FakeResponse(url)
        if url.endswith("tavern.php?form=1"):
            return FakeResponse(
                url,
                '<form action="create_card_game.php"><select name="bet"><option value="0">0</option></select></form>',
            )
        if url.endswith("create_card_game.php"):
            return FakeResponse("https://www.heroeswm.ru/cgame.php?gameid=777")
        tavern_reads += 1
        html = '<a href="cancel_card_game.php">Отменить</a>' if tavern_reads == 1 else ""
        return FakeResponse(url, html)

    monkeypatch.setattr(bot, "_get", fake_get)
    monkeypatch.setattr(bot, "_sleep", lambda _seconds: None)
    assert bot._find_or_create_game() == 777
    assert calls.index("https://www.heroeswm.ru/cancel_card_game.php") < calls.index(
        "https://www.heroeswm.ru/create_card_game.php"
    )


def test_own_application_accepts_connected_opponent_without_reparsing_stake(monkeypatch) -> None:
    bot = make_bot(max_stake=40)
    calls: list[str] = []
    logs: list[str] = []
    bot.log_callback = lambda _level, message: logs.append(message)

    class FakeResponse:
        def __init__(self, url: str, html: str = "") -> None:
            self.url = url
            self.content = html.encode("cp1251")

    def fake_get(url: str, **_kwargs):
        calls.append(url)
        if "acard_game.php?id=9" in url:
            return FakeResponse("https://www.heroeswm.ru/cgame.php?gameid=777")
        return FakeResponse(
            "https://www.heroeswm.ru/tavern.php",
            '<a href="cancel_card_game.php">Отменить свою заявку</a>'
            '<table><tr><td><b>Opponent</b></td>'
            '<td><a href="acard_game.php?id=9">Принять</a></td></tr></table>',
        )

    monkeypatch.setattr(bot, "_get", fake_get)
    assert bot._find_or_create_game() == 777
    assert any("acard_game.php?id=9" in url for url in calls)
    assert any("подключился к нашей комнате" in message.lower() for message in logs)
    assert not any("не удалось прочитать ставку" in message.lower() for message in logs)


def test_accepted_opponent_waits_for_server_gameid_without_second_room(monkeypatch) -> None:
    bot = make_bot(max_stake=40)
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, url: str, html: str = "") -> None:
            self.url = url
            self.content = html.encode("cp1251")

    waiting_html = (
        '<a href="cancel_card_game.php">Отменить свою заявку</a>'
        '<table><tr><td><b>Opponent</b></td>'
        '<td><a href="acard_game.php?id=9">Принять</a></td></tr></table>'
    )
    tavern_reads = 0

    def fake_get(url: str, **_kwargs):
        nonlocal tavern_reads
        calls.append(url)
        if "acard_game.php?id=9" in url:
            return FakeResponse("https://www.heroeswm.ru/tavern.php", "Бой принят")
        tavern_reads += 1
        if tavern_reads == 1:
            return FakeResponse("https://www.heroeswm.ru/tavern.php", waiting_html)
        return FakeResponse(
            "https://www.heroeswm.ru/tavern.php",
            '<a href="cgame.php?gameid=777">Перейти в игру</a>',
        )

    monkeypatch.setattr(bot, "_get", fake_get)
    monkeypatch.setattr(bot, "_sleep", lambda _seconds: None)
    assert bot._find_or_create_game() == 777
    assert any("acard_game.php?id=9" in url for url in calls)
    assert not any("tavern.php?form=1" in url for url in calls)
    assert not any("create_card_game.php" in url for url in calls)


def test_without_own_application_bot_creates_room_instead_of_joining_foreign_one(monkeypatch) -> None:
    bot = make_bot(max_stake=40)
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, url: str, html: str = "") -> None:
            self.url = url
            self.content = html.encode("cp1251")

    def fake_get(url: str, **_kwargs):
        calls.append(url)
        if url.endswith("tavern.php?form=1"):
            return FakeResponse(
                url,
                '<form action="create_card_game.php"><select name="gold">'
                '<option value="1">40 золота</option></select></form>',
            )
        if "create_card_game.php" in url:
            return FakeResponse("https://www.heroeswm.ru/cgame.php?gameid=888")
        return FakeResponse(
            "https://www.heroeswm.ru/tavern.php",
            '<table><tr><td><a href="acard_game.php?id=12">Принять чужую</a></td></tr></table>',
        )

    monkeypatch.setattr(bot, "_get", fake_get)
    assert bot._find_or_create_game() == 888
    assert not any("acard_game.php?id=12" in url for url in calls)
    assert any("create_card_game.php" in url for url in calls)


# Generic 3.9 policy-improvement regressions (AG-AT).  These assert mechanisms,
# not memorized historical card positions.


def test_ag_tower_damage_beats_slow_economy_in_short_terminal_horizon() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [3, 41],
        me=PlayerState(10, 10, 10, 44, 0, 2, 2, 2),
        enemy=PlayerState(10, 10, 10, 3, 0, 2, 2, 2),
        turn=58,
    )
    runtime = strategy.policy_runtime
    assert runtime is not None
    assert runtime.action_score(strategy, state, "turn", 1) > runtime.action_score(strategy, state, "turn", 0)


def test_ah_economy_has_horizon_scaled_option_value_when_safe() -> None:
    strategy = CardStrategy(catalog())
    state = make_state([3, 8], turn=3)
    horizon = strategy.policy_runtime.horizon(strategy, state, state.me, state.opponent, state.hand)
    assert horizon > 6
    assert strategy.choose(state).card.id == 3


def test_ai_self_damage_terminal_action_is_never_selected() -> None:
    strategy = CardStrategy(catalog())
    state = make_state(
        [97, 27],
        me=PlayerState(0, 0, 4, 2, 0, 2, 2, 2),
        enemy=PlayerState(5, 5, 5, 20, 5, 2, 2, 2),
    )
    assert strategy.choose(state).action == "drop"


def test_aj_wall_is_not_counted_as_protection_from_direct_damage() -> None:
    strategy = CardStrategy(catalog())
    defender = PlayerState(5, 5, 5, 4, 30, 2, 2, 2)
    attacker = PlayerState(5, 10, 5, 20, 0, 2, 2, 2)
    state = make_state([8], me=defender, enemy=attacker)
    direct = catalog()[53]
    mirrored = strategy._state_with(state, defender, attacker, opponent_actor=True)
    projected_attacker, projected_defender = strategy.simulate(direct, mirrored)
    assert projected_defender.wall == defender.wall
    assert projected_defender.tower < defender.tower


def test_ak_discard_identity_changes_retained_hand_features() -> None:
    strategy = CardStrategy(catalog())
    state = make_state([27, 14, 58, 30, 25, 44], must_discard=True)
    runtime = strategy.policy_runtime
    first = runtime.action_feature_vector(strategy, state, "drop", 0)
    second = runtime.action_feature_vector(strategy, state, "drop", 1)
    assert first != second


def test_ak2_discard_identity_changes_monte_carlo_pwin_not_only_policy_features() -> None:
    strategy = candidate_strategy(particles=40)
    state = make_state([27, 14, 58, 30, 25, 44], must_discard=True)
    choices = sorted(strategy.rank_choices(state), key=lambda choice: choice.slot)
    assert len({round(float(choice.p_win), 8) for choice in choices}) > 1
    assert choices[0].eta_key_hand_cards != choices[1].eta_key_hand_cards


def test_al_opponent_policy_is_empirical_not_only_perfect_minimax() -> None:
    runtime = PolicyRuntime.load(ROOT / "policy_models.json")
    assert runtime is not None
    model = runtime.payload["opponent_policy"]
    assert model["method"].startswith("pairwise conditional logit")
    assert model["validation"]["top2"] > model["validation"]["top1"]


def test_am_particle_collapse_reports_low_diversity_ess() -> None:
    belief = OpponentBelief(catalog(), particle_count=64)
    state = make_state([1, 2, 3, 4, 5, 6])
    particles = [(10, 11, 12, 13, 14, 15)] * 64
    diagnostic = belief.diagnostics(state, particles)
    assert diagnostic["effective_sample_size"] == pytest.approx(1.0)
    assert diagnostic["unique_opponent_hands"] == 1


def test_an_finite_particle_unanimity_is_not_false_certainty() -> None:
    particles = [(60, 10, 11, 12, 13, 14)] * 64
    assert OpponentBelief.probabilities(particles)[60] < 1.0


def test_ao_structurally_proven_card_retains_true_certainty() -> None:
    particles = [(60, 10, 11, 12, 13, 14)] * 64
    assert OpponentBelief.probabilities(particles, {60})[60] == 1.0


def test_as_learned_horizon_shortens_near_terminal_race() -> None:
    strategy = CardStrategy(catalog())
    safe = make_state([3, 8], me=PlayerState(10, 10, 10, 20, 5, 2, 2, 2), turn=5)
    terminal = make_state([3, 8], me=PlayerState(10, 10, 10, 48, 0, 2, 2, 2), turn=55)
    runtime = strategy.policy_runtime
    assert runtime.horizon(strategy, terminal, terminal.me, terminal.opponent, terminal.hand) < runtime.horizon(strategy, safe, safe.me, safe.opponent, safe.hand)


def test_at_initiative_enters_state_value() -> None:
    strategy = CardStrategy(catalog())
    state = make_state([3, 8], turn=15)
    runtime = strategy.policy_runtime
    state.first_actor = "us"
    first = runtime.state_pwin(strategy, state, state.me, state.opponent, state.hand)
    state.first_actor = "opponent"
    second = runtime.state_pwin(strategy, state, state.me, state.opponent, state.hand)
    assert first != second
