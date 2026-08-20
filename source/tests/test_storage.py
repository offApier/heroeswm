from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

from storage import SettingsStore, StatsTracker


def test_card_stake_and_work_priority_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    store.save(
        {
            "login": "",
            "password": "",
            "api_key": "",
            "remember": False,
            "card_max_stake": 25,
            "card_work_priority": False,
        }
    )
    restored = store.load()
    assert restored["card_max_stake"] == 25
    assert restored["card_work_priority"] is False


def test_stats_separates_session_and_day(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    first = StatsTracker(path)
    first.success()
    first.error("капча")
    snapshot = first.snapshot()
    assert snapshot["session_success"] == 1
    assert snapshot["today_success"] == 1
    assert snapshot["session_errors"] == {"капча": 1}

    second = StatsTracker(path)
    snapshot = second.snapshot()
    assert snapshot["session_success"] == 0
    assert snapshot["today_success"] == 1
    assert snapshot["today_errors"] == {"капча": 1}


def test_active_job_timer_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    shift_end = datetime.now() + timedelta(minutes=60)
    next_attempt = shift_end + timedelta(minutes=3)
    StatsTracker(path).set_active_job(shift_end, next_attempt)

    restored = StatsTracker(path).active_job()
    assert restored == (shift_end.replace(microsecond=0), next_attempt.replace(microsecond=0))
