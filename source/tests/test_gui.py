from datetime import datetime, timedelta

from gui import should_pause_cards_for_work


class LiveThread:
    @staticmethod
    def is_alive() -> bool:
        return True


class FakeWorker:
    def __init__(self, remaining: timedelta | None, *, active_cycle: bool = False) -> None:
        self._remaining = remaining
        self.next_attempt_time = None if active_cycle else datetime.now() + (remaining or timedelta(0))

    def time_until_next_attempt(self) -> timedelta | None:
        return self._remaining


def test_card_priority_accepts_timedelta_and_pauses_inside_guard() -> None:
    assert should_pause_cards_for_work(FakeWorker(timedelta(minutes=14)), LiveThread(), True)
    assert not should_pause_cards_for_work(FakeWorker(timedelta(minutes=16)), LiveThread(), True)


def test_card_priority_pauses_when_work_cycle_is_active_or_already_due() -> None:
    assert should_pause_cards_for_work(FakeWorker(None, active_cycle=True), LiveThread(), True)
    assert should_pause_cards_for_work(FakeWorker(None), LiveThread(), True)


def test_card_priority_can_be_disabled() -> None:
    assert not should_pause_cards_for_work(FakeWorker(timedelta(minutes=1)), LiveThread(), False)
