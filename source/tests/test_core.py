from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

import core
from bs4 import BeautifulSoup

from core import (
    CaptchaChallenge,
    HeroesWMWorker,
    JobPage,
    LOCAL_TEXT_CAPTCHA_MAX_ATTEMPTS,
    SERVICE_TEXT_CAPTCHA_MAX_ATTEMPTS,
    TEXT_CAPTCHA_RETRY_SECONDS,
    apply_scripted_form_values,
    collect_form_fields,
    normalize_captcha_code,
    parse_clock,
    parse_home_work_status,
)
from captcha_calibration import (
    TRAINING_SAMPLE_COUNT,
    calibrated_candidate,
    calibrated_candidates,
)
from storage import StatsTracker


def make_worker() -> HeroesWMWorker:
    return HeroesWMWorker(
        "login",
        "password",
        "",
        stop_event=threading.Event(),
        captcha_mode="manual",
    )


def test_fresh_session_skips_slow_home_probe_without_player_cookie(monkeypatch) -> None:
    worker = make_worker()

    def unexpected_get(*_args, **_kwargs):
        raise AssertionError("fresh unauthenticated session must not probe /home.php")

    monkeypatch.setattr(worker.session, "get", unexpected_get)
    assert worker.is_logged_in() is False


def test_current_getjob_form_is_replayed_instead_of_old_ok_sign_payload() -> None:
    html = """
    <form id="getjob_form" action="object_do.php" method="post">
      <input type="image" id="wbtn" src="/i/getjob/btn_work.png">
      <input type="hidden" name="id" value="286">
      <input type="hidden" name="id2" value="286">
      <input type="hidden" name="idr" value="server-token">
      <input type="hidden" name="time_marker" value="time-token">
      <input type="hidden" name="other_data" value="-100">
      <input type="hidden" id="num" name="num" value="0">
      <input type="hidden" name="work_code_data_element" value="0">
      <input type="hidden" name="id3" value="server-token-3">
    </form>
    """
    form = BeautifulSoup(html, "html.parser").find("form")
    worker = make_worker()
    page = JobPage(
        "286",
        "https://www.heroeswm.ru/object-info.php?id=286",
        "ready",
        form_fields=collect_form_fields(form),
        challenge=CaptchaChallenge("none"),
    )
    payload = worker.build_job_payload(page, "")
    assert payload["idr"] == "server-token"
    assert payload["time_marker"] == "time-token"
    assert payload["id3"] == "server-token-3"
    assert "ok" not in payload
    assert "sign" not in payload
    assert json.loads(payload["other_data"])["navPlatform"] == "Win32"


def test_text_captcha_uses_real_field_and_browser_telemetry() -> None:
    html = """
    <form id="getjob_form" action="object_do.php" method="post">
      <div class="getjob_capcha"><img width="250" height="60" src="/captcha.php?id=1"></div>
      <input class="getjob_capchaInput" id="code" name="code" type="text">
      <input type="hidden" name="idr" value="fresh-token">
      <input type="hidden" id="num" name="num" value="0">
      <input type="hidden" name="work_code_data_element" value="0">
      <script>
        document.getElementById("num").value = (((((1-27)-20)-21)+25)+24)-20;
      </script>
    </form>
    """
    form = BeautifulSoup(html, "html.parser").find("form")
    worker = make_worker()
    challenge = worker._detect_challenge(form, "https://www.heroeswm.ru/object-info.php?id=1")
    assert challenge.kind == "image"
    assert challenge.field_name == "code"
    assert challenge.image_url == "https://www.heroeswm.ru/captcha.php?id=1"
    fields = collect_form_fields(form)
    apply_scripted_form_values(form, fields)
    page = JobPage(
        "1",
        "https://www.heroeswm.ru/object-info.php?id=1",
        "ready",
        form_fields=fields,
        challenge=challenge,
    )
    payload = worker.build_job_payload(page, "фЫв 123")
    assert payload["code"] == "ASD123"
    assert payload["num"] == "-38"
    telemetry = json.loads(payload["work_code_data_element"])
    assert telemetry["cur_time"] > 0
    assert any(value.get("type") == "input" for key, value in telemetry.items() if key.isdigit())
    assert any(value.get("code") == "ASD123" for key, value in telemetry.items() if key.isdigit())


def test_shift_clock_uses_actual_local_clock_without_four_hour_offset() -> None:
    now = datetime(2026, 8, 9, 1, 22)
    end = parse_clock("Окончание смены: 02:07", now)
    assert end == datetime(2026, 8, 9, 2, 7)
    assert end - now == timedelta(minutes=45)


def test_shift_clock_rolls_over_midnight() -> None:
    now = datetime(2026, 8, 9, 23, 55)
    assert parse_clock("Окончание смены 00:15", now) == datetime(2026, 8, 10, 0, 15)


def test_map_parser_reads_salary_column_not_resource_stock() -> None:
    html = """
    <table>
      <tr class="map_obj_table_hover">
        <td><a href="object-info.php?id=286">Лесопилка</a></td>
        <td>Клан</td><td>109,795</td><td><a href="object-info.php?id=286">174</a></td><td>»»»</td>
      </tr>
    </table>
    """
    parsed = HeroesWMWorker._parse_enterprises(BeautifulSoup(html, "html.parser"), "mn")
    assert len(parsed) == 1
    assert parsed[0].salary == 174
    assert parsed[0].category == "mn"


def test_captcha_normalization_matches_game_keyboard_conversion() -> None:
    assert normalize_captcha_code("йцУ-42") == "QWE42"


def test_actual_already_working_wording_has_shift_time() -> None:
    now = datetime(2026, 8, 9, 23, 52)
    text = "Окончание смены: 00:05 Вы уже устроены. Протокол работы на объекте"
    assert "Вы уже устроены" in text
    assert parse_clock(text, now) == datetime(2026, 8, 10, 0, 5)


def test_local_ocr_gets_all_captured_real_captchas_right_first() -> None:
    fixtures = Path(__file__).with_name("fixtures")
    expected = {
        "captcha_dqqq9n.png": "DQQQ9N",
        "captcha_226kqs.png": "226KQS",
        "captcha_ap4qxp.png": "AP4QXP",
    }
    worker = make_worker()
    for filename, answer in expected.items():
        assert worker._local_solve((fixtures / filename).read_bytes()) == answer


def test_twenty_sample_calibration_is_advisory_and_matches_observed_confusions() -> None:
    assert TRAINING_SAMPLE_COUNT == 20
    assert calibrated_candidate("TYVCQY") == "PYVGQY"
    assert calibrated_candidate("S425V3") == "S428V3"
    assert calibrated_candidate("CA9KBN") == "CA9K6N"
    assert calibrated_candidate("468474") == "468AH4"
    assert "4SAPUG" in calibrated_candidates("DSATUC")
    assert "56KYPP" in calibrated_candidates("56KYT7")
    assert calibrated_candidates("S425V3")[0] == "S428V3"
    assert calibrated_candidate("ABC123") == "ABC123"


def test_new_real_captchas_are_solved_within_two_local_attempts() -> None:
    fixtures = Path(__file__).with_name("fixtures")
    expected = {
        "captcha_pyvgqy.png": "PYVGQY",
        "captcha_s428v3.png": "S428V3",
        "captcha_ca9k6n.png": "CA9K6N",
        "captcha_468ah4.png": "468AH4",
    }
    worker = make_worker()
    for filename, answer in expected.items():
        image = (fixtures / filename).read_bytes()
        attempts = [worker._local_solve(image), worker._local_solve(image)]
        assert answer in attempts


def test_local_ocr_reanalyses_same_image_without_resending_rejected_code(
    monkeypatch,
) -> None:
    class FakeOcr:
        def __init__(self) -> None:
            self.calls = 0

        def classification(self, _image: bytes, probability: bool = False):
            self.calls += 1
            assert probability
            return {"text": "ABC124"}

    worker = make_worker()
    fake_ocr = FakeOcr()
    worker._local_ocr = fake_ocr
    monkeypatch.setattr(worker, "_ocr_variant_results", lambda _image: [])

    image = b"the same captcha bytes"
    assert worker._local_solve(image) == "ABC124"
    worker.last_captcha_image = image
    worker._remember_captcha_rejection("ABC124")
    assert worker._local_solve(image) is None
    assert worker._last_local_captcha_exhausted
    assert worker._local_solve(image) is None
    assert fake_ocr.calls == 3
    assert "ABC124" in worker._captcha_rejected_codes[hashlib.sha256(image).hexdigest()]


def test_text_captcha_batch_retry_is_short() -> None:
    assert TEXT_CAPTCHA_RETRY_SECONDS == 20


def test_auto_mode_uses_six_local_codes_then_paid_service(monkeypatch) -> None:
    worker = HeroesWMWorker("login", "password", "api-key", captcha_mode="auto")
    image = b"ordinary text captcha"
    worker.last_captcha_image = image
    local_codes = iter(("LOCAL1", "LOCAL2", "LOCAL3", "LOCAL4", "LOCAL5", "LOCAL6"))
    monkeypatch.setattr(worker, "_local_solve", lambda _image: next(local_codes))
    service_calls: list[bytes] = []

    def fake_service(data: bytes) -> str:
        service_calls.append(data)
        return "PAID01"

    monkeypatch.setattr(worker, "_service_solve_image", fake_service)
    assert [worker._obtain_image_code(image) for _ in range(6)] == [
        "LOCAL1", "LOCAL2", "LOCAL3", "LOCAL4", "LOCAL5", "LOCAL6"
    ]
    assert worker._obtain_image_code(image) == "PAID01"
    assert service_calls == [image]
    assert LOCAL_TEXT_CAPTCHA_MAX_ATTEMPTS == 6


def test_auto_mode_falls_back_early_when_ocr_has_no_more_codes(monkeypatch) -> None:
    worker = HeroesWMWorker("login", "password", "api-key", captcha_mode="auto")
    image = b"hard text captcha"
    monkeypatch.setattr(worker, "_local_solve", lambda _image: None)
    monkeypatch.setattr(worker, "_service_solve_image", lambda _image: "PAID02")
    assert worker._obtain_image_code(image) == "PAID02"


def test_legacy_local_setting_also_falls_back_when_api_key_exists(monkeypatch) -> None:
    worker = HeroesWMWorker("login", "password", "api-key", captcha_mode="local")
    image = b"hard text captcha with legacy settings"
    monkeypatch.setattr(worker, "_local_solve", lambda _image: None)
    monkeypatch.setattr(worker, "_service_solve_image", lambda _image: "PAID03")
    assert worker._obtain_image_code(image) == "PAID03"


def test_auto_mode_without_key_stops_after_local_limit(monkeypatch) -> None:
    worker = HeroesWMWorker("login", "password", "", captcha_mode="auto")
    image = b"ordinary text captcha without key"
    local_codes = iter(("LOCAL1", "LOCAL2", "LOCAL3", "LOCAL4", "LOCAL5", "LOCAL6"))
    monkeypatch.setattr(worker, "_local_solve", lambda _image: next(local_codes))
    monkeypatch.setattr(
        worker,
        "_service_solve_image",
        lambda _image: (_ for _ in ()).throw(AssertionError("paid service called")),
    )
    assert all(worker._obtain_image_code(image) for _ in range(6))
    assert worker._obtain_image_code(image) is None
    assert worker._last_text_captcha_source == "local_no_api"


def test_exhausted_local_captcha_without_key_does_not_use_short_loop() -> None:
    worker = HeroesWMWorker("login", "password", "", captcha_mode="auto")
    worker._last_text_captcha_source = "local_no_api"
    before = datetime.now() + timedelta(minutes=14, seconds=55)
    worker._schedule_unsolved_text_captcha()
    assert worker.next_attempt_time is not None
    assert worker.next_attempt_time >= before


def test_temporary_paid_service_failure_uses_bounded_retry() -> None:
    worker = HeroesWMWorker("login", "password", "api-key", captcha_mode="auto")
    worker._last_text_captcha_source = "service_error"
    before = datetime.now() + timedelta(minutes=1, seconds=55)
    worker._schedule_unsolved_text_captcha()
    assert worker.next_attempt_time is not None
    assert worker.next_attempt_time >= before


def test_displayed_local_remainder_is_capped_by_attempt_budget() -> None:
    worker = HeroesWMWorker("login", "password", "api-key", captcha_mode="auto")
    image = b"captcha with 55 raw candidates"
    image_key = hashlib.sha256(image).hexdigest()
    worker.last_captcha_image = image
    worker._local_candidate_cache[image_key] = [f"CODE{x:02d}" for x in range(55)]
    worker._local_attempt_counts[image_key] = 3
    assert worker._remaining_local_candidates() == 3


def test_paid_text_service_has_three_attempt_limit(monkeypatch) -> None:
    worker = HeroesWMWorker("login", "password", "api-key", captcha_mode="auto")
    image = b"service captcha"
    calls = 0

    def fake_solver(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {"code": f"PAID0{calls}", "captchaId": str(calls)}

    monkeypatch.setattr(worker, "_solve_with_fallback", fake_solver)
    assert [worker._service_solve_image(image) for _ in range(3)] == ["PAID01", "PAID02", "PAID03"]
    assert worker._service_solve_image(image) is None
    assert calls == SERVICE_TEXT_CAPTCHA_MAX_ATTEMPTS == 3


def test_real_work_code_form_is_routed_to_local_image_ocr() -> None:
    html = """
    <form class="getjob_form" id="getjob_form" action="object_do.php" method="POST">
      <img class="getjob_capcha" src="work_codes/20676-175/1234567--496241.jpeg">
      <input class="getjob_capchaInput" type="text" name="code" id="code" maxlength="6">
      <input type="hidden" name="idr" value="fresh-image-token">
    </form>
    """
    form = BeautifulSoup(html, "html.parser").find("form")
    worker = make_worker()
    challenge = worker._detect_challenge(
        form, "https://www.heroeswm.ru/object-info.php?id=131"
    )
    assert challenge.kind == "image"
    assert challenge.field_name == "code"
    assert challenge.image_url == (
        "https://www.heroeswm.ru/work_codes/20676-175/1234567--496241.jpeg"
    )


def test_real_recaptcha_form_is_routed_to_paid_challenge() -> None:
    html = """
    <form name="work" action="object_do.php" method="POST">
      <div class="g-recaptcha" data-sitekey="test-site-key"></div>
      <input type="hidden" value="378" name="id">
      <input type="hidden" value="fresh-recaptcha-token" name="idr">
    </form>
    """
    form = BeautifulSoup(html, "html.parser").find("form")
    worker = make_worker()
    challenge = worker._detect_challenge(
        form, "https://www.heroeswm.ru/object-info.php?id=378"
    )
    assert challenge.kind == "recaptcha"
    assert challenge.sitekey == "test-site-key"


def test_recaptcha_calls_paid_service_only_when_api_key_exists(monkeypatch) -> None:
    page = JobPage(
        "378",
        "https://www.heroeswm.ru/object-info.php?id=378",
        "ready",
        challenge=CaptchaChallenge("recaptcha", sitekey="test-site-key"),
    )
    no_key_worker = HeroesWMWorker("login", "password", "", captcha_mode="auto")
    monkeypatch.setattr(
        no_key_worker,
        "_solve_with_fallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("paid service called")),
    )
    assert no_key_worker.solve_challenge(page) is None

    api_worker = HeroesWMWorker("login", "password", "api-key", captcha_mode="auto")
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_service(method: str, **kwargs: str) -> dict[str, str]:
        calls.append((method, kwargs))
        return {"code": "recaptcha-response-token"}

    monkeypatch.setattr(api_worker, "_solve_with_fallback", fake_service)
    assert api_worker.solve_challenge(page) == "recaptcha-response-token"
    assert calls == [
        (
            "recaptcha",
            {
                "sitekey": "test-site-key",
                "url": "https://www.heroeswm.ru/object-info.php?id=378",
            },
        )
    ]


def test_unique_captcha_samples_are_saved_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(core, "get_captcha_samples_dir", lambda: tmp_path)
    worker = make_worker()
    image = b"\xff\xd8unique captcha jpeg"

    first = worker._save_captcha_sample(image)
    second = worker._save_captcha_sample(image)

    assert first == second
    assert first is not None and first.suffix == ".jpg"
    assert first.read_bytes() == image
    labels = (tmp_path / "labels.csv").read_text(encoding="utf-8-sig").splitlines()
    assert labels == ["file,correct_code", f"{first.name},"]


def test_server_confirmed_text_code_labels_saved_sample(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(core, "get_captcha_samples_dir", lambda: tmp_path)
    worker = make_worker()
    worker.last_captcha_image = b"\xff\xd8confirmed captcha jpeg"
    worker._save_captcha_sample(worker.last_captcha_image)
    worker._label_last_captcha("ab-12cd")
    labels = (tmp_path / "labels.csv").read_text(encoding="utf-8-sig").splitlines()
    assert labels[0] == "file,correct_code"
    assert labels[1].endswith(",AB12CD")


def test_success_timer_is_personal_hour_and_is_stable_after_restart(
    tmp_path: Path, monkeypatch,
) -> None:
    stats = StatsTracker(tmp_path / "stats.json")
    monkeypatch.setattr(core.random, "randint", lambda _low, _high: 180)
    first = HeroesWMWorker("login", "password", "", stats=stats)
    before = datetime.now()
    first._schedule_after_success()
    assert timedelta(minutes=62, seconds=59) <= first.next_attempt_time - before <= timedelta(minutes=63, seconds=1)

    second = HeroesWMWorker("login", "password", "", stats=StatsTracker(stats.path))
    second._schedule_existing_job(None)
    assert second.shift_end_time == first.shift_end_time.replace(microsecond=0)
    assert second.next_attempt_time == first.next_attempt_time.replace(microsecond=0)


def test_home_info_reads_current_workplace_and_personal_start_time() -> None:
    soup = BeautifulSoup(
        """
        <div id="hwm_topline_time">21:35</div>
        <div class="home_container_block home_work_block">
          <div>Гильдия Рабочих</div>
          <span>Место работы:
            <a href="object-info.php?id=315">Плавильный цех</a> с 21:01
          </span>
        </div>
        """,
        "html.parser",
    )
    status = parse_home_work_status(soup, datetime(2026, 8, 10, 21, 35, 42))
    assert status.state == "working"
    assert status.workplace_name == "Плавильный цех"
    assert status.workplace_id == "315"
    assert status.started_at == datetime(2026, 8, 10, 21, 1)
    assert status.next_allowed_at == datetime(2026, 8, 10, 22, 1)


def test_home_info_reads_server_remaining_minutes() -> None:
    soup = BeautifulSoup(
        """
        <div id="hwm_topline_time">21:35</div>
        <div class="home_work_block">
          Последнее место работы: <a href="object-info.php?id=44">Шахта самоцветов</a>.
          Вы можете устроиться на работу через 32 мин.
        </div>
        """,
        "html.parser",
    )
    status = parse_home_work_status(soup, datetime(2026, 8, 10, 21, 35, 42))
    assert status.state == "cooldown"
    assert status.remaining_minutes == 32
    assert status.next_allowed_at == datetime(2026, 8, 10, 22, 7)


def test_home_info_marks_battle_or_travel_as_busy() -> None:
    soup = BeautifulSoup("<div>Вы находитесь в бою</div>", "html.parser")
    assert parse_home_work_status(soup).state == "busy"
