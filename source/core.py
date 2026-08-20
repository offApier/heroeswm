from __future__ import annotations

import base64
import ast
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote_from_bytes, urljoin

import requests
from bs4 import BeautifulSoup, Tag
from twocaptcha import TwoCaptcha

from captcha_calibration import (
    TRAINING_SAMPLE_COUNT,
    calibrated_candidate,
    calibrated_candidates,
)
from storage import StatsTracker


BASE_URL = "https://www.heroeswm.ru"
CATEGORY_LABELS = {"mn": "Добыча", "fc": "Обработка", "sh": "Производство"}
CAPTCHA_SERVERS = ("rucaptcha.com", "2captcha.com")
TEXT_CAPTCHA_RETRY_SECONDS = 20
LOCAL_TEXT_CAPTCHA_MAX_ATTEMPTS = 6
SERVICE_TEXT_CAPTCHA_MAX_ATTEMPTS = 3
TEXT_CAPTCHA_SERVICE_RETRY_MINUTES = 2
TEXT_CAPTCHA_EXHAUSTED_RETRY_MINUTES = 15
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)


def get_captcha_samples_dir() -> Path:
    """Return a user-visible folder beside the executable/source files."""
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    return base / "captcha_samples"


class StopWorker(Exception):
    pass


class SessionExpired(Exception):
    pass


@dataclass(slots=True)
class Enterprise:
    object_id: str
    name: str
    salary: int
    category: str


@dataclass(slots=True)
class CaptchaChallenge:
    kind: str = "none"
    field_name: str | None = None
    image_url: str | None = None
    sitekey: str | None = None


@dataclass(slots=True)
class JobPage:
    object_id: str
    page_url: str
    status: str
    shift_end: datetime | None = None
    slots: int | None = None
    form_action: str | None = None
    form_fields: dict[str, str] = field(default_factory=dict)
    challenge: CaptchaChallenge = field(default_factory=CaptchaChallenge)
    loaded_at_monotonic: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class JobResult:
    status: str
    shift_end: datetime | None = None
    message: str = ""


@dataclass(slots=True)
class WorkGuildStatus:
    state: str
    workplace_name: str | None = None
    workplace_id: str | None = None
    started_at: datetime | None = None
    next_allowed_at: datetime | None = None
    remaining_minutes: int | None = None


def decode_page(content: bytes) -> str:
    return content.decode("windows-1251", errors="replace")


def soup_from_response(response: requests.Response) -> BeautifulSoup:
    return BeautifulSoup(decode_page(response.content), "html.parser")


def parse_clock(text: str, now: datetime | None = None) -> datetime | None:
    match = re.search(r"Окончание\s+смены[:\s]*(\d{1,2}):(\d{2})", text, re.I)
    if not match:
        return None
    now = now or datetime.now()
    candidate = now.replace(
        hour=int(match.group(1)), minute=int(match.group(2)), second=0, microsecond=0
    )
    if candidate < now - timedelta(minutes=2):
        candidate += timedelta(days=1)
    return candidate


def is_character_busy_text(text: str) -> bool:
    low = text.lower()
    markers = (
        "вы находитесь в бою",
        "персонаж находится в бою",
        "персонаж занят",
        "сейчас идет бой",
        "нельзя устроиться во время боя",
        "вы в пути",
        "вы находитесь в пути",
        "вы перемещаетесь",
        "переход по карте",
        "перемещение по карте",
        "во время перемещения",
    )
    return any(marker in low for marker in markers)


def parse_home_work_status(
    soup: BeautifulSoup, now: datetime | None = None
) -> WorkGuildStatus:
    """Parse the personal Laborers' Guild block from ``home.php?info``."""
    now = now or datetime.now()
    server_clock = soup.select_one("#hwm_topline_time")
    page_now = now
    if server_clock:
        match = re.search(r"(\d{1,2}):(\d{2})", server_clock.get_text(" ", strip=True))
        if match:
            page_now = now.replace(
                hour=int(match.group(1)), minute=int(match.group(2)), second=0, microsecond=0
            )

    block = soup.select_one(".home_work_block")
    block_text = block.get_text(" ", strip=True) if block else ""
    workplace_link = block.find("a", href=re.compile(r"object-info\.php\?id=\d+")) if block else None
    workplace_name = workplace_link.get_text(" ", strip=True) if workplace_link else None
    workplace_id = None
    if workplace_link:
        match = re.search(r"[?&]id=(\d+)", str(workplace_link.get("href")))
        workplace_id = match.group(1) if match else None

    active = re.search(r"Место\s+работы:.*?\s+с\s+(\d{1,2}):(\d{2})", block_text, re.I)
    if active:
        started_at = page_now.replace(
            hour=int(active.group(1)), minute=int(active.group(2)), second=0, microsecond=0
        )
        if started_at > page_now + timedelta(minutes=2):
            started_at -= timedelta(days=1)
        return WorkGuildStatus(
            "working",
            workplace_name,
            workplace_id,
            started_at,
            started_at + timedelta(minutes=60),
        )

    cooldown = re.search(
        r"можете\s+устроиться\s+на\s+работу\s+через\s+(\d+)\s*мин",
        block_text,
        re.I,
    )
    if cooldown:
        minutes = int(cooldown.group(1))
        return WorkGuildStatus(
            "cooldown",
            workplace_name,
            workplace_id,
            next_allowed_at=page_now + timedelta(minutes=minutes),
            remaining_minutes=minutes,
        )

    if re.search(r"можете\s+устроиться\s+на\s+работу", block_text, re.I):
        return WorkGuildStatus("ready", workplace_name, workplace_id)

    page_text = soup.get_text(" ", strip=True)
    if is_character_busy_text(page_text):
        return WorkGuildStatus("busy")
    return WorkGuildStatus("unknown", workplace_name, workplace_id)


def collect_form_fields(form: Tag) -> dict[str, str]:
    fields: dict[str, str] = {}
    for control in form.find_all(["input", "textarea", "select"]):
        if control.has_attr("disabled"):
            continue
        name = control.get("name")
        if not name:
            continue
        if control.name == "textarea":
            fields[name] = control.get_text()
            continue
        if control.name == "select":
            option = control.find("option", selected=True) or control.find("option")
            fields[name] = option.get("value", option.get_text()) if option else ""
            continue
        input_type = (control.get("type") or "text").lower()
        if input_type in {"submit", "button", "reset", "file", "image"}:
            continue
        if input_type in {"checkbox", "radio"} and not control.has_attr("checked"):
            continue
        fields[name] = str(control.get("value", ""))
    return fields


def _safe_integer_expression(expression: str) -> int | None:
    """Evaluate the small, obfuscated integer expressions used by HeroesWM."""
    if not re.fullmatch(r"[\d\s()+\-*/%]+", expression):
        return None
    try:
        root = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
        ):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            return left % right
        raise ValueError("unsupported expression")

    try:
        return visit(root)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def apply_scripted_form_values(form: Tag, fields: dict[str, str]) -> None:
    """Replay numeric hidden-field assignments made by the form's JavaScript."""
    script = "\n".join(str(tag.string or tag.get_text(" ")) for tag in form.find_all("script"))
    pattern = re.compile(
        r"getElementById\(\s*['\"](?P<id>[^'\"]+)['\"]\s*\)\.value\s*=\s*"
        r"(?P<expr>[\d\s()+\-*/%]+)\s*;",
        re.I,
    )
    for match in pattern.finditer(script):
        value = _safe_integer_expression(match.group("expr"))
        element = form.find(id=match.group("id"))
        name = element.get("name") if element else None
        if value is not None and name:
            fields[str(name)] = str(value)


def normalize_captcha_code(value: str) -> str:
    # HeroesWM's own change_tr() converts accidentally typed Russian/Ukrainian
    # keyboard characters to their QWERTY positions before submission.
    ru = "йцукенгшщзфывапролдячсмитьЙЦУКЕНГШЩЗФЫВАПРОЛДЯЧСМИТЬ"
    ua = "йцукенгшщзфівапролдячсмитьЙЦУКЕНГШЩЗФІВАПРОЛДЯЧСМИТЬ"
    en = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
    translation = {ord(source): target for source, target in zip(ru, en)}
    translation.update({ord(source): target for source, target in zip(ua, en)})
    translated = str(value or "").translate(translation)
    return "".join(character for character in translated if character.isalnum()).upper()


def form_urlencode_cp1251(fields: dict[str, Any]) -> str:
    parts = []
    for name, value in fields.items():
        encoded_name = quote_from_bytes(str(name).encode("windows-1251", errors="replace"), safe="")
        encoded_value = quote_from_bytes(str(value).encode("windows-1251", errors="replace"), safe="")
        parts.append(encoded_name + "=" + encoded_value)
    return "&".join(parts)


class HeroesWMWorker:
    def __init__(
        self,
        login: str,
        password: str,
        api_key: str,
        strategy: str = "salary",
        categories: Iterable[str] = ("mn", "fc", "sh"),
        captcha_mode: str = "auto",
        stop_event: threading.Event | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        manual_captcha_callback: Callable[[bytes], str | None] | None = None,
        stats: StatsTracker | None = None,
    ) -> None:
        self.login_name = login
        self.password = password
        self.api_key = api_key
        self.strategy = strategy if strategy in {"salary", "slots", "first"} else "salary"
        self.categories = [value for value in categories if value in CATEGORY_LABELS] or ["mn", "fc", "sh"]
        self.captcha_mode = captcha_mode if captcha_mode in {"auto", "local", "service", "manual"} else "auto"
        self.stop_event = stop_event or threading.Event()
        self.log_callback = log_callback
        self.manual_captcha_callback = manual_captcha_callback
        self.stats = stats or StatsTracker()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Language": "ru,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        self.player_id: str | None = None
        self.current_status = "Не запущен"
        self.next_attempt_time: datetime | None = None
        self.shift_end_time: datetime | None = None
        self._captcha_servers = list(CAPTCHA_SERVERS)
        self._local_ocr: Any = None
        self._local_candidate_cache: dict[str, list[str]] = {}
        self._local_exhausted_images: set[str] = set()
        self._local_attempt_counts: dict[str, int] = {}
        self._last_local_captcha_exhausted = False
        self._captcha_rejected_codes: dict[str, set[str]] = {}
        self._service_attempted_images: set[str] = set()
        self._service_failed_images: set[str] = set()
        self._service_attempt_counts: dict[str, int] = {}
        self._service_exhausted_images: set[str] = set()
        self._last_captcha_image_key: str | None = None
        self._last_text_captcha_source = ""
        self._last_service_solution: tuple[str, str, str] | None = None
        self._last_service_solver: Any = None
        # Kept in memory for diagnostics.  This is especially useful when the
        # game changes a form or redirects a successful submission to a page
        # that still contains generic captcha help text.
        self.last_page_html = ""
        self.last_response_html = ""
        self.last_response_url = ""
        self.last_captcha_image = b""

    def emit(self, level: str, message: str) -> None:
        self.current_status = message
        if self.log_callback:
            self.log_callback(level, message)

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise StopWorker()

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0, seconds)
        while time.monotonic() < deadline:
            self._check_stop()
            self.stop_event.wait(min(1.0, deadline - time.monotonic()))

    def _human_delay(self, low: float = 0.7, high: float = 1.8) -> None:
        self._sleep(random.uniform(low, high))

    def time_until_next_attempt(self) -> timedelta | None:
        if not self.next_attempt_time:
            return None
        remaining = self.next_attempt_time - datetime.now()
        return remaining if remaining.total_seconds() > 0 else None

    def _form_action(self, response: requests.Response, form: Tag) -> str:
        return urljoin(response.url, form.get("action") or response.url)

    def _login_form(self, soup: BeautifulSoup) -> Tag | None:
        return soup.find("form", action=re.compile(r"login\.php", re.I)) or soup.find("form")

    def _find_text_input(self, form: Tag, excluded: set[str]) -> Tag | None:
        for field in form.find_all("input"):
            field_type = (field.get("type") or "text").lower()
            name = (field.get("name") or "").lower()
            if field_type in {"text", "search"} and name and name not in excluded:
                return field
        return None

    def _image_url_for_form(self, form: Tag, page_url: str) -> str | None:
        candidates: list[tuple[int, str]] = []
        for image in form.find_all("img"):
            src = image.get("src") or ""
            low = " ".join(
                [src, image.get("id") or "", " ".join(image.get("class") or [])]
            ).lower()
            if any(value in low for value in ("btn_work", "arrow", "logo", "gold.png", "/i/r/")):
                continue
            score = 0
            if any(value in low for value in ("captcha", "capcha", "code")):
                score += 10
            try:
                if int(image.get("width", 0)) >= 100 and int(image.get("height", 0)) >= 30:
                    score += 3
            except (TypeError, ValueError):
                pass
            if src:
                candidates.append((score, urljoin(page_url, src)))
        for element in form.find_all(True):
            marker = " ".join(
                [element.get("id") or "", " ".join(element.get("class") or [])]
            ).lower()
            style = element.get("style") or ""
            if any(value in marker for value in ("captcha", "capcha", "code")):
                match = re.search(r"url\(['\"]?([^)'\"]+)", style, re.I)
                if match:
                    candidates.append((12, urljoin(page_url, match.group(1))))
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates and candidates[0][0] > 0 else None

    def login_once(self) -> bool:
        self.emit("info", "Авторизация...")
        try:
            response = self.session.get(BASE_URL + "/login.php", timeout=20)
            response.raise_for_status()
            soup = soup_from_response(response)
            form = self._login_form(soup)
            fields = collect_form_fields(form) if form else {}
            fields.update(
                {
                    "LOGIN_redirect": fields.get("LOGIN_redirect", "1"),
                    "login": self.login_name,
                    "lreseted": fields.get("lreseted", "1"),
                    "pass": self.password,
                    "preseted": fields.get("preseted", "1"),
                    "pliv": fields.get("pliv", "0"),
                    "x": "78",
                    "y": "34",
                }
            )
            action = self._form_action(response, form) if form else BASE_URL + "/login.php"
            response = self.session.post(
                action,
                data=form_urlencode_cp1251(fields),
                headers={
                    "Referer": BASE_URL + "/?auth_error",
                    "Origin": BASE_URL,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                allow_redirects=True,
                timeout=20,
            )
            if self._login_response_ok(response):
                return True

            soup = soup_from_response(response)
            form = self._login_form(soup)
            if not form:
                self.emit("error", "Форма входа не найдена после отказа в авторизации")
                return False
            captcha_input = self._find_text_input(form, {"login", "pass", "password"})
            image_url = self._image_url_for_form(form, response.url)
            if not captcha_input or not image_url:
                snippet = soup.get_text(" ", strip=True)[:180]
                self.emit("error", "Вход отклонён, капча входа не найдена: " + snippet)
                return False
            image = self.session.get(image_url, headers={"Referer": response.url}, timeout=20)
            image.raise_for_status()
            code = self._obtain_image_code(image.content)
            if not code:
                self.emit("error", "Код капчи входа не получен")
                return False
            fields = collect_form_fields(form)
            fields.update({"login": self.login_name, "pass": self.password})
            fields[captcha_input.get("name")] = code
            response = self.session.post(
                self._form_action(response, form),
                data=form_urlencode_cp1251(fields),
                headers={
                    "Referer": response.url,
                    "Origin": BASE_URL,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                allow_redirects=True,
                timeout=20,
            )
            return self._login_response_ok(response)
        except requests.RequestException as exc:
            self.emit("error", f"Ошибка сети при входе: {exc}")
            return False

    def _login_response_ok(self, response: requests.Response) -> bool:
        if "login.php" in response.url.lower() or "auth_error" in response.url.lower():
            return False
        player_id = self.session.cookies.get("pl_id")
        if player_id:
            self.player_id = player_id
        self.emit("info", "✓ Вход выполнен" + (f" (ID {player_id})" if player_id else ""))
        return True

    def is_logged_in(self) -> bool:
        # A fresh Session cannot be authenticated without the player cookie.
        # Avoid a pointless /home.php request which used to consume the full
        # 15-second timeout before every first login.
        if not self.session.cookies.get("pl_id"):
            return False
        try:
            response = self.session.get(BASE_URL + "/home.php", timeout=15)
            if "login.php" in response.url.lower() or "auth_error" in response.url.lower():
                return False
            return bool(self.session.cookies.get("pl_id"))
        except requests.RequestException:
            return False

    def get_work_guild_status(self) -> WorkGuildStatus:
        response = self.session.get(BASE_URL + "/home.php?info", timeout=20)
        response.raise_for_status()
        if "login.php" in response.url.lower() or "auth_error" in response.url.lower():
            return WorkGuildStatus("session_expired")
        return parse_home_work_status(soup_from_response(response))

    def ensure_logged_in(self) -> None:
        if self.is_logged_in():
            return
        self.session.cookies.clear()
        delay = 15
        attempt = 0
        while True:
            self._check_stop()
            attempt += 1
            self.emit("warning", f"Сессия недействительна. Попытка входа #{attempt}")
            if self.login_once():
                return
            self.stats.error("ошибка входа")
            self.emit("warning", f"Вход не удался; повтор через {delay} сек")
            self._sleep(delay)
            delay = min(300, max(30, int(delay * 1.6)))

    def get_enterprises(self) -> list[Enterprise]:
        self.emit("info", "Загружаю предприятия выбранных категорий...")
        base = self.session.get(BASE_URL + "/map.php", timeout=20)
        base.raise_for_status()
        if "login.php" in base.url.lower() or "auth_error" in base.url.lower():
            raise SessionExpired()
        base_soup = soup_from_response(base)
        category_urls: dict[str, str] = {}
        for link in base_soup.find_all("a", href=True):
            href = str(link.get("href"))
            match = re.search(r"[?&]st=(mn|fc|sh)(?:&|$)", href)
            if match:
                category_urls[match.group(1)] = urljoin(base.url, href.replace("&amp;", "&"))

        result: dict[tuple[str, str], Enterprise] = {}
        for category in self.categories:
            self._check_stop()
            url = category_urls.get(category, BASE_URL + "/map.php?st=" + category)
            response = base if category in category_urls and response_url_equal(base.url, url) else self.session.get(url, timeout=20)
            response.raise_for_status()
            if "login.php" in response.url.lower() or "auth_error" in response.url.lower():
                raise SessionExpired()
            soup = soup_from_response(response)
            for enterprise in self._parse_enterprises(soup, category):
                result[(category, enterprise.object_id)] = enterprise
            self._human_delay(0.25, 0.65)
        enterprises = list(result.values())
        if self.strategy == "salary":
            enterprises.sort(key=lambda item: item.salary, reverse=True)
        elif self.strategy == "first":
            pass
        else:
            # Slots are only known on the object page; salary is the best useful
            # pre-sort before those pages are checked.
            enterprises.sort(key=lambda item: item.salary, reverse=True)
        self.emit("info", f"Найдено предприятий: {len(enterprises)}")
        return enterprises

    @staticmethod
    def _parse_enterprises(soup: BeautifulSoup, category: str) -> list[Enterprise]:
        enterprises: list[Enterprise] = []
        seen: set[str] = set()
        rows = soup.select("tr.map_obj_table_hover") or soup.find_all("tr")
        for row in rows:
            link = row.find("a", href=re.compile(r"object-info\.php\?id=\d+"))
            if not link:
                continue
            match = re.search(r"object-info\.php\?id=(\d+)", str(link.get("href")))
            if not match or match.group(1) in seen:
                continue
            object_id = match.group(1)
            cells = row.find_all("td")
            name = cells[0].get_text(" ", strip=True) if cells else link.get_text(" ", strip=True)
            salary = 0
            if len(cells) >= 4:
                numbers = re.findall(r"\d[\d,\s]*", cells[3].get_text(" ", strip=True))
                if numbers:
                    salary = int(re.sub(r"\D", "", numbers[-1]) or 0)
            if not salary:
                numbers = [int(re.sub(r"\D", "", item)) for item in re.findall(r"\d[\d,\s]*", row.get_text(" ", strip=True)) if re.sub(r"\D", "", item)]
                salary = numbers[-1] if numbers else 0
            seen.add(object_id)
            enterprises.append(Enterprise(object_id, name, salary, category))
        return enterprises

    def get_job_page(self, enterprise: Enterprise) -> JobPage:
        url = BASE_URL + "/object-info.php?id=" + enterprise.object_id
        response = self.session.get(url, headers={"Referer": BASE_URL + "/map.php"}, timeout=20)
        response.raise_for_status()
        self.last_page_html = decode_page(response.content)
        if "login.php" in response.url.lower() or "auth_error" in response.url.lower():
            return JobPage(enterprise.object_id, response.url, "session_expired")
        soup = soup_from_response(response)
        text = soup.get_text(" ", strip=True)
        shift_end = parse_clock(text)
        if is_character_busy_text(text):
            return JobPage(enterprise.object_id, response.url, "busy")
        if (
            "Вы уже работаете" in text
            or "Вы уже устроены" in text
            or "already working" in text.lower()
        ):
            return JobPage(enterprise.object_id, response.url, "already_working", shift_end=shift_end)
        if "Прошло меньше часа" in text or "часа с последнего устройства" in text:
            return JobPage(enterprise.object_id, response.url, "cooldown", shift_end=shift_end)
        slots_match = re.search(r"Свободных\s+мест[:\s]*(\d+)", text, re.I)
        slots = int(slots_match.group(1)) if slots_match else None
        if slots == 0:
            return JobPage(enterprise.object_id, response.url, "no_slots", slots=0)
        form = soup.find("form", id="getjob_form") or soup.find(
            "form", action=re.compile(r"object_do\.php", re.I)
        )
        if not form:
            return JobPage(enterprise.object_id, response.url, "no_form", slots=slots)
        challenge = self._detect_challenge(form, response.url)
        form_fields = collect_form_fields(form)
        apply_scripted_form_values(form, form_fields)
        return JobPage(
            enterprise.object_id,
            response.url,
            "ready",
            slots=slots,
            form_action=urljoin(response.url, form.get("action") or "/object_do.php"),
            form_fields=form_fields,
            challenge=challenge,
        )

    def _detect_challenge(self, form: Tag, page_url: str) -> CaptchaChallenge:
        turnstile = form.select_one(".cf-turnstile[data-sitekey]")
        if turnstile:
            return CaptchaChallenge("turnstile", sitekey=turnstile.get("data-sitekey"))
        recaptcha = form.select_one(".g-recaptcha[data-sitekey]")
        if recaptcha:
            return CaptchaChallenge("recaptcha", sitekey=recaptcha.get("data-sitekey"))
        code_input = form.find("input", id="code") or form.find("input", attrs={"name": "code"})
        if not code_input:
            code_input = self._find_text_input(form, {"login", "pass", "password", "buy_count"})
        if code_input:
            return CaptchaChallenge(
                "image",
                field_name=code_input.get("name") or code_input.get("id") or "code",
                image_url=self._image_url_for_form(form, page_url),
            )
        return CaptchaChallenge("none")

    def _solve_with_fallback(self, method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        last_error: Exception | None = None
        for index, server in enumerate(list(self._captcha_servers)):
            try:
                solver = TwoCaptcha(self.api_key, server=server)
                result = getattr(solver, method)(*args, **kwargs)
                self._last_service_solver = solver
                if index:
                    self._captcha_servers.remove(server)
                    self._captcha_servers.insert(0, server)
                    self.emit("info", f"Сервис капчи переключён на {server}")
                return result
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if not any(word in message for word in ("connect", "timeout", "network", "ssl", "dns", "503", "502")):
                    raise
                self.emit("warning", f"{server} недоступен, пробую следующий")
        assert last_error is not None
        raise last_error

    @staticmethod
    def _rank_ocr_candidates(
        result: dict[str, Any], limit: int = 5, expected_length: int = 6
    ) -> list[str]:
        """Return likely CTC decodings, merging upper/lower-case variants."""
        charset = result.get("charset")
        rows = result.get("probabilities")
        if not isinstance(charset, list) or not isinstance(rows, list):
            return []
        valid = [
            index
            for index, character in enumerate(charset)
            if character == ""
            or (
                isinstance(character, str)
                and len(character) == 1
                and character.isascii()
                and character.isalnum()
            )
        ]
        beams: dict[tuple[str, str], float] = {("", ""): 0.0}
        for raw_row in rows:
            row = raw_row[0] if isinstance(raw_row, list) and len(raw_row) == 1 else raw_row
            if not isinstance(row, list) or len(row) < len(charset):
                return []
            choices = sorted(valid, key=lambda index: row[index], reverse=True)[:12]
            next_beams: dict[tuple[str, str], float] = {}
            for (output, previous), score in beams.items():
                for index in choices:
                    character = charset[index]
                    decoded = output if character == "" or character == previous else output + character
                    if len(decoded) > 7:
                        continue
                    candidate_score = score + math.log(max(float(row[index]), 1e-30))
                    key = (decoded, character)
                    if candidate_score > next_beams.get(key, -1e300):
                        next_beams[key] = candidate_score
            beams = dict(
                sorted(next_beams.items(), key=lambda item: item[1], reverse=True)[:1200]
            )

        ranked: dict[str, float] = {}
        for (raw, _previous), score in beams.items():
            candidate = normalize_captcha_code(raw)
            if len(candidate) == expected_length:
                ranked[candidate] = max(score, ranked.get(candidate, -1e300))
        return [
            candidate
            for candidate, _score in sorted(ranked.items(), key=lambda item: item[1], reverse=True)[:limit]
        ]

    @staticmethod
    def _font_confusion_candidate(code: str) -> str:
        """Build a possible game-font correction; callers must verify it."""
        corrected: list[str] = []
        last_index = len(code) - 1
        for index, character in enumerate(code):
            if character == "Z":
                character = "2"
            elif character in {"0", "O", "U"}:
                character = "Q"
            elif character == "D" and index not in {0, last_index}:
                character = "Q"
            elif index == last_index and character in {"3", "5"}:
                character = "S"
            corrected.append(character)
        return "".join(corrected)

    def _ocr_variant_results(self, image_bytes: bytes) -> list[dict[str, Any]]:
        """Run independent views that preserve outlines and suppress thin noise."""
        results: list[dict[str, Any]] = []
        try:
            from PIL import Image

            source = Image.open(io.BytesIO(image_bytes)).convert("L")
            enlarged = source.resize(
                (source.width * 2, source.height * 2), Image.Resampling.NEAREST
            )
            output = io.BytesIO()
            enlarged.save(output, format="PNG")
            result = self._local_ocr.classification(output.getvalue(), probability=True)
            if isinstance(result, dict):
                results.append(result)
        except Exception:
            pass

        try:
            import cv2
            import numpy as np

            grayscale = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if grayscale is not None:
                _level, cleaned = cv2.threshold(
                    grayscale, 230, 255, cv2.THRESH_BINARY
                )
                cleaned = cv2.morphologyEx(
                    cleaned, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)
                )
                encoded, cleaned_bytes = cv2.imencode(".png", cleaned)
                if encoded:
                    result = self._local_ocr.classification(
                        cleaned_bytes.tobytes(), probability=True
                    )
                    if isinstance(result, dict):
                        results.append(result)
        except Exception:
            pass
        return results

    def _save_captcha_sample(self, image_bytes: bytes) -> Path | None:
        """Save each unique text captcha once for later human labelling."""
        try:
            image_key = hashlib.sha256(image_bytes).hexdigest()
            sample_dir = get_captcha_samples_dir()
            sample_dir.mkdir(parents=True, exist_ok=True)
            extension = ".jpg" if image_bytes.startswith(b"\xff\xd8") else ".png"
            image_path = sample_dir / f"sample_{image_key[:12]}{extension}"
            if image_path.exists():
                return image_path
            image_path.write_bytes(image_bytes)
            labels_path = sample_dir / "labels.csv"
            new_file = not labels_path.exists()
            with labels_path.open("a", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                if new_file:
                    writer.writerow(("file", "correct_code"))
                writer.writerow((image_path.name, ""))
            self.emit("info", f"Новый образец капчи сохранён: {image_path}")
            return image_path
        except OSError as exc:
            self.emit("warning", f"Не удалось сохранить образец капчи: {exc}")
            return None

    def _label_last_captcha(self, code: str | None) -> None:
        """Store a server-confirmed answer beside the saved unique image."""
        normalized = normalize_captcha_code(code or "")
        if len(normalized) != 6 or not self.last_captcha_image:
            return
        try:
            image_path = self._save_captcha_sample(self.last_captcha_image)
            if image_path is None:
                return
            labels_path = image_path.parent / "labels.csv"
            rows: list[dict[str, str]] = []
            if labels_path.exists():
                with labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
            found = False
            for row in rows:
                if row.get("file") == image_path.name:
                    row["correct_code"] = normalized
                    found = True
            if not found:
                rows.append({"file": image_path.name, "correct_code": normalized})
            temporary = labels_path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("file", "correct_code"))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(labels_path)
            self.emit("info", f"Подтверждённый ответ сохранён для обучения: {normalized}")
        except (OSError, csv.Error) as exc:
            self.emit("warning", f"Не удалось подписать образец капчи: {exc}")

    def _local_solve(self, image_bytes: bytes) -> str | None:
        try:
            self._last_local_captcha_exhausted = False
            if self._local_ocr is None:
                self.emit("info", "Загружаю локальную модель ddddocr...")
                import ddddocr

                self._local_ocr = ddddocr.DdddOcr(show_ad=False)
                self.emit(
                    "info",
                    f"OCR откалиброван по {TRAINING_SAMPLE_COUNT} реальным капчам HeroesWM",
                )
            image_key = hashlib.sha256(image_bytes).hexdigest()
            attempts = self._local_attempt_counts.get(image_key, 0)
            if attempts >= LOCAL_TEXT_CAPTCHA_MAX_ATTEMPTS:
                self._local_exhausted_images.add(image_key)
                self._last_local_captcha_exhausted = True
                return None
            rejected = self._captcha_rejected_codes.get(image_key, set())
            candidates = self._local_candidate_cache.get(image_key)
            # An empty cache is rebuilt for the same image on the next short
            # retry. Already rejected codes remain excluded, so the worker
            # keeps analysing the captcha without resending known bad answers.
            if not candidates:
                raw_result = self._local_ocr.classification(image_bytes, probability=True)
                if isinstance(raw_result, dict):
                    primary = normalize_captcha_code(str(raw_result.get("text", "")))
                    result_views = [raw_result, *self._ocr_variant_results(image_bytes)]
                    ranked_views = [
                        self._rank_ocr_candidates(result, limit=40)
                        for result in result_views
                    ]
                    model_candidates = ranked_views[0]
                    enlarged_candidates = (
                        ranked_views[1] if len(ranked_views) > 1 else model_candidates
                    )
                    cleaned_candidates = ranked_views[2] if len(ranked_views) > 2 else []
                    bases: list[str] = []
                    # Enlarging with nearest-neighbour preserves the thin Q/A tails
                    # that the model loses at the captcha's native 250x60 size.
                    bases.extend(enlarged_candidates[:20])
                    bases.extend(model_candidates[:20])
                    bases.extend(cleaned_candidates[:20])
                    if len(primary) == 6:
                        bases.append(primary)
                    candidates = []
                    for base in bases:
                        corrected = self._font_confusion_candidate(base)
                        # A correction is promoted only when the complete corrected
                        # six-character code also occurs in at least two independent
                        # OCR views.  This prevents blind substitutions such as 9->A.
                        support = sum(
                            corrected in ranked for ranked in ranked_views
                        )
                        original_order = (
                            (corrected, base)
                            if corrected != base and support >= 2
                            else (base, corrected)
                        )
                        ordered = [original_order[0]]
                        # The seven-image calibration is advisory: the generic
                        # model keeps its first answer and the learned game-font
                        # correction is tried immediately after it.
                        for trained in calibrated_candidates(base):
                            if trained not in ordered:
                                ordered.append(trained)
                        if original_order[1] not in ordered:
                            ordered.append(original_order[1])
                        for candidate in ordered:
                            if len(candidate) == 6 and candidate not in candidates:
                                candidates.append(candidate)
                    candidates = candidates[:80]
                else:
                    candidates = [normalize_captcha_code(str(raw_result))]
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate and candidate not in rejected
                ]
                self._local_candidate_cache[image_key] = candidates
                if len(self._local_candidate_cache) > 12:
                    del self._local_candidate_cache[next(iter(self._local_candidate_cache))]
            else:
                candidates[:] = [candidate for candidate in candidates if candidate not in rejected]
            if not candidates:
                self._local_exhausted_images.add(image_key)
                self._last_local_captcha_exhausted = True
                self._local_candidate_cache.pop(image_key, None)
                if len(self._local_exhausted_images) > 32:
                    self._local_exhausted_images.pop()
                return None
            self._local_exhausted_images.discard(image_key)
            code = candidates.pop(0)
            available = min(
                len(candidates),
                max(0, LOCAL_TEXT_CAPTCHA_MAX_ATTEMPTS - attempts - 1),
            )
            if available:
                self.emit(
                    "info",
                    f"Локальный OCR подготовил ещё {available} приоритетных вариант(а) "
                    f"(общий лимит {LOCAL_TEXT_CAPTCHA_MAX_ATTEMPTS})",
                )
            return code
        except Exception as exc:
            self.emit("error", f"Локальное распознавание не удалось: {exc}")
            return None

    def _mark_last_local_captcha_exhausted(self) -> None:
        if not self.last_captcha_image:
            return
        image_key = hashlib.sha256(self.last_captcha_image).hexdigest()
        self._local_candidate_cache.pop(image_key, None)
        self._local_exhausted_images.add(image_key)
        self._last_local_captcha_exhausted = True
        if len(self._local_exhausted_images) > 32:
            self._local_exhausted_images.pop()

    def _remaining_local_candidates(self) -> int:
        if not self.last_captcha_image:
            return 0
        image_key = hashlib.sha256(self.last_captcha_image).hexdigest()
        attempts = self._local_attempt_counts.get(image_key, 0)
        budget = max(0, LOCAL_TEXT_CAPTCHA_MAX_ATTEMPTS - attempts)
        return min(len(self._local_candidate_cache.get(image_key) or []), budget)

    def _report_last_service_solution(self, correct: bool) -> None:
        solution = self._last_service_solution
        solver = self._last_service_solver
        self._last_service_solution = None
        if not solution or solver is None:
            return
        try:
            solver.report(solution[2], correct)
        except Exception as exc:
            self.emit("warning", f"Не удалось отправить оценку решения сервису: {exc}")

    def _remember_captcha_rejection(self, code: str | None) -> None:
        if not self.last_captcha_image:
            return
        image_key = hashlib.sha256(self.last_captcha_image).hexdigest()
        normalized = normalize_captcha_code(code or "")
        if normalized:
            self._captcha_rejected_codes.setdefault(image_key, set()).add(normalized)
            cached = self._local_candidate_cache.get(image_key)
            if cached is not None:
                cached[:] = [candidate for candidate in cached if candidate != normalized]
        solution = self._last_service_solution
        if solution and solution[0] == image_key and solution[1] == normalized:
            self._service_failed_images.add(image_key)
            self._report_last_service_solution(False)

    def _service_solve_image(self, image_bytes: bytes) -> str | None:
        if not self.api_key:
            self.emit("error", "Для сервиса капчи не указан API-ключ")
            return None
        image_key = hashlib.sha256(image_bytes).hexdigest()
        attempts = self._service_attempt_counts.get(image_key, 0)
        if attempts >= SERVICE_TEXT_CAPTCHA_MAX_ATTEMPTS:
            self._last_text_captcha_source = "service_exhausted"
            if image_key not in self._service_exhausted_images:
                self.emit(
                    "error",
                    f"Сервис уже дал {attempts} отклонённых/неудачных решения этой капчи; "
                    f"следующая проверка через {TEXT_CAPTCHA_EXHAUSTED_RETRY_MINUTES} мин",
                )
                self._service_exhausted_images.add(image_key)
            return None
        rejected = self._captcha_rejected_codes.get(image_key, set())
        self._service_attempted_images.add(image_key)
        attempts += 1
        self._service_attempt_counts[image_key] = attempts
        self.emit(
            "info",
            f"Передаю текстовую капчу в RuCaptcha/2Captcha "
            f"(попытка сервиса {attempts}/{SERVICE_TEXT_CAPTCHA_MAX_ATTEMPTS})",
        )
        try:
            result = self._solve_with_fallback(
                "normal",
                base64.b64encode(image_bytes).decode("ascii"),
                minLen=6,
                maxLen=6,
                phrase=0,
                caseSensitive=0,
            )
            code = normalize_captcha_code(result["code"]) or None
            captcha_id = str(result.get("captchaId") or "")
            if not code:
                self._last_text_captcha_source = "service_error"
                return None
            self._last_text_captcha_source = "service"
            self._last_service_solution = (image_key, code, captcha_id)
            if code in rejected:
                self.emit(
                    "warning",
                    f"Сервис снова вернул уже отклонённый код {code}; POST не отправляю",
                )
                self._service_failed_images.add(image_key)
                self._report_last_service_solution(False)
                self._last_text_captcha_source = "service_error"
                return None
            return code
        except Exception as exc:
            self._last_text_captcha_source = "service_error"
            self.emit("error", f"Сервис не решил текстовую капчу: {exc}")
            return None

    def _obtain_image_code(self, image_bytes: bytes) -> str | None:
        image_key = hashlib.sha256(image_bytes).hexdigest()
        self._last_captcha_image_key = image_key
        self._last_service_solution = None
        self._last_text_captcha_source = ""
        if self.captcha_mode == "manual":
            if not self.manual_captcha_callback:
                return None
            return normalize_captcha_code(self.manual_captcha_callback(image_bytes) or "") or None

        if self.captcha_mode == "service":
            self._last_text_captcha_source = "service"
            return self._service_solve_image(image_bytes)

        attempts = self._local_attempt_counts.get(image_key, 0)
        if attempts < LOCAL_TEXT_CAPTCHA_MAX_ATTEMPTS:
            code = self._local_solve(image_bytes)
            if code:
                attempts += 1
                self._local_attempt_counts[image_key] = attempts
                self._last_text_captcha_source = "local"
                self.emit(
                    "info",
                    f"Локальная попытка {attempts}/{LOCAL_TEXT_CAPTCHA_MAX_ATTEMPTS}",
                )
                return code

        self._last_local_captcha_exhausted = True
        self._local_exhausted_images.add(image_key)
        # Old settings may still contain the legacy "local" value.  Treat it
        # like auto mode when a key exists, otherwise an upgraded installation
        # could silently keep looping locally and never reach the paid fallback.
        if not self.api_key:
            self._last_text_captcha_source = "local_no_api"
            return None
        if image_key not in self._service_attempted_images:
            self.emit(
                "warning",
                "Локальные варианты закончились или достигнут лимит; переключаюсь на RuCaptcha/2Captcha",
            )
        self._last_text_captcha_source = "service"
        return self._service_solve_image(image_bytes)

    def _schedule_unsolved_text_captcha(self) -> None:
        source = self._last_text_captcha_source
        if source == "service_error":
            self.emit(
                "warning",
                f"Платный сервис пока не дал нового ответа; повтор через "
                f"{TEXT_CAPTCHA_SERVICE_RETRY_MINUTES} мин",
            )
            self.next_attempt_time = datetime.now() + timedelta(
                minutes=TEXT_CAPTCHA_SERVICE_RETRY_MINUTES
            )
            return
        if source == "service_exhausted":
            self.next_attempt_time = datetime.now() + timedelta(
                minutes=TEXT_CAPTCHA_EXHAUSTED_RETRY_MINUTES
            )
            return
        if source == "local_no_api":
            self.emit(
                "warning",
                "Локальные варианты закончились, а API-ключ не указан; "
                f"повторная проверка через {TEXT_CAPTCHA_EXHAUSTED_RETRY_MINUTES} мин",
            )
            self.next_attempt_time = datetime.now() + timedelta(
                minutes=TEXT_CAPTCHA_EXHAUSTED_RETRY_MINUTES
            )
            return
        self.emit(
            "warning",
            f"Капчу не удалось распознать; повтор через {TEXT_CAPTCHA_RETRY_SECONDS} сек",
        )
        self.next_attempt_time = datetime.now() + timedelta(seconds=TEXT_CAPTCHA_RETRY_SECONDS)

    def solve_challenge(self, page: JobPage) -> str | None:
        challenge = page.challenge
        if challenge.kind == "none":
            return ""
        if challenge.kind == "image":
            if not challenge.image_url:
                self.emit("error", "Поле текстовой капчи найдено, но изображение не найдено")
                return None
            try:
                if challenge.image_url.startswith("data:"):
                    image = base64.b64decode(challenge.image_url.split(",", 1)[1])
                else:
                    response = self.session.get(
                        challenge.image_url, headers={"Referer": page.page_url}, timeout=20
                    )
                    response.raise_for_status()
                    image = response.content
                self.last_captcha_image = image
                self._save_captcha_sample(image)
                code = self._obtain_image_code(image)
                if code:
                    self.emit("info", f"✓ Текстовая капча распознана: {code}")
                return code
            except requests.RequestException as exc:
                self.emit("error", f"Не удалось скачать капчу: {exc}")
                return None
        if not self.api_key:
            self.emit(
                "warning",
                f"Обнаружена {challenge.kind}, но API-ключ не указан; платный сервис не вызываю",
            )
            return None
        try:
            method = "turnstile" if challenge.kind == "turnstile" else "recaptcha"
            self.emit("info", f"Обнаружена {challenge.kind}; передаю в RuCaptcha/2Captcha")
            result = self._solve_with_fallback(
                method, sitekey=challenge.sitekey, url=page.page_url
            )
            self.emit("info", f"✓ {challenge.kind} решена")
            return result["code"]
        except Exception as exc:
            self.emit("error", f"Ошибка решения {challenge.kind}: {exc}")
            return None

    def _fingerprint(self) -> dict[str, Any]:
        return {
            "clickX": 24,
            "clickY": 24,
            "globalX": 795,
            "globalY": 512,
            "screenX": 795,
            "screenY": 512,
            "videoCard": "ANGLE (Intel, Intel(R) UHD Graphics Direct3D11) xx Google Inc. (Intel)",
            "deviceMemory": 8,
            "hardwareConcurrency": 8,
            "screenResolution": "1920x1080",
            "screenWidth": 1920,
            "screenHeight": 1080,
            "pixelRatio": 1,
            "navAppName": "Netscape",
            "navProduct": "Gecko",
            "navAppVersion": USER_AGENT.replace("Mozilla/", ""),
            "navUserAgent": USER_AGENT,
            "navPlatform": "Win32",
            "browserLang": "ru-RU",
            "browserLangs": "ru-RU, ru, en",
            "touchPoints": 0,
            "innerWidth": 1280,
            "innerHeight": 720,
            "outerWidth": 1280,
            "outerHeight": 800,
            "clientWidth": 1263,
            "clientHeight": 720,
        }

    def _typing_telemetry(self, code: str, page_age_ms: int | None = None) -> str:
        data: dict[str, Any] = {
            "screen_width": 1920,
            "screen_height": 1080,
            "pixel_ratio": 1,
            "nav_appName": "Netscape",
            "nav_product": "Gecko",
            "nav_appVersion": USER_AGENT.replace("Mozilla/", ""),
            "nav_userAgent": USER_AGENT,
            "nav_platform": "Win32",
            "browserLang": "ru-RU",
            "innerWidth": 1280,
            "innerHeight": 720,
            "outerWidth": 1280,
            "outerHeight": 800,
        }
        clock_offset = random.randint(75, 180)
        elapsed = random.randint(180, 520)
        events: list[dict[str, Any]] = []

        def mouse_event(
            event_type: str,
            code_value: str,
            x: int,
            y: int,
            movement_x: int = 0,
            movement_y: int = 0,
        ) -> dict[str, Any]:
            return {
                "type": event_type,
                "time2": elapsed,
                "cX": x,
                "cY": y,
                "mX": movement_x,
                "mY": movement_y,
                "code": code_value,
                "time": elapsed + clock_offset,
            }

        # The input has its own mousedown handler in addition to the body
        # handler, so a real browser records the same event twice.
        events.append(mouse_event("mousemove", "", 620, 440, 4, 2))
        elapsed += random.randint(30, 90)
        down = mouse_event("mousedown", "", 650, 450)
        events.extend((down, dict(down)))
        elapsed += random.randint(25, 70)
        events.append(mouse_event("mouseup", "", 650, 450))
        elapsed += random.randint(180, 420)
        current = ""
        for character in code:
            key_code = ord(character.upper())
            events.append({"type": "keydown", "time2": elapsed, "keyCode": key_code, "code": current, "time": elapsed + clock_offset})
            elapsed += random.randint(25, 70)
            current += character
            events.append({"type": "input", "time2": elapsed, "code": current, "time": elapsed + clock_offset})
            elapsed += random.randint(12, 38)
            events.append({"type": "keyup", "time2": elapsed, "code": current, "time": elapsed + clock_offset})
            elapsed += random.randint(65, 165)

        # Account for the actual OCR/service delay while keeping the event
        # order natural, then reproduce the move/click on the pickaxe button.
        target_elapsed = max(elapsed + 250, page_age_ms or 0)
        elapsed = target_elapsed + random.randint(80, 240)
        events.append(mouse_event("mousemove", current, 795, 512, 145, 62))
        elapsed += random.randint(35, 95)
        events.append(mouse_event("mousedown", current, 795, 512))
        elapsed += random.randint(25, 65)
        events.append(mouse_event("mouseup", current, 795, 512))
        for index, event in enumerate(events):
            data[str(index)] = event
        data["cur_time"] = elapsed + random.randint(8, 45)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def build_job_payload(self, page: JobPage, captcha_token: str | None) -> dict[str, str]:
        payload = dict(page.form_fields)
        payload["other_data"] = json.dumps(
            self._fingerprint(), ensure_ascii=False, separators=(",", ":")
        )
        challenge = page.challenge
        if challenge.kind == "image":
            code = normalize_captcha_code(captcha_token or "")
            payload[challenge.field_name or "code"] = code
            page_age_ms = int(max(0.0, time.monotonic() - page.loaded_at_monotonic) * 1000)
            payload["work_code_data_element"] = self._typing_telemetry(code, page_age_ms)
        elif captcha_token:
            response_field = (
                "cf-turnstile-response" if challenge.kind == "turnstile" else "g-recaptcha-response"
            )
            payload[response_field] = captcha_token
        return payload

    def take_job(self, page: JobPage, captcha_token: str | None) -> JobResult:
        if not page.form_action:
            return JobResult("error", message="Нет action формы устройства")
        try:
            self._human_delay(0.7, 1.5)
            payload = self.build_job_payload(page, captcha_token)
            response = self.session.post(
                page.form_action,
                data=payload,
                headers={"Referer": page.page_url, "Origin": BASE_URL},
                allow_redirects=True,
                timeout=20,
            )
            self.last_response_url = response.url
            self.last_response_html = decode_page(response.content)
            response_soup = soup_from_response(response)
            text = response_soup.get_text(" ", strip=True)
            low_url = response.url.lower()
            shift_end = parse_clock(text)
            if "login.php" in low_url or "auth_error" in low_url:
                return JobResult("session_expired", message="Сессия завершилась во время POST")
            if "got_job" in low_url or "устроились на работу" in text.lower():
                return JobResult("success", shift_end, "Устройство подтверждено сервером")
            if "Вы уже работаете" in text or "Вы уже устроены" in text or "already" in low_url:
                return JobResult("already_working", shift_end, "Персонаж уже работает")
            if "Прошло меньше часа" in text or "часа с последнего устройства" in text:
                return JobResult("cooldown", shift_end, "Сервер сообщил о перерыве между устройствами")
            if is_character_busy_text(text):
                return JobResult("busy", shift_end, "Персонаж занят боем или перемещением")
            returned_job_form = response_soup.find("form", id="getjob_form")
            returned_code = returned_job_form and (
                returned_job_form.find("input", id="code")
                or returned_job_form.find("input", attrs={"name": "code"})
            )
            if returned_code or "неверный код" in text.lower():
                return JobResult(
                    "captcha_failed",
                    shift_end,
                    "Сервер не принял код и выдал новую текстовую капчу",
                )
            return JobResult("error", shift_end, f"Неожиданный ответ: {response.url}; {text[:160]}")
        except requests.RequestException as exc:
            return JobResult("network", message=str(exc))

    def _announce_schedule(self, reason: str) -> None:
        remaining = self.time_until_next_attempt()
        minutes = max(0, int((remaining.total_seconds() + 59) // 60)) if remaining else 0
        self.emit(
            "info",
            f"{reason}. Следующая попытка через {minutes} мин, в {self.next_attempt_time:%H:%M}",
        )

    def _schedule_after_success(self) -> None:
        """A confirmed personal job lasts an hour; enterprise shift clocks do not."""
        started_at = datetime.now()
        self.shift_end_time = started_at + timedelta(minutes=60)
        random_delay = random.randint(0, 300)
        self.next_attempt_time = self.shift_end_time + timedelta(seconds=random_delay)
        self.stats.set_active_job(self.shift_end_time, self.next_attempt_time)
        delay_minutes = int((random_delay + 59) // 60) if random_delay else 0
        self._announce_schedule(
            f"Работа подтверждена: 60 мин + случайная задержка {delay_minutes} мин"
        )

    def _schedule_work_status(self, status: WorkGuildStatus) -> None:
        """Schedule from the personal status block, keeping jitter stable on restart."""
        now = datetime.now()
        allowed_at = status.next_allowed_at
        if not allowed_at:
            self.next_attempt_time = now + timedelta(minutes=15)
            self.shift_end_time = None
            self._announce_schedule("Страница персонажа не указала время; проверю позже")
            return
        if allowed_at <= now:
            self.next_attempt_time = now + timedelta(minutes=5)
            self.shift_end_time = allowed_at
            self._announce_schedule("Время уже истекло, но персонаж ещё занят; повтор через 5 мин")
            return

        saved = self.stats.active_job(now)
        if saved and abs((saved[0] - allowed_at).total_seconds()) <= 120:
            self.shift_end_time, self.next_attempt_time = saved
        else:
            self.shift_end_time = allowed_at
            self.next_attempt_time = allowed_at + timedelta(seconds=random.randint(0, 300))
            self.stats.set_active_job(self.shift_end_time, self.next_attempt_time)

        if status.state == "working":
            started = f" с {status.started_at:%H:%M}" if status.started_at else ""
            workplace = status.workplace_name or "неизвестное предприятие"
            reason = f"Работает: {workplace}{started}"
        else:
            remaining = status.remaining_minutes
            reason = (
                f"До следующего устройства по данным страницы: {remaining} мин"
                if remaining is not None
                else "Ожидаю разрешённое сервером время устройства"
            )
        self._announce_schedule(reason)

    def _schedule_busy_retry(self, reason: str = "Персонаж занят") -> None:
        self.shift_end_time = None
        self.next_attempt_time = datetime.now() + timedelta(minutes=5)
        self._announce_schedule(reason + "; повтор через 5 мин")

    def _schedule_existing_job(self, server_shift_end: datetime | None) -> None:
        """Restore a stable timer, falling back cautiously when HeroesWM omits it."""
        now = datetime.now()
        saved = self.stats.active_job(now)
        if saved:
            self.shift_end_time, self.next_attempt_time = saved
            self._announce_schedule("Использую сохранённое время текущей работы")
            return

        # Version 2.1 stored the success timestamp but not the timer.  Recover a
        # sensible one-time schedule when upgrading during an active job.
        last_success = self.stats.last_success_time()
        if last_success and timedelta(0) <= now - last_success <= timedelta(minutes=65):
            self.shift_end_time = last_success + timedelta(minutes=60)
            random_delay = random.randint(0, 300)
            self.next_attempt_time = self.shift_end_time + timedelta(seconds=random_delay)
            if self.next_attempt_time > now:
                self.stats.set_active_job(self.shift_end_time, self.next_attempt_time)
                self._announce_schedule("Восстановил время по последнему успешному устройству")
                return

        if server_shift_end and server_shift_end > now:
            self.shift_end_time = server_shift_end
            self.next_attempt_time = server_shift_end + timedelta(seconds=random.randint(20, 75))
            self.stats.set_active_job(self.shift_end_time, self.next_attempt_time)
            self._announce_schedule("Использую окончание работы, сообщённое сервером")
            return

        self.shift_end_time = None
        self.next_attempt_time = now + timedelta(minutes=15)
        self._announce_schedule(
            "Сервер не указал оставшееся время; контрольная проверка через 15 мин"
        )

    def _wait_until_due(self) -> None:
        while self.next_attempt_time and datetime.now() < self.next_attempt_time:
            self._check_stop()
            self.stop_event.wait(1.0)
        self.next_attempt_time = None

    def _run_impl(self) -> None:
        self.emit("info", "HeroesWM Worker запущен")
        try:
            while True:
                self._check_stop()
                self._wait_until_due()
                self.ensure_logged_in()
                try:
                    work_status = self.get_work_guild_status()
                except requests.RequestException as exc:
                    self.stats.error("сеть")
                    self.emit("warning", f"Статус персонажа не загрузился: {exc}")
                    self.next_attempt_time = datetime.now() + timedelta(minutes=2)
                    continue
                if work_status.state == "session_expired":
                    self.ensure_logged_in()
                    continue
                if work_status.state in {"working", "cooldown"}:
                    self._schedule_work_status(work_status)
                    continue
                if work_status.state == "busy":
                    self._schedule_busy_retry("Персонаж в бою или перемещается по карте")
                    continue
                try:
                    enterprises = self.get_enterprises()
                except SessionExpired:
                    self._schedule_busy_retry(
                        "Сессия изменилась на другом устройстве; не мешаю пользователю"
                    )
                    continue
                except requests.RequestException as exc:
                    self.stats.error("сеть")
                    self.emit("error", f"Карта не загрузилась: {exc}")
                    self.next_attempt_time = datetime.now() + timedelta(minutes=2)
                    continue
                if not enterprises:
                    self.stats.error("карта/предприятия")
                    self.emit("warning", "На карте не найдено предприятий; повтор через 1 мин")
                    self.next_attempt_time = datetime.now() + timedelta(minutes=1)
                    continue

                finished_cycle = False
                pages_without_form = 0
                for enterprise in enterprises:
                    self._check_stop()
                    self.emit(
                        "info",
                        f"Проверяю {CATEGORY_LABELS.get(enterprise.category)}: "
                        f"{enterprise.name} (зарплата {enterprise.salary})",
                    )
                    try:
                        page = self.get_job_page(enterprise)
                    except requests.RequestException as exc:
                        self.stats.error("сеть")
                        self.emit("warning", f"Предприятие {enterprise.object_id}: {exc}")
                        continue
                    if page.status == "session_expired":
                        self._schedule_busy_retry(
                            "Сессия изменилась на другом устройстве; не выполняю мгновенный повторный вход"
                        )
                        finished_cycle = True
                        break
                    if page.status in {"already_working", "cooldown"}:
                        self.emit("info", "Персонаж уже устроен")
                        self._schedule_existing_job(page.shift_end)
                        finished_cycle = True
                        break
                    if page.status == "busy":
                        self._schedule_busy_retry("Персонаж в бою или перемещается по карте")
                        finished_cycle = True
                        break
                    if page.status == "no_form":
                        pages_without_form += 1
                        self.emit(
                            "info",
                            f"Пропускаю {enterprise.name}: сервер не дал форму устройства",
                        )
                        self._human_delay(0.15, 0.35)
                        continue
                    if page.status == "no_slots":
                        self.emit(
                            "info",
                            f"Пропускаю {enterprise.name}: свободных мест нет",
                        )
                        self._human_delay(0.15, 0.35)
                        continue
                    if page.status != "ready":
                        self.stats.error("форма устройства")
                        self.emit("warning", f"Неизвестное состояние формы: {page.status}")
                        continue

                    if page.slots is not None:
                        self.emit(
                            "info",
                            f"Доступно мест: {page.slots}; пытаюсь устроиться с зарплатой {enterprise.salary}",
                        )

                    for captcha_attempt in range(1, 4):
                        token = self.solve_challenge(page)
                        if page.challenge.kind != "none" and token is None:
                            self.stats.error("капча")
                            if page.challenge.kind in {"recaptcha", "turnstile"}:
                                self.emit(
                                    "warning",
                                    f"{page.challenge.kind} не решена; повторная проверка через 5 мин",
                                )
                                self.next_attempt_time = datetime.now() + timedelta(minutes=5)
                                finished_cycle = True
                                break
                            self._schedule_unsolved_text_captcha()
                            finished_cycle = True
                            break
                        result = self.take_job(page, token)
                        if result.status == "success":
                            if page.challenge.kind == "image":
                                self._label_last_captcha(token)
                            self._report_last_service_solution(True)
                            self.stats.success()
                            self.emit("info", "✓ Устроился на работу!")
                            try:
                                confirmed = self.get_work_guild_status()
                            except requests.RequestException:
                                confirmed = WorkGuildStatus("unknown")
                            if confirmed.state in {"working", "cooldown"}:
                                workplace = confirmed.workplace_name or "предприятие не указано"
                                started = (
                                    f" с {confirmed.started_at:%H:%M}"
                                    if confirmed.started_at
                                    else ""
                                )
                                self.emit(
                                    "info",
                                    f"✓ Страница персонажа подтверждает: {workplace}{started}",
                                )
                                # The page exposes only HH:MM.  Use the precise
                                # POST-confirmation moment for a true 60-65 minute wait.
                                self._schedule_after_success()
                            else:
                                self._schedule_after_success()
                            finished_cycle = True
                            break
                        if result.status == "already_working":
                            self.emit("info", "Сервер подтвердил, что персонаж уже работает")
                            self._schedule_existing_job(result.shift_end)
                            finished_cycle = True
                            break
                        if result.status == "cooldown":
                            self._schedule_existing_job(result.shift_end)
                            finished_cycle = True
                            break
                        if result.status == "busy":
                            self._schedule_busy_retry("Персонаж в бою или перемещается по карте")
                            finished_cycle = True
                            break
                        if result.status == "session_expired":
                            self._schedule_busy_retry(
                                "Сессия изменилась на другом устройстве; повторный вход отложен"
                            )
                            finished_cycle = True
                            break
                        if result.status == "captcha_failed":
                            self._remember_captcha_rejection(token)
                            self.stats.error("неверная капча")
                            if captcha_attempt < 3:
                                self.emit(
                                    "warning",
                                    f"Капча отклонена ({captcha_attempt}/3), обновляю защитные поля формы",
                                )
                                self._human_delay(0.8, 1.5)
                                page = self.get_job_page(enterprise)
                                if page.status == "session_expired":
                                    self._schedule_busy_retry(
                                        "Сессия изменилась на другом устройстве; повторный вход отложен"
                                    )
                                    finished_cycle = True
                                    break
                                if page.status != "ready" or not page.form_action:
                                    self.emit("warning", "Форма или локация изменилась; новый обход через 1 мин")
                                    self.next_attempt_time = datetime.now() + timedelta(minutes=1)
                                    finished_cycle = True
                                    break
                                continue
                            remaining = self._remaining_local_candidates()
                            if self._last_text_captcha_source == "service":
                                image_key = self._last_captcha_image_key or ""
                                service_attempts = self._service_attempt_counts.get(image_key, 0)
                                if service_attempts < SERVICE_TEXT_CAPTCHA_MAX_ATTEMPTS:
                                    self.emit(
                                        "warning",
                                        "Ответ платного сервиса отклонён; запрошу другое решение через 2 сек",
                                    )
                                    self.next_attempt_time = datetime.now() + timedelta(seconds=2)
                                else:
                                    self._last_text_captcha_source = "service_exhausted"
                                    self._schedule_unsolved_text_captcha()
                            elif remaining:
                                self.emit(
                                    "warning",
                                    "Три разных ответа отклонены; старые коды запомнил, "
                                    f"осталось новых локальных вариантов: {remaining}. "
                                    f"Продолжу ту же капчу через {TEXT_CAPTCHA_RETRY_SECONDS} сек",
                                )
                                self.next_attempt_time = datetime.now() + timedelta(seconds=TEXT_CAPTCHA_RETRY_SECONDS)
                            elif self.api_key and self.captcha_mode in {"auto", "local"}:
                                self.emit(
                                    "warning",
                                    "Локальный лимит исчерпан; переключусь на RuCaptcha/2Captcha через 2 сек",
                                )
                                self.next_attempt_time = datetime.now() + timedelta(seconds=2)
                            else:
                                self._last_text_captcha_source = "local_no_api"
                                self._schedule_unsolved_text_captcha()
                            finished_cycle = True
                            break
                        self.stats.error("сеть" if result.status == "network" else "устройство")
                        self.emit("warning", result.message or "Устройство не подтверждено")
                        break
                    if finished_cycle:
                        break

                if not finished_cycle and self.next_attempt_time is None:
                    if pages_without_form:
                        self._schedule_busy_retry(
                            "Формы устройства недоступны: возможен бой или переход по карте"
                        )
                    else:
                        self.emit("warning", "Подходящих свободных мест нет; повтор через 1 мин")
                        self.next_attempt_time = datetime.now() + timedelta(minutes=1)
        except StopWorker:
            self.emit("info", "Скрипт остановлен")
        except Exception as exc:
            self.stats.error("непредвиденная ошибка")
            self.emit("error", f"Непредвиденная ошибка: {exc}")
            raise

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._run_impl()
                return
            except Exception:
                if self.stop_event.is_set():
                    return
                self.next_attempt_time = datetime.now() + timedelta(minutes=5)
                self.emit("warning", "После ошибки работа будет продолжена через 5 минут")
                try:
                    self._wait_until_due()
                except StopWorker:
                    return


def response_url_equal(left: str, right: str) -> bool:
    return left.rstrip("/") == right.rstrip("/")
