from __future__ import annotations

import base64
import ctypes
import json
import os
import threading
from collections import Counter
from ctypes import wintypes
from datetime import date, datetime
from pathlib import Path
from typing import Any


APP_DIR = Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HeroesWMWorker"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(data: bytes) -> tuple[_DataBlob, Any]:
    buffer = ctypes.create_string_buffer(data)
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer


def protect_secret(value: str) -> str:
    if not value:
        return ""
    raw = value.encode("utf-8")
    if os.name != "nt":
        return "plain:" + base64.b64encode(raw).decode("ascii")
    source, source_buffer = _blob(raw)
    output = _DataBlob()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(source), "HeroesWM Worker", None, None, None, 0, ctypes.byref(output)
    ):
        raise ctypes.WinError()
    try:
        protected = ctypes.string_at(output.pbData, output.cbData)
        return "dpapi:" + base64.b64encode(protected).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)
        del source_buffer


def unprotect_secret(value: str) -> str:
    if not value:
        return ""
    if value.startswith("plain:"):
        return base64.b64decode(value[6:]).decode("utf-8")
    if not value.startswith("dpapi:") or os.name != "nt":
        return ""
    source, source_buffer = _blob(base64.b64decode(value[6:]))
    output = _DataBlob()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)
    ):
        return ""
    try:
        return ctypes.string_at(output.pbData, output.cbData).decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)
        del source_buffer


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or APP_DIR / "settings.json"

    def load(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "login": "",
            "password": "",
            "api_key": "",
            "remember": True,
            "strategy": "salary",
            "captcha_mode": "auto",
            "categories": ["mn", "fc", "sh"],
            "card_timeout": 40,
            "card_deck_type": 1,
            "card_continuous": False,
            "card_max_stake": 0,
            "card_work_priority": True,
        }
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            defaults.update({key: value for key, value in data.items() if key in defaults})
            defaults["password"] = unprotect_secret(data.get("password_protected", ""))
            defaults["api_key"] = unprotect_secret(data.get("api_key_protected", ""))
        except (OSError, ValueError, TypeError):
            pass
        return defaults

    def save(self, settings: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        remember = bool(settings.get("remember", True))
        data = {
            "version": 3,
            "login": settings.get("login", "") if remember else "",
            "password_protected": protect_secret(settings.get("password", "")) if remember else "",
            "api_key_protected": protect_secret(settings.get("api_key", "")) if remember else "",
            "remember": remember,
            "strategy": settings.get("strategy", "salary"),
            "captcha_mode": settings.get("captcha_mode", "auto"),
            "categories": list(settings.get("categories") or ["mn", "fc", "sh"]),
            "card_timeout": int(settings.get("card_timeout", 40)),
            "card_deck_type": int(settings.get("card_deck_type", 1)),
            "card_continuous": bool(settings.get("card_continuous", False)),
            "card_max_stake": max(0, int(settings.get("card_max_stake", 0))),
            "card_work_priority": bool(settings.get("card_work_priority", True)),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


class StatsTracker:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or APP_DIR / "stats.json"
        self._lock = threading.Lock()
        self._session_started = datetime.now()
        self._session_success = 0
        self._session_errors: Counter[str] = Counter()
        self._data: dict[str, Any] = {"version": 2, "days": {}}
        self._load()

    def _load(self) -> None:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("days"), dict):
                self._data = loaded
        except (OSError, ValueError, TypeError):
            pass

    def _day(self) -> dict[str, Any]:
        key = date.today().isoformat()
        return self._data.setdefault("days", {}).setdefault(
            key, {"success": 0, "errors": {}, "last_event": None}
        )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def success(self) -> None:
        with self._lock:
            self._session_success += 1
            day = self._day()
            day["success"] = int(day.get("success", 0)) + 1
            occurred_at = datetime.now().isoformat(timespec="seconds")
            day["last_event"] = occurred_at
            day["last_success"] = occurred_at
            self._save()

    def set_active_job(self, shift_end: datetime, next_attempt: datetime) -> None:
        """Persist the personal work timer so restarting the GUI cannot move it."""
        with self._lock:
            self._data["active_job"] = {
                "shift_end": shift_end.isoformat(timespec="seconds"),
                "next_attempt": next_attempt.isoformat(timespec="seconds"),
            }
            self._save()

    def active_job(self, now: datetime | None = None) -> tuple[datetime, datetime] | None:
        now = now or datetime.now()
        with self._lock:
            value = self._data.get("active_job")
            if not isinstance(value, dict):
                return None
            try:
                shift_end = datetime.fromisoformat(str(value["shift_end"]))
                next_attempt = datetime.fromisoformat(str(value["next_attempt"]))
            except (KeyError, TypeError, ValueError):
                return None
            if next_attempt <= now:
                return None
            return shift_end, next_attempt

    def last_success_time(self) -> datetime | None:
        """Read the last confirmed success, including migration from version 1 stats."""
        with self._lock:
            day = self._day()
            value = day.get("last_success")
            if not value and int(day.get("success", 0)) > 0:
                value = day.get("last_event")
            try:
                return datetime.fromisoformat(str(value)) if value else None
            except (TypeError, ValueError):
                return None

    def error(self, category: str) -> None:
        category = category or "прочее"
        with self._lock:
            self._session_errors[category] += 1
            day = self._day()
            errors = day.setdefault("errors", {})
            errors[category] = int(errors.get(category, 0)) + 1
            day["last_event"] = datetime.now().isoformat(timespec="seconds")
            self._save()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            day = dict(self._day())
            return {
                "session_started": self._session_started,
                "session_success": self._session_success,
                "session_errors": dict(self._session_errors),
                "today_success": int(day.get("success", 0)),
                "today_errors": dict(day.get("errors", {})),
            }
