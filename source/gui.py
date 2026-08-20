from __future__ import annotations

import io
import os
import queue
import threading
import tkinter as tk
from datetime import datetime, timedelta
from tkinter import messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

from card_game import CardGameBot, GameState
from core import CATEGORY_LABELS, HeroesWMWorker, get_captcha_samples_dir
from storage import APP_DIR, SettingsStore, StatsTracker


STRATEGIES = {
    "Самая высокая зарплата": "salary",
    "Больше свободных мест": "slots",
    "Первое подходящее": "first",
}
CAPTCHA_MODES = {
    "Авто — 6 попыток локально, затем API": "auto",
    "Вручную — показать текстовую картинку": "manual",
}


def should_pause_cards_for_work(
    worker: HeroesWMWorker | None,
    worker_thread: threading.Thread | None,
    enabled: bool,
    guard_seconds: int = 15 * 60,
) -> bool:
    """Return whether card matchmaking must yield to the work cycle."""
    if not enabled or worker is None or worker_thread is None or not worker_thread.is_alive():
        return False
    if worker.next_attempt_time is None:
        # The work cycle is currently checking enterprises / submitting a job.
        return True
    remaining = worker.time_until_next_attempt()
    # ``time_until_next_attempt`` returns timedelta (or None once already due),
    # never a raw number of seconds.
    return remaining is None or remaining <= timedelta(seconds=max(0, guard_seconds))


class HeroesWMGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HeroesWM Worker 3.9.1 P(win) Candidate")
        self.root.geometry("940x760")
        self.root.minsize(820, 650)
        self.settings_store = SettingsStore()
        self.settings = self.settings_store.load()
        self.stats = StatsTracker()
        self.worker: HeroesWMWorker | None = None
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._captcha_window: tk.Toplevel | None = None
        self.card_bot: CardGameBot | None = None
        self.card_thread: threading.Thread | None = None
        self.card_stop_event = threading.Event()
        self.card_log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.card_state_queue: queue.Queue[GameState] = queue.Queue()
        self._card_window: tk.Toplevel | None = None
        self.card_status_var: tk.StringVar | None = None
        self.card_score_var: tk.StringVar | None = None
        self.card_board_var: tk.StringVar | None = None
        self.card_log_text: scrolledtext.ScrolledText | None = None
        self.card_start_button: ttk.Button | None = None
        self.card_stop_button: ttk.Button | None = None
        self._build_ui()
        self._load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(250, self._poll)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1)

        account = ttk.LabelFrame(outer, text="Аккаунт", padding=10)
        account.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        account.columnconfigure(1, weight=1)
        account.columnconfigure(3, weight=1)
        ttk.Label(account, text="Логин:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.login_var = tk.StringVar()
        ttk.Entry(account, textvariable=self.login_var).grid(row=0, column=1, sticky="ew", padx=(0, 12))
        ttk.Label(account, text="Пароль:").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.password_var = tk.StringVar()
        ttk.Entry(account, textvariable=self.password_var, show="•").grid(row=0, column=3, sticky="ew")
        self.remember_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            account,
            text="Запомнить на этом компьютере (пароль шифруется Windows)",
            variable=self.remember_var,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        options = ttk.LabelFrame(outer, text="Поиск и капча", padding=10)
        options.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)
        ttk.Label(options, text="Стратегия:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.strategy_var = tk.StringVar()
        ttk.Combobox(
            options,
            textvariable=self.strategy_var,
            state="readonly",
            values=list(STRATEGIES),
        ).grid(row=0, column=1, sticky="ew", padx=(0, 12))
        ttk.Label(options, text="Режим капчи:").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.captcha_var = tk.StringVar()
        captcha_box = ttk.Combobox(
            options,
            textvariable=self.captcha_var,
            state="readonly",
            values=list(CAPTCHA_MODES),
        )
        captcha_box.grid(row=0, column=3, sticky="ew")
        captcha_box.bind("<<ComboboxSelected>>", lambda _event: self._update_api_state())

        ttk.Label(options, text="API-ключ:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        self.api_key_var = tk.StringVar()
        self.api_entry = ttk.Entry(options, textvariable=self.api_key_var, show="•")
        self.api_entry.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(8, 0))

        ttk.Label(options, text="Категории:").grid(row=2, column=0, sticky="nw", pady=(10, 0))
        categories = ttk.Frame(options)
        categories.grid(row=2, column=1, columnspan=3, sticky="w", pady=(8, 0))
        self.category_vars: dict[str, tk.BooleanVar] = {}
        for column, (code, label) in enumerate(CATEGORY_LABELS.items()):
            variable = tk.BooleanVar(value=True)
            self.category_vars[code] = variable
            ttk.Checkbutton(categories, text=label, variable=variable).grid(
                row=0, column=column, padx=(0, 16)
            )

        controls = ttk.Frame(outer)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(5, weight=1)
        self.start_button = ttk.Button(controls, text="Запустить", command=self.start_worker)
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="Остановить", command=self.stop_worker, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 8))
        ttk.Button(controls, text="Очистить журнал", command=self._clear_log).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(controls, text="Папка капч", command=self._open_captcha_samples).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(controls, text="Карточный бой", command=self._open_card_game).grid(row=0, column=4)
        self.timer_var = tk.StringVar(value="Следующая попытка: —")
        ttk.Label(controls, textvariable=self.timer_var, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=5, sticky="e"
        )

        stats_frame = ttk.LabelFrame(outer, text="Статистика", padding=8)
        stats_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)
        self.session_stats_var = tk.StringVar()
        self.day_stats_var = tk.StringVar()
        ttk.Label(stats_frame, textvariable=self.session_stats_var, justify="left").grid(
            row=0, column=0, sticky="nw", padx=(0, 16)
        )
        ttk.Label(stats_frame, textvariable=self.day_stats_var, justify="left").grid(
            row=0, column=1, sticky="nw"
        )

        log_frame = ttk.LabelFrame(outer, text="Журнал (текст можно выделять и копировать)", padding=6)
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            height=18,
            font=("Consolas", 9),
            state="disabled",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.tag_configure("error", foreground="#b00020")
        self.log_text.tag_configure("warning", foreground="#9a5b00")
        self.log_text.tag_configure("info", foreground="#222222")
        self.log_text.bind("<Control-a>", self._select_all_log)
        self.log_text.bind("<Button-3>", self._show_log_menu)
        self.log_menu = tk.Menu(self.root, tearoff=False)
        self.log_menu.add_command(label="Копировать", command=lambda: self.log_text.event_generate("<<Copy>>"))
        self.log_menu.add_command(label="Выделить всё", command=self._select_all_log)

    def _load_settings(self) -> None:
        self.login_var.set(self.settings.get("login", ""))
        self.password_var.set(self.settings.get("password", ""))
        self.api_key_var.set(self.settings.get("api_key", ""))
        self.remember_var.set(bool(self.settings.get("remember", True)))
        strategy = self.settings.get("strategy", "salary")
        self.strategy_var.set(next((label for label, code in STRATEGIES.items() if code == strategy), next(iter(STRATEGIES))))
        mode = self.settings.get("captcha_mode", "auto")
        if mode in {"local", "service"}:
            mode = "auto"
        self.captcha_var.set(next((label for label, code in CAPTCHA_MODES.items() if code == mode), next(iter(CAPTCHA_MODES))))
        selected = set(self.settings.get("categories") or CATEGORY_LABELS)
        for code, variable in self.category_vars.items():
            variable.set(code in selected)
        self._update_api_state()

    def _collect_settings(self) -> dict[str, object]:
        return {
            "login": self.login_var.get().strip(),
            "password": self.password_var.get(),
            "api_key": self.api_key_var.get().strip(),
            "remember": self.remember_var.get(),
            "strategy": STRATEGIES[self.strategy_var.get()],
            "captcha_mode": CAPTCHA_MODES[self.captcha_var.get()],
            "categories": [code for code, variable in self.category_vars.items() if variable.get()],
            "card_timeout": int(self.settings.get("card_timeout", 40)),
            "card_deck_type": int(self.settings.get("card_deck_type", 1)),
            "card_continuous": bool(self.settings.get("card_continuous", False)),
            "card_max_stake": max(0, int(self.settings.get("card_max_stake", 0))),
            "card_work_priority": bool(self.settings.get("card_work_priority", True)),
        }

    def _update_api_state(self) -> None:
        # The key is optional. It is used after six unsuccessful local text
        # answers and immediately for reCAPTCHA/Turnstile.
        self.api_entry.configure(state="normal")

    def start_worker(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        settings = self._collect_settings()
        if not settings["login"] or not settings["password"]:
            messagebox.showerror("HeroesWM Worker", "Укажите логин и пароль.")
            return
        if not settings["categories"]:
            messagebox.showerror("HeroesWM Worker", "Выберите хотя бы одну категорию предприятий.")
            return
        try:
            self.settings_store.save(settings)
        except Exception as exc:
            messagebox.showwarning("HeroesWM Worker", f"Настройки не удалось сохранить: {exc}")
        self.stop_event = threading.Event()
        self.worker = HeroesWMWorker(
            str(settings["login"]),
            str(settings["password"]),
            str(settings["api_key"]),
            strategy=str(settings["strategy"]),
            categories=settings["categories"],
            captcha_mode=str(settings["captcha_mode"]),
            stop_event=self.stop_event,
            log_callback=self._queue_log,
            manual_captcha_callback=self._manual_captcha,
            stats=self.stats,
        )
        self.worker_thread = threading.Thread(target=self._run_worker, name="HeroesWMWorker", daemon=True)
        self.worker_thread.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_inputs_state("disabled")

    def _run_worker(self) -> None:
        try:
            assert self.worker is not None
            self.worker.run()
        except Exception as exc:
            self._queue_log("error", f"Рабочий поток завершился: {exc}")
        finally:
            self.root.after(0, self._worker_finished)

    def stop_worker(self) -> None:
        self.stop_event.set()
        if self._captcha_window:
            self._captcha_window.destroy()
            self._captcha_window = None
        self._queue_log("info", "Запрошена остановка...")

    def _worker_finished(self) -> None:
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._set_inputs_state("normal")
        self._update_api_state()
        self.timer_var.set("Следующая попытка: —")

    def _set_inputs_state(self, state: str) -> None:
        # Buttons remain usable; entry/combobox changes are simply deferred to
        # the next launch. Walking descendants keeps the layout code readable.
        for child in self.root.winfo_children():
            self._set_descendants_state(child, state)

    def _set_descendants_state(self, widget: tk.Misc, state: str) -> None:
        for child in widget.winfo_children():
            if child in {self.start_button, self.stop_button, self.log_text}:
                continue
            if isinstance(child, (ttk.Entry, ttk.Combobox, ttk.Checkbutton)):
                try:
                    target_state = "readonly" if state == "normal" and isinstance(child, ttk.Combobox) else state
                    child.configure(state=target_state)
                except tk.TclError:
                    pass
            self._set_descendants_state(child, state)

    def _queue_log(self, level: str, message: str) -> None:
        self.log_queue.put((level, message))

    def _poll(self) -> None:
        try:
            while True:
                level, message = self.log_queue.get_nowait()
                self._append_log(level, message)
        except queue.Empty:
            pass
        if self.worker and self.worker_thread and self.worker_thread.is_alive():
            remaining = self.worker.time_until_next_attempt()
            if remaining:
                seconds = max(0, int(remaining.total_seconds()))
                hours, remainder = divmod(seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                self.timer_var.set(
                    f"Следующая попытка: {self.worker.next_attempt_time:%H:%M:%S} "
                    f"(через {hours:02d}:{minutes:02d}:{seconds:02d})"
                )
            else:
                self.timer_var.set("Следующая попытка: сейчас")
        self._update_stats()
        self._poll_card_game()
        self.root.after(300, self._poll)

    def _append_log(self, level: str, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{datetime.now():%H:%M:%S} {message}\n", level)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_stats(self) -> None:
        snapshot = self.stats.snapshot()
        session_errors = self._format_errors(snapshot["session_errors"])
        day_errors = self._format_errors(snapshot["today_errors"])
        self.session_stats_var.set(
            f"С момента запуска: устройств — {snapshot['session_success']}\nОшибки: {session_errors}"
        )
        self.day_stats_var.set(
            f"Сегодня: устройств — {snapshot['today_success']}\nОшибки: {day_errors}"
        )

    @staticmethod
    def _format_errors(errors: dict[str, int]) -> str:
        if not errors:
            return "нет"
        return ", ".join(f"{name}: {count}" for name, count in sorted(errors.items()))

    def _manual_captcha(self, image_bytes: bytes) -> str | None:
        event = threading.Event()
        result: dict[str, str | None] = {"code": None}
        self.root.after(0, lambda: self._open_captcha_dialog(image_bytes, event, result))
        while not event.wait(0.2):
            if self.stop_event.is_set():
                return None
        return result["code"]

    def _open_captcha_dialog(
        self, image_bytes: bytes, event: threading.Event, result: dict[str, str | None]
    ) -> None:
        window = tk.Toplevel(self.root)
        self._captcha_window = window
        window.title("Введите код капчи")
        window.transient(self.root)
        window.grab_set()
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both", expand=True)
        image = Image.open(io.BytesIO(image_bytes))
        if image.width < 500:
            image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label = ttk.Label(frame, image=photo)
        label.image = photo
        label.pack(pady=(0, 10))
        value = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=value, font=("Segoe UI", 18), justify="center")
        entry.pack(fill="x")
        entry.focus_set()

        def finish(code: str | None) -> None:
            result["code"] = code
            self._captcha_window = None
            try:
                window.grab_release()
                window.destroy()
            finally:
                event.set()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(10, 0))
        ttk.Button(buttons, text="Отправить", command=lambda: finish(value.get().strip())).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(buttons, text="Пропустить", command=lambda: finish(None)).pack(side="left")
        entry.bind("<Return>", lambda _event: finish(value.get().strip()))
        window.protocol("WM_DELETE_WINDOW", lambda: finish(None))

    def _show_log_menu(self, event: tk.Event) -> None:
        self.log_menu.tk_popup(event.x_root, event.y_root)

    def _select_all_log(self, _event: tk.Event | None = None) -> str:
        self.log_text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _open_captcha_samples(self) -> None:
        try:
            sample_dir = get_captcha_samples_dir()
            sample_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(sample_dir)
        except OSError as exc:
            messagebox.showerror("HeroesWM Worker", f"Не удалось открыть папку капч: {exc}")

    def _open_card_game(self) -> None:
        if self._card_window and self._card_window.winfo_exists():
            self._card_window.deiconify()
            self._card_window.lift()
            return
        window = tk.Toplevel(self.root)
        self._card_window = window
        window.title("HeroesWM Worker — карточный бой")
        window.geometry("940x700")
        window.minsize(760, 560)
        window.protocol("WM_DELETE_WINDOW", self._hide_card_window)

        outer = ttk.Frame(window, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        settings_frame = ttk.LabelFrame(outer, text="Автоматическая игра", padding=10)
        settings_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        settings_frame.columnconfigure(8, weight=1)
        ttk.Label(settings_frame, text="Время на ход:").grid(row=0, column=0, padx=(0, 6))
        self.card_timeout_var = tk.StringVar(value=str(self.settings.get("card_timeout", 40)))
        ttk.Combobox(
            settings_frame,
            textvariable=self.card_timeout_var,
            values=("15", "30", "40"),
            width=5,
            state="readonly",
        ).grid(row=0, column=1, padx=(0, 14))
        ttk.Label(settings_frame, text="Колода:").grid(row=0, column=2, padx=(0, 6))
        deck_value = "Одна" if int(self.settings.get("card_deck_type", 1)) == 1 else "Бесконечная"
        self.card_deck_var = tk.StringVar(value=deck_value)
        ttk.Combobox(
            settings_frame,
            textvariable=self.card_deck_var,
            values=("Одна", "Бесконечная"),
            width=13,
            state="readonly",
        ).grid(row=0, column=3, padx=(0, 14))
        ttk.Label(settings_frame, text="Макс. ставка:").grid(row=0, column=4, padx=(0, 6))
        self.card_max_stake_var = tk.StringVar(value=str(self.settings.get("card_max_stake", 0)))
        ttk.Spinbox(
            settings_frame,
            textvariable=self.card_max_stake_var,
            from_=0,
            to=1000000,
            increment=1,
            width=8,
        ).grid(row=0, column=5, padx=(0, 14))
        self.card_start_button = ttk.Button(settings_frame, text="Начать", command=self._start_card_game)
        self.card_start_button.grid(row=0, column=6, padx=(0, 8))
        self.card_stop_button = ttk.Button(
            settings_frame, text="Остановить", command=self._stop_card_game, state="disabled"
        )
        self.card_stop_button.grid(row=0, column=7)
        ttk.Button(settings_frame, text="Папка записей", command=self._open_card_records).grid(
            row=0, column=8, sticky="e", padx=(8, 0)
        )

        self.card_continuous_var = tk.BooleanVar(value=bool(self.settings.get("card_continuous", False)))
        ttk.Checkbutton(
            settings_frame,
            text="Играть следующие партии автоматически",
            variable=self.card_continuous_var,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0), padx=(0, 14))
        self.card_work_priority_var = tk.BooleanVar(value=bool(self.settings.get("card_work_priority", True)))
        ttk.Checkbutton(
            settings_frame,
            text="Приоритет работы: не начинать бой за 15 мин и отменять ожидающую заявку",
            variable=self.card_work_priority_var,
        ).grid(row=1, column=4, columnspan=5, sticky="w", pady=(8, 0))

        self.card_status_var = tk.StringVar(value="Готов к запуску. 0 означает: не рисковать золотом.")
        ttk.Label(outer, textvariable=self.card_status_var, font=("Segoe UI", 10, "bold")).grid(
            row=1, column=0, sticky="w", pady=(0, 6)
        )
        self.card_score_var = tk.StringVar(value="Партий с запуска: 0; побед: 0; поражений: 0")
        self.card_board_var = tk.StringVar(value="Состояние партии: —")
        state_frame = ttk.LabelFrame(outer, text="Состояние", padding=8)
        state_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(state_frame, textvariable=self.card_score_var).pack(anchor="w")
        ttk.Label(state_frame, textvariable=self.card_board_var, justify="left").pack(anchor="w", pady=(4, 0))

        log_frame = ttk.LabelFrame(outer, text="Журнал решений", padding=6)
        log_frame.grid(row=3, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.card_log_text = scrolledtext.ScrolledText(
            log_frame, wrap="word", font=("Consolas", 9), state="disabled"
        )
        self.card_log_text.grid(row=0, column=0, sticky="nsew")
        for tag, color in (("error", "#b00020"), ("warning", "#9a5b00"), ("info", "#222222")):
            self.card_log_text.tag_configure(tag, foreground=color)
        self.card_log_text.bind("<Control-a>", self._select_all_card_log)
        self.card_log_text.bind("<Control-c>", self._copy_card_log)
        self.card_log_text.bind("<Button-3>", self._show_card_log_menu)
        self.card_log_menu = tk.Menu(window, tearoff=False)
        self.card_log_menu.add_command(label="Копировать", command=self._copy_card_log)
        self.card_log_menu.add_command(label="Выделить всё", command=self._select_all_card_log)
        self.card_log_menu.add_separator()
        self.card_log_menu.add_command(label="Открыть папку записей", command=self._open_card_records)
        if self.card_thread and self.card_thread.is_alive():
            self._set_card_controls(True)

    def _hide_card_window(self) -> None:
        if self._card_window:
            self._card_window.withdraw()

    def _start_card_game(self) -> None:
        if self.card_thread and self.card_thread.is_alive():
            return
        login = self.login_var.get().strip()
        password = self.password_var.get()
        if not login or not password:
            messagebox.showerror("Карточный бой", "Укажите логин и пароль в главном окне.")
            return
        timeout = int(self.card_timeout_var.get())
        deck_type = 1 if self.card_deck_var.get() == "Одна" else 2
        continuous = self.card_continuous_var.get()
        try:
            max_stake = max(0, int(self.card_max_stake_var.get()))
        except ValueError:
            messagebox.showerror("Карточный бой", "Максимальная ставка должна быть целым числом не меньше 0.")
            return
        work_priority = self.card_work_priority_var.get()
        self.settings["card_timeout"] = timeout
        self.settings["card_deck_type"] = deck_type
        self.settings["card_continuous"] = continuous
        self.settings["card_max_stake"] = max_stake
        self.settings["card_work_priority"] = work_priority
        try:
            values = self._collect_settings()
            values.update(
                card_timeout=timeout,
                card_deck_type=deck_type,
                card_continuous=continuous,
                card_max_stake=max_stake,
                card_work_priority=work_priority,
            )
            self.settings_store.save(values)
        except Exception as exc:
            self._queue_card_log("warning", f"Настройки не удалось сохранить: {exc}")

        self.card_stop_event = threading.Event()
        # Reuse an active employment session. Logging in a second time can
        # invalidate the first HeroesWM session. The employment loop spends
        # almost all of its time waiting, while card requests are short.
        if self.worker and self.worker_thread and self.worker_thread.is_alive():
            login_worker = self.worker
            self._queue_card_log("info", "Использую активную сессию основного Worker")
        else:
            login_worker = HeroesWMWorker(
                login,
                password,
                self.api_key_var.get().strip(),
                stop_event=self.card_stop_event,
                log_callback=self._queue_card_log,
                manual_captcha_callback=self._manual_captcha,
                stats=self.stats,
            )
        # Use the exact same session object instead of a one-time cookie copy.
        # A later re-login then updates both the work and card modules at once.
        card_session = login_worker.session

        def ensure_card_login() -> None:
            login_worker.ensure_logged_in()

        def should_pause_for_work() -> bool:
            return should_pause_cards_for_work(
                self.worker,
                self.worker_thread,
                work_priority,
            )

        self.card_bot = CardGameBot(
            card_session,
            login_name=login,
            ensure_login=ensure_card_login,
            stop_event=self.card_stop_event,
            log_callback=self._queue_card_log,
            state_callback=self.card_state_queue.put,
            timeout=timeout,
            deck_type=deck_type,
            continuous=continuous,
            max_stake=max_stake,
            work_pause_callback=should_pause_for_work,
        )
        self.card_thread = threading.Thread(target=self._run_card_game, name="HeroesWMCardGame", daemon=True)
        self.card_thread.start()
        self._set_card_controls(True)

    def _run_card_game(self) -> None:
        try:
            assert self.card_bot is not None
            self.card_bot.run()
        except Exception as exc:
            if self.card_stop_event.is_set():
                self._queue_card_log("info", "Автоигра остановлена")
            else:
                detail = str(exc).strip() or type(exc).__name__
                self._queue_card_log("error", f"Карточный поток завершился: {detail}")
        finally:
            self.root.after(0, lambda: self._set_card_controls(False))

    def _stop_card_game(self) -> None:
        if self.card_stop_event.is_set():
            return
        self.card_stop_event.set()
        if self.card_stop_button:
            self.card_stop_button.configure(state="disabled")
        self._queue_card_log("info", "Запрошена остановка карточной игры...")

    def _set_card_controls(self, running: bool) -> None:
        if self.card_start_button:
            self.card_start_button.configure(state="disabled" if running else "normal")
        if self.card_stop_button:
            self.card_stop_button.configure(state="normal" if running else "disabled")
        if self.card_status_var:
            self.card_status_var.set("Автоигра работает" if running else "Автоигра остановлена")

    def _queue_card_log(self, level: str, message: str) -> None:
        self.card_log_queue.put((level, message))

    def _poll_card_game(self) -> None:
        try:
            while True:
                level, message = self.card_log_queue.get_nowait()
                if self.card_status_var:
                    self.card_status_var.set(message)
                if self.card_log_text:
                    self.card_log_text.configure(state="normal")
                    self.card_log_text.insert("end", f"{datetime.now():%H:%M:%S} {message}\n", level)
                    self.card_log_text.see("end")
                    self.card_log_text.configure(state="disabled")
        except queue.Empty:
            pass
        latest: GameState | None = None
        try:
            while True:
                latest = self.card_state_queue.get_nowait()
        except queue.Empty:
            pass
        if latest and self.card_board_var:
            me, enemy = latest.me, latest.opponent
            turn = "наш" if latest.is_your_turn else "соперника"
            self.card_board_var.set(
                f"Игра #{latest.game_id}, ход {latest.turn} ({turn}), осталось {latest.time_left} сек.\n"
                f"Мы: башня {me.tower}, стена {me.wall}, руда/мана/отряды {me.ore}/{me.mana}/{me.army}, "
                f"производства {me.mine}/{me.monastery}/{me.barracks}.\n"
                f"Соперник: башня {enemy.tower}, стена {enemy.wall}, ресурсы "
                f"{enemy.ore}/{enemy.mana}/{enemy.army}, производства "
                f"{enemy.mine}/{enemy.monastery}/{enemy.barracks}."
            )
        if self.card_bot and self.card_score_var:
            self.card_score_var.set(
                f"Партий с запуска: {self.card_bot.games_played}; побед: {self.card_bot.wins}; "
                f"поражений: {self.card_bot.losses}"
            )

    def _select_all_card_log(self, _event: tk.Event | None = None) -> str:
        if self.card_log_text:
            self.card_log_text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _copy_card_log(self, _event: tk.Event | None = None) -> str:
        if not self.card_log_text:
            return "break"
        try:
            selected = self.card_log_text.get("sel.first", "sel.last")
        except tk.TclError:
            selected = self.card_log_text.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(selected)
        return "break"

    def _show_card_log_menu(self, event: tk.Event) -> None:
        self.card_log_menu.tk_popup(event.x_root, event.y_root)

    def _open_card_records(self) -> None:
        try:
            APP_DIR.mkdir(parents=True, exist_ok=True)
            (APP_DIR / "card_games").mkdir(exist_ok=True)
            (APP_DIR / "card_logs").mkdir(exist_ok=True)
            os.startfile(APP_DIR)
        except OSError as exc:
            messagebox.showerror("Карточный бой", f"Не удалось открыть папку записей: {exc}")

    def _on_close(self) -> None:
        self.stop_event.set()
        self.card_stop_event.set()
        try:
            self.settings_store.save(self._collect_settings())
        except Exception:
            pass
        self.root.destroy()


def run_gui() -> None:
    root = tk.Tk()
    HeroesWMGui(root)
    root.mainloop()
