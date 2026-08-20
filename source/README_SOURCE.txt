HeroesWM Worker 3.9.0 Production — исходный код

Стратегия: 3.9.0-policy-improvement-production
OOS series: 3.9.0-production-freeze-2026-08-20

Эта сборка сохраняет policy-improvement 3.9 без изменения весов, глубины
поиска, opponent model, discard semantics или terminal/quiescence logic.
Производительность восстановлена эквивалентными кэшами детерминированных
переходов, ответов и вероятностей внутри одного решения.

Monte Carlo использует выбранный серверный таймер:
- 15 сек. -> до 11 сек. анализа / 1600 particles;
- 30 сек. -> до 26 сек. анализа / 4000 particles;
- 40 сек. -> до 36 сек. анализа / 6000 particles.
Фактический deadline всегда дополнительно ограничен текущим остатком
серверного таймера минус 4 секунды safety reserve.

Проверки production-приёмки:
- 131 тест пройден;
- симулятор: 35 973 / 35 973 точных переходов, все 102 карты;
- fixed seed/particles: 0 расхождений score, ranking и selected action;
- median online particles: 120 -> 360 на фиксированных 24 состояниях;
- holdout mean regret: 1.048 pp; p95: 6.216 pp;
- immediate-threat Brier delta статистически не доказана (95% CI включает 0).

Основные файлы:
- main.py — точка входа;
- gui.py — интерфейс;
- core.py — устройство на работу и сетевой цикл;
- card_game.py — карточная игра и policy;
- policy_models.json — зафиксированные параметры policy;
- cards_catalog.json — каталог 102 карт;
- storage.py — настройки и журналы.

Полный production-отчёт и исходные JSON-результаты находятся рядом со
сборкой в каталоге work/production_3_9.
