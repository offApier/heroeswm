# HeroesWM Worker 3.9.0 Production — итоговый отчёт

Решение: **RELEASE**. Улучшенная policy 3.9 заморожена; веса и игровая
логика не перенастраивались. Выполнены только implementation-equivalent
оптимизации.

## Профиль и bottlenecks

Фиксированный workload: 8 репрезентативных состояний, по 12 одинаковых
particles. Общее время уменьшилось с 5.545 до 2.598 с (−53.1%). Профиль
inclusive, поэтому вложенные проценты не складываются в 100%.

| Узел | До, с | После, с | Calls | p95 до/после, мс |
|---|---:|---:|---:|---:|
| deterministic simulation | 2.187 | 0.821 | 122590 / 120934 | 25.0 / 14.2 |
| PLAY evaluation | 1.901 | 1.449 | 28 / 28 | 114.0 / 83.4 |
| DISCARD evaluation | 3.592 | 1.106 | 48 / 48 | 114.7 / 60.9 |
| replacement draw | 0.172 | 0.143 | 1259 / 1259 | 0.261 / 0.241 |
| opponent response | 5.248 | 2.362 | 912 / 912 | 16.32 / 11.58 |
| extra-turn/quiescence | 0.055 | 0.050 | 44 / 44 | 2.53 / 2.68 |
| Q/value | 0.032 | 0.032 | 365 / 365 | 0.199 / 0.194 |
| feature extraction | 0.023 | 0.024 | 365 / 365 | 0.193 / 0.180 |
| diagnostics | 0.015 | 0.008 | 24 / 24 | 1.83 / 1.33 |

Главные bottlenecks были в повторном opponent-response/next-win анализе,
детерминированном simulate и DISCARD evaluation. Отдельный opponent-policy
ранжировщик runtime не вызывает: его поведение уже дистиллировано в policy
teacher.

## Эквивалентные оптимизации

- immutable-конструирование PlayerState вместо рекурсивного `asdict`;
- cache детерминированного `simulate(state, action)`;
- decision-local caches для reply, extra reply, next-win и quantile;
- общий unseen-pool finisher calculation для вариантов скрытой руки;
- общий initial particle set и last-seen prefix для всех actions/batches;
- очистка всех decision caches на границе решения, исключающая утечку между
  состояниями и reconnect.

Не менялись depth, discard continuation, terminal search, opponent model,
tower/horizon valuation, resource ETA и обученные Q/value параметры.

## Эквивалентность

24 разнообразных состояния × 40 фиксированных particles:

- ranking mismatches: 0;
- selected-action mismatches: 0;
- max absolute policy-score difference: 0;
- max absolute P(win) difference: 0.

## Online performance

Фиксированный набор из 24 состояний (normal, many actions/discards,
extra-turn, terminal race, complex belief, reconnect, high resources),
15-секундный игровой бюджет:

| Метрика particles | До | После |
|---|---:|---:|
| p10 | 80 | 200 |
| p25 | 80 | 200 |
| median | 120 | 360 |
| p75 | 170 | 810 |
| p90 | 200 | 1200 |
| p95 | 234 | 1200 |
| min / max | 40 / 240 | 200 / 1200 |
| fraction <200 | 75.0% | 0% |
| fraction >=200 | 25.0% | 100% |
| fraction >=400 | 0% | 45.8% |

Deadline precheck: 22 -> 5; deadline-in-batch: 0 -> 0; hard timeout: 0 -> 0.
На 23 состояниях, покрытых high-budget oracle, mean regret выбранного online
действия улучшился 0.661 -> 0.520 pp, p95 3.182 -> 2.620 pp.

## Decision quality

Holdout после оптимизации полностью совпадает с замороженной policy:

- mean regret: 1.048 pp (historical 4.590 pp);
- median: 0;
- p90: 3.403 pp;
- p95: 6.216 pp (historical 24.667 pp);
- mistakes >2 pp: 186;
- mistakes >5 pp: 88 (historical 330);
- mistakes >10 pp: 31.

Random representative oracle sample также не изменился благодаря точной
fixed-particle equivalence: candidate mean 1.242 pp против historical 5.149
pp; p90 3.804 против 16.936 pp; >5 pp: 26 против 108.

## Threat calibration

Game-level paired bootstrap: 51 game_id, 5000 resamples.

- delta Brier candidate−historical: +0.000406;
- 95% CI: [−0.004445; +0.005386];
- P(candidate worse): 0.566.

CI включает 0, поэтому ухудшение статистически не доказано. Calibration и
ranking не менялись; ложных exact-100% у candidate нет.

## Validation

- tests: 131 passed;
- simulator: 35 973 / 35 973 exact, все 102 карты;
- runtime feature parity: PASS;
- hard timeouts: 0;
- новая внешняя OOS-серия не использовалась.

## Server-timer-aware Monte Carlo

Без изменения номера версии устранён фиксированный 8-секундный потолок:

- режим 15 сек.: compute cap 11 сек., particle cap 1600;
- режим 30 сек.: compute cap 26 сек., particle cap 4000;
- режим 40 сек.: compute cap 36 сек., particle cap 6000.

Фактический deadline вычисляется как
`min(decision_start + mode_compute_cap, server_deadline - 4 sec)`.
Поэтому первый ответ со значением 119/121 сек. не расширяет выбранный режим,
а уже уменьшившийся серверный остаток безопасно сокращает анализ. При
статистической сходимости расчёт может завершиться раньше. Fixed-seed проверка
после изменения: 24 состояния × 40 particles, 0 расхождений score/ranking/
selected action.

Версия: `3.9.0-policy-improvement-production`.
OOS series: `3.9.0-production-freeze-2026-08-20`.
Финальные SHA-256 приведены в `SHA256SUMS_PRODUCTION.txt` после сборки.
