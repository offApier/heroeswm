# HeroesWM Worker 3.9 — полный отчёт policy improvement

Дата: 2026-08-20  
Статус: **исследовательский кандидат отклонён для production-сборки**.  
Причина: policy-regret существенно улучшен, но строгий acceptance не пройден по online particle budget и immediate-threat Brier. После открытия финального holdout policy не изменялась; `.exe` намеренно не собрана.

## 1. Dataset inventory

- 850 JSON-сегментов, 767 уникальных `game_id`.
- 761 завершённая партия: 363 победы, 397 поражений, 1 ничья.
- 6 оборванных партий.
- 72 партии с reconnect, включая 62 с двумя, 9 с тремя и 1 с четырьмя сегментами.
- Все сегменты исходно записаны стратегией `3.6.0-belief-tactical`.
- Один ранее вручную просмотренный game (`121031044`) исключён из final freeze и помечен `INSUFFICIENT_INFORMATION`.

Источник: `results/fresh_dataset_inventory.json`.

## 2. Реконструкция решений

- 40 459 канонических card actions.
- 20 021 наших decision states.
- 186 325 legal actions в offline dataset.
- 1 910 extra-turn продолжений.
- Удалено 9 terminal-polling artifacts; повторные reconnect-сегменты объединены по `game_id` и `turn`.
- Для каждого state сохранены видимое состояние, рука, legal PLAY/DISCARD, публичная история, cooldown age, reconnect uncertainty и historical choice.

Артефакты: `data/all_decisions.jsonl.gz`, `data/game_index.json`.

## 3. Simulator audit 102/102

- Наблюдались все 102 карты.
- 35 973 сопоставимых фактических PLAY.
- 35 973 точных predicted deltas, 0 расхождений, точность 100%.
- 339 начальных состояний без корректной пары и 11 разрывов timeline исключены из сравнения; 118 допустимых overshoot башни нормализованы.

Артефакт: `results/simulator_actual_delta_audit_final.json`.

## 4. Baseline regret

Stage A охватил 17 203 development-state и 160 401 действие, по 500 paired worlds с common random numbers.

- Mean regret: 5.855 pp.
- Median: 0.189 pp.
- p90: 21.229 pp; p95: 29.675 pp.
- Ошибок >5 pp: 5 039; >10 pp: 3 493.
- Oracle agreement исторической policy: 47.20%.

Поздние состояния были хуже всего: mean 7.964 pp; midgame 6.402 pp; terminal race 4.863 pp.

## 5. Основные систематические ошибки

Выделены механизмы, а не бонусы конкретным картам:

1. Линейная оценка башни и недостаточная terminal-distance чувствительность.
2. Постоянная ценность production без зависимости от горизонта.
3. Слабая идентичность DISCARD: недооценка оставшихся пяти карт, ETA и finishers.
4. Perfect-response bias и огромные штрафы за просто возможный ответ.
5. Навсегда исключавшиеся observed cards и сброс belief при reconnect.
6. Ложная абсолютная уверенность из конечного набора частиц.
7. Неустойчивое ранжирование близких решений при fixed 200 particles.

## 6. Tower-race findings

В runtime/model добавлены непрерывные признаки расстояния обеих башен до 50, расстояния до разрушения, квадрат расстояния, terminal pressure, tower-race swing и взаимодействие с коротким горизонтом. Tactical prescan выполняется до общей оценки: немедленная победа, extra-turn lethal, forced-loss avoidance, затем позиционная оценка. Фиксированного правила вида «при башне N всегда строить/атаковать» нет.

## 7. Economy и learned horizon

Ridge-модель горизонта обучена только на train-game IDs. Validation MAE 14.592 card actions против 17.333 у constant-median baseline; RMSE 19.97 против 22.54. Production оценивается через прогноз горизонта и вероятность дожить до использования ресурса. ETA карты, resource reservation, short/long horizon interactions входят в state/action features.

Артефакт: `models/horizon_model.json`.

## 8. DISCARD

Для каждого слота моделируется собственный remaining-five hand и stochastic replacement draw. PLAY и DISCARD сравниваются в одних latent worlds. В candidate development mean regret выбранных discard остаётся самым слабым классом — 2.477 pp, однако это намного информативнее прежней статической retention-эвристики. Generic regression подтверждает, что discard бесполезной карты превосходит discard будущего finisher и разные discard не являются клонами.

## 9. Threat calibration

Историческая immediate-loss модель была резко overconfident: bucket 80–95% реализовывался в 26.8%, 95–100% — в 7.7%; девять state-level заявлений exactly 100% реализовались в 55.6%.

Candidate убрал exact-100 claims, но финальный calibration criterion **не пройден**:

- Freeze validation Brier: candidate 0.01755, historical 0.01480; ECE 0.01830 против 0.00642.
- Untouched holdout Brier: candidate 0.01634, historical 0.01593; ECE 0.01554 против 0.01707.

То есть ECE holdout слегка лучше, однако Brier хуже на 0.00041, поэтому считать угрозы улучшенными нельзя. Кроме того, candidate чаще выдаёт ровно 0 для редких поражений, то есть стал underconfident в хвосте. Требуется отдельный calibrator, обученный только на новом train/validation цикле; после открытия holdout делать это в 3.9 запрещено.

## 10. Аудит 100% вероятностей

В старых diagnostics было 4 card-level claims 100%, только один немедленно наблюдаемо подтвердился. В последовательном candidate-аудите 1 000 states: ложных 100% — 0; structurally proven 100% — 0. Вероятность 1.0 теперь разрешена только при структурном доказательстве, а не из-за одинаковых карт у малого числа surviving particles.

## 11. ESS и particle collapse

При 200 particles на 1 000 последовательных states:

- ESS p5 35.90, median 90.09, p95 200, min 31.01.
- Unique opponent hands p5 45, median 116, min 41.
- Opponent-hand entropy p5 3.676, median 4.619.
- Controlled rejuvenations: 399; ложных certainties: 0.

Resampling сохраняет ancestry/constraints; rejuvenation заменяет только допустимый новый шестой card slot. Это устраняет абсолютную уверенность, но ESS всё ещё требует наблюдения в новых партиях.

## 12. Opponent-policy audit

Обучен pairwise conditional-logit ranker на 14 786 opponent events и 554 883 pair rows, по 4 совместимых pseudo-hands на событие. Validation: top-1 25.83%, top-2 59.10%, mean rank 2.713, MRR 0.528, NLL 1.676. Это подтверждает, что реальный соперник не является perfect minimax агентом.

## 13. Perfect vs empirical

Oracle хранит `q_empirical`, `q_adversarial`, CVaR и robust mixture. На holdout regret относительно этого oracle: empirical-only 0.0058 pp, adversarial-only 0.0422 pp, final candidate 1.048 pp. Эти числа не являются реальным win rate: они показывают, что teacher в этой базе в основном определяется population expectation, а adversarial tail должен оставаться ограниченным risk-компонентом.

## 14. Value architecture

Используется game-balanced nonlinear logistic value с 60 state/horizon interaction features и Platt calibration. Будущее конкретной партии не является online feature. Архитектура включает terminal distances, resource race, wall against expected normal damage, initiative, reconnect uncertainty и learned horizon.

Артефакт: `models/state_value.json`.

## 15. Search depth

Online evaluator делает tactical prescan, paired current action, opponent reply и our next action; extra-turn chains разворачиваются рекурсивно до передачи хода или лимита. Offline Stage B/C использует более крупные budgets и несколько independent seeds. Полного game-theoretic solve нет: high-budget oracle является вероятностным teacher, а не доказательством оптимальности.

## 16. Performance profile

На 9 стратифицированных сложных benchmark states:

- p50 5.281 s; p90 6.651 s; p95 7.232 s; p99 7.697 s; max 7.813 s.
- Hard move timeout не нарушен; best-so-far возвращался до safety margin.
- Но все 9 случаев остановлены `deadline_precheck`, 4 остались statistically uncertain.

Старая фактическая telemetry: median около 0.8 s, p90 1.4 s, p95 1.6 s, p99 2.1 s, max 3.4 s при fixed 200. Следовательно candidate существенно медленнее.

## 17. Particle count before/after

- Production baseline: fixed 200.
- Candidate benchmark: min 40, median 120, p90/p95/max 160.
- Распределение: 40×1, 80×3, 120×2, 160×3.
- 9/9 benchmark decisions завершили меньше 200 particles.

Это material performance regression и самостоятельная причина отклонения.

## 18. Train metrics

Refined action ranker обучался по трёхseedовому 5 000-particle teacher с low weight для uncertain pairs. На 696 train states: mean regret 1.265 pp против historical 23.856 pp; p90 2.543 pp против 49.459 pp; >5 pp 44 против 437. На internal validation 104 states: 0.487 pp против 29.265 pp; p90 1.994 pp против 55.521 pp; >5 pp 2 против 71.

## 19. Freeze validation

56 game IDs, 1 530 decisions, policy была заморожена до открытия holdout.

| Метрика | Historical | Candidate |
|---|---:|---:|
| Mean regret | 5.842 pp | 1.021 pp |
| Median | 0 | 0 |
| p90 | 21.119 pp | 3.593 pp |
| p95 | 29.152 pp | 6.470 pp |
| >2 pp | 594 | 213 |
| >5 pp | 458 | 113 |
| >10 pp | 328 | 31 |
| Oracle agreement | 50.13% | 74.58% |

## 20. Random representative oracle sample

400 случайных стратифицированных Stage-B states, 5 000 particles × 3 seeds:

- Candidate mean regret 1.242 pp против historical 5.149 pp.
- p90 3.804 pp против 16.936 pp.
- >5 pp: 26 против 108.
- Oracle agreement candidate: 58.75%.

Дополнительно Stage B включал 400 suspicious states; 176/800 labels имели CI, включающий ноль. Stage C пересчитал 100 крупнейших regrets по 20 000 × 3, uncertain осталось 6, seed-ranking disagreement — 1.

## 21. Untouched holdout

51 game ID, 1 266 decisions. Holdout открыт ровно один раз после freeze; после него policy не менялась.

| Метрика | Historical | Candidate |
|---|---:|---:|
| Mean regret | 4.590 pp | 1.048 pp |
| Median | 0 | 0 |
| p90 | 16.327 pp | 3.403 pp |
| p95 | 24.667 pp | 6.216 pp |
| >2 pp | 451 | 186 |
| >5 pp | 330 | 88 |
| >10 pp | 218 | 31 |
| Oracle agreement | 51.18% | 70.22% |

Decision-regret не деградировал; threat Brier и online particle budget всё же нарушили общий acceptance.

## 22. Per-card 102/102

`results/per_card_audit_102.json` содержит по каждой карте PLAY/DISCARD обоих игроков, historical/candidate mean oracle regret, candidate preference по phase и direct-effect accuracy. Все карты покрыты. Никаких learned bonus по ID карты в runtime нет; выявленные различия используются только как диагностика механизма.

## 23. Per-action-type regret

Holdout candidate mean regret:

- extra turn 0.255 pp;
- direct tower damage 0.998 pp;
- production 1.264 pp;
- normal damage 1.284 pp;
- other/resource/wall 1.328 pp;
- discard 2.170 pp.

DISCARD остаётся главным классом для следующего цикла; extra-turn search работает лучше всего.

## 24. Historical loss rescue

Все 397 поражений аудированы без использования будущих карт как input policy:

- `RESCUABLE`: 275;
- `POSSIBLY_RESCUABLE`: 37;
- `NO_CLEAR_RESCUE`: 85.

Это counterfactual классификация по снижению largest oracle regret, а не утверждение, что 275 партий обязательно стали бы победами.

## 25. Historical win quality

Из 362 не загрязнённых побед историческая policy имела хотя бы один regret >10 pp в 300 играх, candidate — в 146. Средний largest regret снизился с 25.97 pp до 10.70 pp; средний cumulative regret — с 148.55 pp до 27.79 pp. Победы не использовались как автоматические положительные labels.

## 26. Regression tests

Полный результат: **126 passed** за 202.88 s после последней belief-поправки. Покрыты старые tactical/deck/reconnect/discard regressions и AG–AT: tower damage vs economy, safe economy, nonlinear self-damage, wall semantics, discard identity, empirical opponent, ESS collapse, false/true 100%, quiescence, extra-turn chain, online floor, ETA/horizon и initiative.

## 27. Runtime percentiles

Candidate benchmark: p50 5.281 s, p90 6.651 s, p95 7.232 s, p99 7.697 s, max 7.813 s. Feature parity runtime/offline проверена на 1 000 states и 9 510 actions: 0 ranking mismatches; max score difference `3.15e-8`.

## 28. Deadline failures

Формальных timeouts: 0; возврат best-so-far работает. Однако 9/9 benchmark states остановились по deadline precheck и ни один не достиг 200 particles. Поэтому safety правильна, но производительность неприемлема для выпуска.

## 29. Candidate version и SHA-256

Version: `3.9.0-policy-improvement-candidate`  
OOS series: `3.9.0-final-freeze-2026-08-20`

- `card_game.py`: `671ACBD868037F3BAC6F14751A938EFE8ECA72708B020C7C0D1FD888E2F2077C`
- `policy_runtime.py`: `890D66782F6DCBBCE49C2191C40324F4E0774726D16EEF09D201A000EAA39EEF`
- `policy_models.json`: `B91D3228B923FBBEDD7435345498D9CB7A06CF79E9686DF30A75D859141D1FEA`
- `cards_catalog.json`: `5C46DAC4B8F0D14CFD4AA2E58DCC9F96FE5F95A636256E8D3EA97BA0FACEF451`
- `tests/test_card_game.py`: `10CE7386AAA9AD83F22B87B72FBB0EC8058BA51068122A15FE1D30B714A3504B`
- Split fingerprint: `CA807F6433718DE56E3BD993DD30487FFB21798C4464DEFB06136AA7CB45C23B`

Существующие 3.7/3.8 и production `.exe` не перезаписаны.

## 30. Recommendation и acceptance

| Criterion | Result |
|---|---|
| Simulator / tactical / deck / reconnect / discard / tower tests | PASS |
| Mean, p90, p95, >5 pp, random sample, validation, holdout regret | PASS |
| False 100% decreases; diversity healthy | PASS |
| Hard deadline / best-so-far | PASS |
| Threat calibration improves or does not degrade | **FAIL** |
| Completed particle budget not materially worse | **FAIL** |
| No hardcoded positions / no future leakage | PASS |

Рекомендация: **не выдавать 3.9.0 как рабочую `.exe`**. Сохранить её как research candidate. Следующий цикл должен сначала оптимизировать evaluator (векторизация/кэш/двухпроходный pruning с обязательным tactical prescan), затем обучить immediate-threat calibrator на новом game-level train/validation и проверить на действительно новых партиях. Повторно использовать уже открытый holdout как untouched запрещено.

## Воспроизводимые артефакты

- `results/development_all_policy.json`
- `results/freeze_validation_policy.json`
- `results/final_holdout_policy.json`
- `results/freeze_validation_threat_calibration.json`
- `results/final_holdout_threat_calibration.json`
- `results/candidate_belief_audit.json`
- `results/benchmark_3_9.json`
- `results/per_card_audit_102.json`
- `results/per_game_audit_767.json`
- `data/LOCKED_GAME_SPLIT.json`
- `data/FINAL_FREEZE_SPLIT.json`
- `source/`
