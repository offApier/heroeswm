# HeroesWM Worker 3.9.1 — P(win) objective audit

## Status

This build is a **candidate**, not a replacement for the existing 3.9.0
production executable.  The previously named final holdout was opened to make
the selector decision and is now treated as audit/validation data.  A new clean
OOS battle series must use `3.9.1-clean-oos-pending-2026-08-20` without further
algorithm changes.

## Oracle independence

The runtime-like 200-particle screen and the high-budget judge estimate the
same expected-Pwin target, but they do not share samples:

- runtime seed domain:
  `sha256("stage-a:" + state_id + ":runtime200")`;
- oracle seed domains:
  `fh1`, `fh2`, `fh3` (or `fv1`, `fv2`, `fv3` on freeze validation);
- every oracle seed has 5000 particles; the judge is the mean of three seeds;
- runtime estimates are not copied or reused by oracle merging;
- the evaluator receives visible state, our hand and history up to the current
  action only; it does not read future real draws or opponent cards.

Therefore the precise conclusion is: runtime P(win) approximates the
independent high-budget expected-Pwin oracle much better than the learned
action ranker does.

## Final selector

1. Exact immediate terminal win.
2. Risk-adjusted `P(win)`.
3. If the entire paired 95% CI of the difference lies inside
   `[-0.05, +0.05]` percentage point, declare practical equivalence.
4. Inside that zone: lower `P(lose next)`, lower tail risk, better tactical
   two-action horizon, then deterministic stable action order.

`policy_score` is diagnostic-only.  It never changes production selection.
Its model uncertainty is explicitly reported as not estimated; a separate MC
SE is reported only for its MC-dependent risk component.

## Epsilon validation (1530 states)

The four requested epsilon values were checked on freeze validation against an
independent 5000x3 oracle.  `0.05 pp` had the lowest mean regret.  Increasing
the zone to `0.10`, `0.20`, or `0.30 pp` progressively reduced oracle
agreement and increased regret.  The learned score did not improve the
risk/tail/tactical tie-break inside the selected zone.

Detailed output: `results/practical_equivalence_validation.json`.

## Fixed 200 versus real adaptive modes

A deterministic 24-state set fully covered by the independent 5000x3 oracle
was run through the actual production strategy and wall clock:

| policy | mean regret | p95 regret | agreement | median particles | p95 particles | max runtime | hard timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed 200 | 0.676 pp | 3.261 pp | 75.00% | 200 | 200 | 7.97 s* | 0 |
| adaptive 15 s | 0.676 pp | 3.261 pp | 75.00% | 200 | 465 | 8.44 s | 0 |
| adaptive 30 s | 0.676 pp | 3.261 pp | 75.00% | 200 | 465 | 11.87 s | 0 |
| adaptive 40 s | 0.676 pp | 3.261 pp | 75.00% | 204 | 469 | 12.73 s | 0 |

`*` Fixed-200 is an offline reference run under the 40-second room budget, not
a 15-second policy.

On this representative set additional particles did **not** change the chosen
actions, so they neither improved nor worsened oracle regret.  They reduced
uncertainty where needed and stopped early elsewhere.  This is the intended
behavior: the 30/40 modes may spend more, but are not forced to exhaust 4000 or
6000 particles after convergence.

Detailed output: `results/oracle_set_adaptive_modes_audit_v391c.json`.

## DISCARD identity and batch invariance

- Different discarded cards preserve different hands and cooldown states.
- The calibrated state-value base is retained.  When cycling is mandatory (or
  the hand has no legal play), an exact-card residual charges horizon-aware
  future option value and rewards removal of a currently harmful literal
  effect.  It contains no card-id-specific constants.
- Symmetric effects use their realized, zero-clamped resource losses; a card
  where we lose 6 and the opponent only 2 receives a negative correction,
  while the inverse state receives a positive correction.
- A regression position with six possible discards no longer produces one
  cloned P(win) value.
- Replacement draw sampling now uses a global particle index.  Thus the first
  N particles are identical across 15/30/40-second modes regardless of batch
  size.
- A paired `(0, 0)` difference stops at `practical_equivalence` instead of
  exhausting the mode cap.

## Stratification

The 1530-state validation report includes first/second mover, phase,
normal/reconnect, PLAY/DISCARD, production, direct-tower, defense and
extra-turn strata.  The weakest visible class is DISCARD (mean regret 0.188 pp,
agreement 79.75%), which remains a monitoring priority; no class has a p95
regret above 1.25 pp in this validation audit.

Detailed output: `results/pwin_selector_stratification.json`.

## Verification

The final candidate source passes all `134` automated tests.  The suite covers
literal simulation, the supplied tactical regressions, reconnect/deck state,
batch-prefix invariance, P(win)-objective stopping, exact discard identity and
the 15/30/40-second timing contract.
