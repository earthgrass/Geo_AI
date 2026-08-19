# LEAKAGE_AND_TEST_LIMIT_AUDIT — Data-leakage prevention + limited-test-set disclosure

> Companion to: `docs/RESEARCH_DESIGN_C_FREEZE.md`, `docs/EVALUATION_PROTOCOL_V2.md`, `docs/PRE_FINAL_TEST_FREEZE.md`, `docs/FINAL_TEST_AUTHORIZATION.md`.
> This file is the auditor's reference; the canonical event-level statistical analysis is `scripts/analyze_ablation_results.py`.

---

## 1. Anchor-grid causal construction (prevention of "future-position leakage")

The predictor receives **only** data with timestamps **at-or-before `anchor_time`**, the timestamp of the last input frame. The target field is the precipitation field at **`anchor_time + 30 minutes`**. Every input channel and the target use the **same** anchor-aligned grid — there is no separate "future crop." If the legacy code path that re-cropped the target around the *future observed* typhoon center had remained, the model would be cheating on typhoon-center displacement; we deliberately do not maintain that code path.

Implication: any change that introduces a second crop, a second resampling, or a target centring against the future observed track is a leakage bug and must be reverted.

## 2. Event-disjoint split

- Train (2014–2021): 25 typhoon events; 4,894 windows.
- Validation (2022): 7 typhoon events; 1,266 windows.
- Test (2023–2024): 4 typhoon events; 707 windows — **SEALED**.

Frozen split file: `configs/splits_v1.yaml`. Splits are disjoint **at the event level** (a single typhoon never crosses split boundaries). Within-event windows are temporally correlated; they are NOT independent samples. The statistical unit is the **typhoon event**, never the window.

## 3. Train-only normalization

Both source-derived inputs and the precipitation target are normalized with parameters fit **on the train events only**:

- Precipitation min/max: fit on train windows; saved to `configs/normalization_v1.json`.
- Track and terrain z-scores: fit on train windows; saved to the same file.
- Evaluation thresholds for categorical metrics: frozen at `[5, 10, 20, 30] mm/h` (see `src/evaluation/metrics.py::DEFAULT_THRESHOLDS`); not learned from any event.

Any script that calls a `fit_*` routine on val or test windows is a leakage bug and must be reverted. The training code is the only path that touches `configs/normalization_v1.json` to write normalization parameters.

## 4. 7 validation events

The validation set has **seven** typhoon events. Any paired event-level analysis therefore operates on at most seven paired observations per `(baseline, candidate)` contrast per metric. Any CI or p-value computed from fewer than four paired events is **not** reported; the analysis script `scripts/analyze_ablation_results.py` reports descriptive-only entries in that case.

## 5. 4 sealed test events

The held-out test set contains **four** typhoon events, totalling 707 windows. Per the freeze, the test set is currently SEALED. No path in the normal runner ever constructs a test `Dataset` / `DataLoader` / inference loop / metric calculation. `FINAL_TEST_STATUS = NOT_AUTHORIZED`. The only future path to a test evaluation is the explicit authorization recorded in `docs/FINAL_TEST_AUTHORIZATION.md`.

## 6. Windows are correlated; event is the statistical unit

Within a single typhoon, sliding-window samples are temporally correlated by physical continuity: a heavy-precip band evolves on the order of hours, the typhoon center drifts slowly, and adjacent 30-min windows are not IID samples. Treating windows as independent observations would inflate nominal significance by an order of magnitude or more. The analysis script `scripts/analyze_ablation_results.py` therefore pools categorical contingency counts first (per `docs/EVALUATION_PROTOCOL_V2.md` §12) and resamples at the **typhoon event** level for the bootstrap CI. Window-level significance testing is forbidden by construction (no test in `scripts/analyze_ablation_results.py` accepts per-window resampling).

## 7. Single-seed limitation

The current matrix trains each row once with seed 42. This characterizes each row's performance for the *frozen* (model-architecture, optimizer, data-loader-shuffling, GPU-deterministic) configuration; it does **not** characterize initialization robustness. Any claim of "performance improvement over a baseline across random seeds" is OUT OF SCOPE for this matrix. A multi-seed extension, if separately approved, must use the same seed set for every trainable row and must not be motivated by test results.

## 8. Independence unit summary

| What | Independent unit | Resampling |
|---|---|---|
| Pooled categorical scores (CSI/POD/FAR/HSS/BIAS/ACC) | pixel within an event | not resampled |
| Continuous scores per event (MAE_event, RMSE_event) | event | event-bootstrap (10,000 resamples, seed 42) |
| Paired differences across contrasts | paired event | event-bootstrap (10,000 resamples, seed 42); n_pairs = #events with both models defined |
| Inferential raw p-value (exact two-sided sign-flip) | paired event | n_pairs ≥ 4 required; else descriptive-only |
| Holm correction | within (metric, threshold) family | step-down |

## 9. Audit checklist (run before any final-test authorization)

- [ ] `configs/splits_v1.yaml` SHA256 matches the original frozen hash on `git log`.
- [ ] `configs/normalization_v1.json` SHA256 matches the original frozen hash.
- [ ] `configs/evaluation_thresholds_v1.json` SHA256 matches the original frozen hash.
- [ ] `git diff` against the implementation gate shows no changes in `src/data/`, `src/training/`, `src/evaluation/{evaluator,metrics}.py`, or any frozen config.
- [ ] `tests/` full run passes.
- [ ] `scripts/analyze_ablation_results.py` produces `ABLATION_ANALYSIS.md` whose `n_pairs` columns match the validation-event count.
- [ ] `results/<id>_seed<N>/manifest.json` for every frozen row reports `protocol_id=evaluation_v2`, `split=val`, `test_status=SEALED`, `smoke=false`, `n_events=7`, `n_windows=1266`.
