# Evaluation Protocol V2 — Frozen Validation/Test Specification

**Protocol ID:** `evaluation_v2`  
**Applies to:** every model in Research Design C  
**Primary forecast unit:** one 128 × 128 precipitation field at +30 minutes  
**Physical unit:** mm/h  
**Frozen categorical thresholds:** 5, 10, 20, and 30 mm/h  
**Test status:** SEALED

## 1. Notation and required output levels

Let `y_{ewp}` be observed precipitation and `f_{ewp}` the forecast for event `e`, window `w`, and pixel `p`. Predictions and observations are denormalized to mm/h before every metric. A threshold event is present when the value is **greater than or equal to** threshold `tau`.

Evaluator v2 must emit four explicitly named levels:

1. `per_window`: one row per forecast window; diagnostic metrics and raw sufficient statistics.
2. `per_event`: one row per typhoon; continuous errors pooled over all pixels/windows in that event, categorical counts pooled within the event, and mean window SSIM.
3. `overall_global`: continuous errors and categorical counts pooled across every validation pixel/window.
4. `overall_window_mean`: arithmetic mean of defined per-window continuous/structural diagnostics; never used as the primary categorical score.

It may additionally emit `overall_event_macro`, the equal-event mean of defined per-event metrics, for interpretation. This must not be mislabeled as the pooled overall score.

Every output records `protocol_id=evaluation_v2`, split name, number of events/windows/pixels, threshold list, and undefined-value counts. Bare legacy names whose aggregation cannot be inferred (`RMSE`, `CSI_10mmh`, etc.) are prohibited in machine-readable v2 output; names must encode their level or live under a level-qualified object.

## 2. MAE

For any aggregation set `S` of pixels,

`MAE(S) = (1 / |S|) sum_{i in S} |f_i - y_i|`.

- **Per window:** `MAE_window(w)` uses the 16,384 pixels in one target field.
- **Per event:** `MAE_event(e)` pools all pixels in all windows belonging to event `e`.
- **Overall primary:** `MAE_global` pools all pixels in all windows of the evaluated split.
- **Window diagnostic:** `MAE_window_mean` is the mean of `MAE_window(w)`.

Because every valid field has the same grid size, `MAE_global` equals `MAE_window_mean` up to floating-point rounding. Both names remain explicit so the aggregation rule is auditable.

## 3. RMSE

For any aggregation set `S`,

`RMSE(S) = sqrt((1 / |S|) sum_{i in S} (f_i - y_i)^2)`.

- **Per window:** compute from one target field.
- **Per event:** pool squared errors across all event pixels, then take one square root.
- **Overall primary:** pool squared errors across all split pixels, then take one square root (`RMSE_global`).
- **Window diagnostic:** `RMSE_window_mean` is the arithmetic mean of per-window RMSE values.

`RMSE_global` and `RMSE_window_mean` are not interchangeable. Evaluator v1 reported the latter under the ambiguous key `RMSE`; legacy values must be labeled accordingly.

## 4. SSIM

For each forecast window, SSIM is the spatial mean of the standard local index

`SSIM(x,y) = ((2 mu_x mu_y + C1)(2 sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))`,

with a 7 × 7 uniform window, `K1=0.01`, `K2=0.03`, `C1=(K1 R)^2`, `C2=(K2 R)^2`, and the fixed range `R=100 mm/h`. The implementation is `skimage.metrics.structural_similarity` with `data_range=100.0`, `win_size=7`, `gaussian_weights=False`, `use_sample_covariance=True`, `channel_axis=None`, `full=False`, `K1=0.01`, and `K2=0.03`. Inputs remain in mm/h and are not rescaled or clipped per sample.

- **Per window:** one SSIM value.
- **Per event:** arithmetic mean of finite window SSIM values in that event.
- **Overall:** arithmetic mean of finite window SSIM values across the split (`SSIM_window_mean`). There is no “pooled-image SSIM.”

An all-zero observation/all-zero forecast is a valid perfect structural match. Non-finite input values are an evaluator error, not silently omitted.

Evaluator v1 used `max(y.max(), f.max(), 1e-6)` separately for every window. Those legacy SSIM values are not directly comparable to v2.

## 5. Contingency counts

At threshold `tau`, define for any aggregation set:

- `a = sum 1[f >= tau and y >= tau]` — hits;
- `b = sum 1[f >= tau and y < tau]` — false alarms;
- `c = sum 1[f < tau and y >= tau]` — misses;
- `d = sum 1[f < tau and y < tau]` — correct negatives;
- `n = a + b + c + d`.

Counts are integer-valued sufficient statistics. They must be retained for every window, summed exactly for each event, and summed exactly across the full split. Metrics are computed from the pooled counts only after summation.

## 6. CSI

`CSI = a / (a + b + c)`.

The primary overall CSI pools `a,b,c` over the entire split. Per-event CSI pools counts over all windows of that event. If `a+b+c=0`, CSI is undefined (`NaN`), not zero.

## 7. POD

`POD = a / (a + c)`.

Overall and per-event rules use pooled counts. If no observed positives exist (`a+c=0`), POD is undefined (`NaN`).

## 8. FAR

`FAR = b / (a + b)`.

Here FAR means **false alarm ratio**, not false alarm rate. Overall and per-event rules use pooled counts. If no positive forecasts exist (`a+b=0`), FAR is undefined (`NaN`).

## 9. HSS

`HSS = 2(ad - bc) / ((a+c)(c+d) + (a+b)(b+d))`.

Overall and per-event HSS are computed only after pooling all four counts at the relevant level. If the denominator is zero, HSS is undefined (`NaN`).

## 10. BIAS

`BIAS = (a+b) / (a+c)`.

Overall and per-event BIAS use pooled counts. If no observed positives exist (`a+c=0`), BIAS is undefined (`NaN`). The ideal value is 1; it is not an accuracy measure.

## 11. ACC

`ACC = (a+d) / n`.

Overall and per-event ACC use pooled counts. It is defined when `n>0`, including all-dry cases. Because correct negatives dominate sparse precipitation fields, ACC is a secondary diagnostic and may not be used alone to claim forecast skill.

## 12. Overall categorical aggregation decision

**Decision: POOL CONTINGENCY COUNTS FIRST.** For each threshold, sum `a,b,c,d` across all pixels and windows in the evaluated split, then calculate CSI, POD, FAR, HSS, BIAS, and ACC once.

Mean window-level categorical ratios are not primary or secondary headline results. They weight a nearly dry window equally with an active-rain window, omit or distort undefined cases, and are not algebraically equal to the score of the full forecast set. If retained solely for debugging, they must be named `*_window_mean_diagnostic` and accompanied by the number of defined windows.

## 13. Per-event categorical aggregation decision

For each typhoon and threshold, sum `a,b,c,d` over every pixel in every window belonging to that typhoon, then calculate the six scores once. Never average window CSI/POD/FAR/HSS/BIAS/ACC to obtain an event score.

Per-event counts and denominators must appear in machine-readable output so undefined event scores are auditable. Paper-level paired comparisons use these per-event pooled scores.

## 14. NRMSE decision

**Decision: DROP FROM PRIMARY.** The evaluator-v1 definition

`RMSE_window / max(max(y_window) - min(y_window), 1e-6)`

is rejected for paper use because nearly dry windows make the denominator arbitrarily small. The E2 value `1623.65774` is a denominator artifact, not interpretable forecast error.

If a dimensionless diagnostic is desired, v2 may report only

`NRMSE_fixed100 = RMSE_global / 100 mm/h`,

using the frozen precipitation normalization scale. It must be labeled secondary, must not overwrite legacy `NRMSE`, and adds no ranking information beyond RMSE.

## 15. Peak-relative-error decision

**Decision: DROP `peak_rel_error` FROM PRIMARY AND STANDARD V2 TABLES.** Dividing by a per-window observed peak with a `1e-6` floor is unstable for dry/nearly dry windows; the legacy E2 value is not scientifically useful.

Retain only these optional diagnostics:

- `peak_abs_error_window = |max_p f - max_p y|` in mm/h;
- its split median and interquartile range;
- event-level peak error `|max_{w,p} f - max_{w,p} y|` in mm/h.

No relative peak ratio is calculated when the observed peak is zero. A future relative event-peak metric would require a separately preregistered, physically meaningful denominator and is outside v2.

## 16. Zero-rain and nearly-zero-rain handling

1. Dry windows remain in MAE, RMSE, SSIM, pooled contingency counts, and ACC; they are part of the forecast problem.
2. No metric denominator is replaced by `1`, `1e-6`, or another epsilon merely to force a finite ratio.
3. Undefined window/event ratios are `NaN` with an explicit reason and defined-count field. They do not become zero.
4. Counts from dry windows still enter higher-level pooling. For example, correct negatives enter pooled ACC and HSS; false alarms enter pooled CSI/FAR/BIAS.
5. A report must distinguish “undefined because denominator is zero” from missing/non-finite data. Missing/non-finite prediction or target arrays fail evaluation.
6. There is no exclusion of low-rain windows based on model output or target intensity.

## 17. Event-level statistical analysis

The independent analysis unit is the typhoon event. Validation comparisons have seven paired events.

For every preregistered contrast and metric:

1. Compute the metric within each event using the rules above.
2. Form paired differences in a direction where positive means improvement: for lower-is-better metrics use `baseline − candidate`; for higher-is-better metrics use `candidate − baseline`; for FAR use `baseline − candidate`; for BIAS use `|BIAS_baseline−1| − |BIAS_candidate−1|` and also report both raw BIAS values.
3. Report all event differences, the equal-event mean difference, median difference, and IQR.
4. Report a 95% event-bootstrap confidence interval for the mean difference using 10,000 paired event resamples and bootstrap seed 42. Never resample individual windows as independent cases.
5. Optionally report the two-sided exact paired sign-flip/randomization p-value over all `2^n` sign assignments for complete pairs. Label it exploratory and report `n`.
6. If fewer than four paired events have a defined metric, report descriptively only; do not calculate a p-value or confidence interval.
7. If inferential p-values are emphasized, apply Holm correction across the preregistered contrasts within each metric/threshold family. Raw and adjusted p-values must both be shown.

With only four test events, final test reporting is effect-size-first: list all four paired event differences, pooled metrics, and equal-event summaries. Do not make a strong significance claim; an exact two-sided sign-flip test has very low resolution at `n=4`.

## 18. Primary reporting set

The validation and final test tables must contain:

- `MAE_global`, `RMSE_global`, and `SSIM_window_mean`;
- pooled CSI, POD, FAR, HSS, BIAS, and ACC at 5/10/20/30 mm/h;
- all corresponding per-event values and contingency counts;
- `MAE_window_mean` and `RMSE_window_mean` as explicitly labeled diagnostics;
- optional absolute peak and displacement diagnostics only if their implementation passes v2 tests.

NRMSE, peak-relative error, mean window categorical ratios, and legacy per-category “accuracy” are excluded from the primary table.

## 19. Evaluator-v1 audit findings

- `compute_all_metrics` recursively computes each window and averages every returned ratio for batched input.
- `_aggregate` and `_aggregate_by_event` then average those window metrics again. Consequently current overall and per-event categorical metrics are means of window-level scores, not pooled-count scores.
- Zero denominators are replaced with 1.0 in categorical formulas, converting some undefined scores to zero.
- `d_correct_negatives` is not returned, preventing correct pooled HSS/ACC reconstruction from saved rows.
- Current NRMSE and peak-relative-error denominators explain the extreme E2 values.
- `scripts/run_experiment.py` asks the per-event dictionary for `CSI_10.0mmh` and `CSI_20.0mmh`; the actual keys are `CSI_10mmh` and `CSI_20mmh`. The validation Markdown `nan` values are a presentation bug, not evidence that the internal event CSI arrays were undefined.

## 20. Test-set policy

Evaluator v2 is developed, tested, and accepted using synthetic fixtures and validation only. Test IDs must be rejected by every ordinary evaluation entry point while the seal is active. No command-line “allow test” convenience flag is permitted in the validation-stage runner.

After a separately documented unseal decision, final test inference runs once for the frozen accepted checkpoints and protocol. Test data cannot select epochs, loss weights, thresholds, seeds, models, table contents, or claims. Results are written to a new immutable final-test directory with config, code commit, dataset/split/normalization hashes, checkpoint hashes, and `protocol_id=evaluation_v2`.
