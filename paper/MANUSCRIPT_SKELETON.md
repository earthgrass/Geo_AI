# Manuscript Skeleton — Two-Axis Controlled Ablation (skeleton only)

> **Status:** Results / Discussion / Conclusion are not written. They depend on the formal validation matrix that is currently in progress. **DO NOT** fill them until `scripts/analyze_ablation_results.py` has produced `tables/ablation_analysis/ABLATION_ANALYSIS.md`. Any contribution that depends on results carries the placeholder `TODO — WAIT FOR FORMAL VALIDATION RESULTS`.
>
> **What is NOT claimed here.** No claim that this paper "outperforms" or sets SOTA. No claim that terrain inputs *must* help on the GPM grid. No claim based on window-level significance. No multi-seed claim.

---

## Working title candidates

- A controlled two-axis ablation for 30-minute tropical-cyclone precipitation nowcasting with sealed test and event-level bootstrap CI.
- Information and inductive bias in TC nowcasting: controlled empirical evidence under a four-event held-out test.
- A leakage-safe two-axis controlled ablation framework for typhoon precipitation nowcasting.

(Choice is editorial; current working title is the first.)

## Abstract structure (≤ 220 words)

- 1 sentence on the operational problem (30-min nowcasting for landfalling TCs).
- 1 sentence on the gap (most ablation studies conflate backbone changes with information effects; auxiliary losses are reported as monolithic "physics" without isolating each component).
- 1 sentence on the design (two-axis controlled ablation under a leakage-safe protocol; backbone frozen; I5 ≡ P0 ≡ same artifact identity).
- 2–3 sentences on method (12 channels; event-disjoint split; train-only normalization; evaluator v2 with pooled contingency counts and event-level 95% bootstrap CI; final test held under a sealed, single-shot authorization).
- 2–3 sentences on findings → **TODO — WAIT FOR FORMAL VALIDATION RESULTS**.
- 1 sentence on what the contribution is (controlled empirical evidence + a reusable two-axis protocol).
- 1 sentence on the disclosed limitations (`n_event = 4` test; single seed; no `P4/P5` because wind data not yet available).

## 1. Introduction

- Operational motivation (TC precipitation at the 30-min horizon; emergency-management use of nowcasts).
- Why this matters now (with the recent rise in landfalling TCs, model evidence must be auditable).
- Why a controlled ablation rather than a "new model" framing.
- What we actually test (RQ1, RQ2, see below).
- What is *not* claimed (refer to §"Decoded status").

## 2. Research questions

### RQ1 — Information contribution.
What information improves 30-minute tropical-cyclone precipitation nowcasting: storm state, static terrain, or terrain geometry? — Adjacent Axis-I contrasts `I3 − I2`, `I4 − I3`, `I5 − I4`. Backbone held fixed.

### RQ2 — Inductive bias.
Given identical all-12-channel inputs and the same ResConvLSTM backbone, do smoothness regularization, extreme-MSE emphasis, or their combination improve validation skill over MSE alone? — `P1 − P0`, `P2 − P0`, `P3 − P0`.

## 3. Data

- Sources: GPM IMERG (precipitation), CMA Best Track (track + intensity), ETOPO1 DEM.
- Anchor-grid construction (single point in time, predictor past + target future at `+30 min`).
- 12-channel schema and fixed meanings (see `src/config.py::CHANNEL_NAMES`).
- Train-only normalization; thresholds frozen at 5/10/20/30 mm/h.

## 4. Models

- I0 Persistence (non-trainable lower bound).
- I1 PlainConvLSTM (plain recurrent baseline, MSE).
- I2 ResConvLSTM (residual-backbone control, MSE) — reuses `E2_resconvlstm_seed42`.
- I3 + storm-state (CMA channels); I4 + DEM + land mask; I5 + terrain gradients (same artifact as P0).
- B1 TrajGRU (precipitation-only), for backbone sanity.

## 5. Axis I — Input information ablation (frozen contrast: I3 − I2, I4 − I3, I5 − I4)

**VALIDATION RESULTS — single seed = 42. NOT A TEST-SET RESULT. See §11.**

- Table `axis_i_setup.csv` enumerating channel subsets.
- Paired differences per metric per event.
- Caterpillar / event-scatter figures.
- **Validation-only numerical results (single seed = 42, n_events = 7 for
  continuous, n = 3–4 for categorical@τ):**

  | Contrast | MAE Δ | RMSE Δ | SSIM Δ | MAE raw p | MAE Holm p |
  |---|---:|---:|---:|---:|---:|
  | I3 − I2 (CMA channels added) | −0.011 | −0.014 | −0.001 | 0.125 | 0.750 |
  | I4 − I3 (static terrain added) | +0.002 | −0.001 | +0.0005 | 0.453 | 1.000 |
  | I5 − I4 (terrain geometry added) | −0.002 | −0.003 | −0.0003 | 1.000 | 1.000 |

  Positive Δ on lower-is-better metrics = baseline better; positive Δ on
  higher-is-better metrics = candidate better. **No Axis I contrast is
  statistically significant at the preregistered Holm gate** on the MAE
  family. Categorical@τ = 5 mm/h shows a borderline positive Δ for
  I3 − I2 (CSI Δ = +0.020, HSS Δ = +0.024, both raw p = 0.125 with n = 4
  events) but this is descriptive-only at this seed and at this n.

- **Interpretation bound:** at this single seed and n = 7 validation
  events, the three adjacent information contrasts are *descriptively*
  neutral. I5 does not beat I4 under the current GPM 0.1° reprojection;
  the 12-channel terrain geometry does not carry signal that the
  single static-terrain channel misses at this scale.

## 6. Axis II — Loss / inductive-bias ablation (P1 − P0, P2 − P0, P3 − P0)

**VALIDATION RESULTS — single seed = 42. NOT A TEST-SET RESULT. See §11.**

- Loss components (MSE, Smooth, Extreme), lambda values frozen.
- I5 ≡ P0 same artifact identity (verifier prints the same manifest_sha256).
- Tables `axis_ii_setup.csv`, paired differences, trade-off figures.

  | Contrast | MAE Δ | RMSE Δ | SSIM Δ | MAE raw p | MAE Holm p |
  |---|---:|---:|---:|---:|---:|
  | P1 − P0 (smoothness penalty) | **+0.020** | +0.019 | **+0.0015** | **0.0156** | **0.094** |
  | P2 − P0 (extreme-rain penalty) | **−0.210** | **−0.312** | **−0.078** | **0.0156** | **0.094** |
  | P3 − P0 (smooth + extreme) | −0.080 | −0.152 | −0.024 | 1.000 | 1.000 |

  Raw p = 0.0156 on n = 7 paired events ⇒ descriptive-significant.
  Holm-adjusted p = 0.094 across the 6-contrast MAE family ⇒ NOT
  inferentially significant at the preregistered gate. **No Axis II
  contrast crosses the preregistered family-adjusted significance
  threshold at this seed.**

- **Interpretation bound:** at this single seed, the smoothness penalty
  (P1) is *directionally worse* than MSE alone on every continuous
  metric; the extreme-rain penalty (P2) is *directionally better* on
  every continuous metric. **Neither reaches Holm-adjusted
  significance**, and the paper MUST NOT claim either effect as an
  established empirical result. Multi-seed confirmation (see §11) is the
  gate for any inferential paper claim.

## 7. Evaluator v2 and statistical surface

- Pooled contingency counts (CSI, POD, FAR, HSS, BIAS, ACC) — protocol §12.
- SSIM fixed at `data_range = 100 mm/h`.
- Independent unit = typhoon event. Paired event-bootstrap 95% CI (`n_bootstrap = 10000`, `seed = 42`).
- Holm correction within `(metric, threshold)` family when family size ≥ 3 and `n_pairs ≥ 4`.
- `scripts/analyze_ablation_results.py` is the **only** analysis surface that produces paper inferences.

## 8. Results tables (column sets, not values)

| Table | Columns | Source |
|---|---|---|
| `table_baselines` | Model · MAE_global · RMSE_global · SSIM_window_mean · CSI_5mmh · CSI_10mmh · CSI_20mmh · CSI_30mmh · n_events | `experiment_summary.csv` |
| `table_axis_i` | (baseline, candidate) · n_pairs · MAE_event Δ · RMSE_event Δ · SSIM_event_mean Δ · CSI_10mmh Δ · CI95 · p_raw · p_holm | `contrasts_long.csv` + `statistical_summary.csv` |
| `table_axis_ii` | (baseline, candidate) · n_pairs · MAE_event Δ · RMSE_event Δ · SSIM_event_mean Δ · CSI_10mmh Δ · CI95 · p_raw · p_holm | `contrasts_long.csv` + `statistical_summary.csv` |
| `table_per_event_diffs` | (axis, contrast, metric, threshold, typhoon_id, diff) | `per_event_differences.csv` |

**Validation values (single seed = 42) — see `deliverables/REAL_GPU_RESULT_AUDIT.md` §E, §F, §G for the full set.** Headline:

- Event-macro MAE: I0 = 0.138, I2 = 0.131, I3 = 0.143, I4 = 0.140, I5 = 0.143,
  P1 = 0.123, P2 = 0.353, P3 = 0.223. I1 (PlainConvLSTM) and B1 (TrajGRU)
  are backbone sanity only.
- Best formal contrast on validation: P2 − I5 with MAE Δ = −0.21 mm/h
  (raw p = 0.0156, Holm-adjusted p = 0.094 across the 6-contrast MAE
  family — **not inferentially significant** at the preregistered gate).
- No contrast is Holm-adjusted p < 0.05 at this seed.

**TEST RESULTS ARE NOT REPORTED IN THIS PAPER AT THIS REVISION.
TEST_STATUS = SEALED.** The 4-event held-out test remains sealed per
`docs/FINAL_TEST_AUTHORIZATION.md` §3 and is gated by the single-shot
authorization template at `docs/FINAL_TEST_AUTHORIZATION_TEMPLATE.md`.

## 9. Discussion

**(reserved)**

- Write only after `ABLATION_ANALYSIS.md` is produced and interpreted.

## 10. Conclusion

**(reserved)**

## 11. Limitations

- `n_event = 4` test set. Statistical inference on final-test is effect-size-first, not significance-first.
- Single seed (42). Multi-seed robustness is OUT of scope; if reported later, must use the same seed set for every row.
- GPM 0.1° grid; sub-grid-scale terrain variation is smoothed away by the reprojection. This is a documented caveat; the controlled I4 − I3 and I5 − I4 contrasts inform the reader whether the smoother version still carries signal at this resolution.
- `P4`/`P5` (orographic / full-stack physics) are blocked by environmental wind data; out of scope.

## 12. Reproducibility

- Code, configs, frozen normalizations, evaluator v2, alias registry, manifest fingerprint, single one-shot test authorization template, analysis script all released.
- A `python scripts/verify_experiment_artifact.py --manifest ...` rejects incompatible artifacts.

## 13. Data + code availability

- Public GPM and CMA data; ETOPO1 DEM. Derived `ConvLSTM_Dataset_128.h5` is a frozen binary; provide either a git-lfs blob or a release-time SHA256 registry.
- Software: release-time SHA256 registry of every manifest; freeze commit provided.
