# Real-GPU Validation Audit — 2026-08-19 Single-Seed Run

> **VALIDATION MATRIX = COMPLETE (single seed = 42).**
> **MULTI-SEED = NOT YET CONFIRMED.**
> **TEST STATUS = SEALED.** This audit covers only the held-out
> validation evaluation. The 4-event held-out test split remains
> sealed per `docs/FINAL_TEST_AUTHORIZATION.md`.

| Field | Value |
|---|---|
| **Branch** | `research-analysis-infra` (off `main`) |
| **Source archive** | `Geo_AI_validation_results_ONLY_20260819.tar.gz` |
| **Source SHA256** | `2e8a33a468ddb7b7013b83eb57198f19d33822e4eebee893ac104c7690149897` ✓ |
| **Staging area** | `_local_artifacts/20260819_validation/outputs/` (excluded via `.git/info/exclude`) |
| **Canonical targets** | `results/<canonical>_seed<N>/` (10 dirs) |
| **Statistical unit** | typhoon event (per `docs/EVALUATION_PROTOCOL_V2.md` §17) |
| **Bootstrap** | n=10000, seed=42 |
| **Sign-flip** | exact two-sided; `n_pairs ≥ 4` ⇒ inferential, else descriptive-only |
| **Holm correction** | per `(metric, threshold)` family, `family_size ≥ 3`, `n_pairs ≥ 4` |
| **Window-level significance** | FORBIDDEN by construction |
| **Improvement-delta convention** | positive Δ = candidate is better than baseline (lower-is-better: `baseline − candidate`; higher-is-better: `candidate − baseline`; BIAS: `\|b−1\| − \|c−1\|`). See `src/evaluation/evaluator.py::paired_event_differences` docstring. |
| **SCIENTIFIC_SEMANTICS_CHANGED** | **NO** |

> **Revision note (R39):** Earlier drafts of this audit (and the
> MANUSCRIPT_SKELETON §5/§6/§8 prose) had the Axis II P1/P2/P3
> interpretation inverted. The analyzer code, the canonical CSVs in
> `tables/ablation_analysis/`, and the evaluator's
> `paired_event_differences` have ALWAYS obeyed
> "positive diff = improvement". Only the human-readable prose was
> wrong; the underlying tables were correct. This revision flips the
> prose to match the tables. The B1 row in §E was also brought into
> agreement with `results/B1_trajgru_seed42/result_v2.json`.

---

## A. Artifact integrity

All 11 source `manifest.json` and `result_v2.json` files were inspected
against the strict fingerprint contract. Each artifact carries:

- **Manifest fields (real GPU schema)**: `experiment`, `aliases`,
  `mode`, `model`, `seed`, `batch_size`, `epochs`, `device`,
  `amp_resolved`, `n_params`, `git_commit`, `git_dirty`, `config_path`,
  `config_sha256`, `dataset_sha256`, `split_sha256`,
  `normalization_sha256`, `checkpoint_path` (nullable), `checkpoint_sha256`
  (nullable), `selection_metric`, `best_epoch` (nullable), `best_val_mse`
  (nullable), `input_channel_indices`, `loss_components`, `protocol_id`,
  `test_status`, `smoke`, `runtime_seconds`.
- **Inner result_v2 fields (written by `write_v2_json`)**: `protocol_id`,
  `split`, `test_status`, `smoke`, `n_events`, `n_windows`, `thresholds`,
  `per_event`, `overall_global`, plus a `model` wrapper field.

Every artifact passes:

| Check | Required | Observed (all 11) |
|---|---|---|
| `manifest.protocol_id` | `evaluation_v2` | ✓ |
| `inner.protocol_id` | `evaluation_v2` | ✓ |
| `manifest.protocol_id == inner.protocol_id` | equal | ✓ |
| `manifest.test_status` | `SEALED` | ✓ |
| `inner.test_status` | `SEALED` | ✓ |
| `manifest.test_status == inner.test_status` | equal | ✓ |
| `manifest.smoke` | `False` | ✓ |
| `inner.split` | `val` | ✓ |
| `inner.n_events` | `7` | ✓ |
| `inner.n_windows` | `1266` | ✓ |
| `manifest.seed` | `42` | ✓ |
| `manifest.config_sha256` | non-empty | ✓ |
| `manifest.normalization_sha256` | non-empty | ✓ |
| `manifest.dataset_sha256` | non-empty | ✓ |
| `manifest.split_sha256` | non-empty | ✓ |
| `git_commit` (all 11) | `1391b2d2149eb2445cc0872707320817768e36cc` | ✓ |
| Wrapper `{"model": ..., "result": ...}` | required | ✓ |

**I0 (non-parametric baseline)**: `checkpoint_path` and
`checkpoint_sha256` are `null`. The contract explicitly allows this for
non-trainable baselines. No checkpoint file is required and no `.pth`
file is ever copied.

---

## B. 11 source → 10 canonical resolution

| Canonical target | Source root | Experiment(s) | Aliases |
|---|---|---|---|
| `results/I0_persistence_seed42/` | `backbone_gate/I0_persistence/` | `E0_persistence` | `I0` |
| `results/I1_plain_convlstm_seed42/` | `backbone_gate/I1_plain_convlstm/` | `E1_plain_convlstm` | `I1` |
| `results/I2_resconvlstm_seed42_v2/` | `backbone_gate/I2_resconvlstm/` + `axis_i/I2_resconvlstm/` | `E2_resconvlstm` | `I2` (×2 source dirs) |
| `results/B1_trajgru_seed42/` | `backbone_gate/B1_trajgru/` | `B1_trajgru` | *(none)* |
| `results/I3_resconvlstm_cma_seed42/` | `axis_i/I3_resconvlstm_cma/` | `E3_resconvlstm_cma` | `I3` |
| `results/I4_static_terrain_seed42/` | `axis_i/I4_static_terrain/` | `E4_static_terrain` | `I4` |
| `results/I5_terrain_geometry_seed42/` | `axis_i/I5_terrain_geometry/` | `E5_terrain_geometry` | `I5`, `P0` |
| `results/P1_smooth_seed42/` | `axis_ii_c1/P1_resconvlstm_smooth/` | `P1_resconvlstm_smooth` | `P1` |
| `results/P2_extreme_seed42/` | `axis_ii_c1/P2_terrain_extreme/` | `E6_terrain_extreme` | `P2` |
| `results/P3_smooth_extreme_seed42/` | `axis_ii_c1/P3_resconvlstm_smooth_extreme/` | `P3_resconvlstm_smooth_extreme` | `P3` |

11 source dirs were collapsed to **10 canonical experiments** by the
duplicate I2 fingerprint match (see §C). No canonical target was created
for `P0` separately — `P0` shares the I5 artifact (see §D).

The `result_v2.json` files were copied **byte-for-byte** via
`shutil.copy2`. No re-sampling occurred. SHA256 of source vs target
`result_v2.json` is identical for every canonical directory.

---

## C. I2 duplicate identity

Two source dirs both claim the `I2` alias:

- `_local_artifacts/.../backbone_gate/I2_resconvlstm/manifest.json`
- `_local_artifacts/.../axis_i/I2_resconvlstm/manifest.json`

Both manifests are **scientifically identical** on the 8-field
fingerprint tuple:

| Fingerprint field | backbone_gate/I2 | axis_i/I2 |
|---|---|---|
| `checkpoint_sha256` | `3263e135ba8312da7fd4451c261936cb4189c56a2b496d57ac62c1e968d51934` | same |
| `config_sha256` | `59a1dee37a1cd2afb1e4210b4099ccba4f41c16c9e39ffd0cd7d8df11143c222` | same |
| `dataset_sha256` | `bb83be4616f1f3a9399f98107bbc7d7c6cd4fc5bdaf33f27fb847703241c02ea` | same |
| `split_sha256` | `e46cb948ecaf303910882b26a770e3ee15765e62fcfb995a003d48696d7f4a9e` | same |
| `normalization_sha256` | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` | same |
| `git_commit` | `1391b2d2149eb2445cc0872707320817768e36cc` | same |
| `epochs` | `20` | same |
| `best_epoch` | `null` (validate-only reuse) | `null` |

The two manifests differ only in `runtime_seconds` (10.073 vs 10.109)
and `best_val_mse` formatting. **Runtime is excluded from the scientific
fingerprint** because it is wall-clock dependent and conveys no
reproducibility information.

**Resolution**: both source dirs are deduplicated into a single canonical
target `results/I2_resconvlstm_seed42_v2/`. The `ARCHIVE_MANIFEST.csv`
records `duplicate_sources = "...axis_i\I2_resconvlstm"` for transparency.

If the two manifests had disagreed on ANY fingerprint field, the archiver
would have raised `ArchiveError` and refused to collapse them. This is
covered by `test_duplicate_i2_different_fingerprint_fails`.

---

## D. I5 ≡ P0 identity

`outputs/axis_i/I5_terrain_geometry/manifest.json` declares
`aliases: ["I5", "P0"]`. This single artifact plays BOTH the role of the
final Axis I entry and the Axis II baseline.

The analyzer enforces this as follows: when both aliases appear in the
loaded alias map, their scientific fingerprints must match exactly.
Because both aliases resolve to the same manifest (no separate `P0`
artifact exists), the check is trivially satisfied.

| Alias | Experiment | checkpoint_sha256 |
|---|---|---|
| `I5` | `E5_terrain_geometry` | `95013e4dc5a529de08b96bcc824985cfd955cc3c35c3b8f8f104ee2759afa7f9` |
| `P0` | `E5_terrain_geometry` | same |

A separate `P0_*` artifact was NEVER produced by the GPU runner; the
canonical target is `I5_terrain_geometry_seed42/`, and `P0` is a logical
alias on that same artifact. Any future `P0_<other>` artifact with a
different fingerprint would cause the `enforce_i5_p0_identity` check to
FAIL FAST.

---

## E. Overall validation metric table (event-macro means)

Computed as the **equal-event mean of `per_event[<tid>][metric]`** in the
canonical `results/<exp>/result_v2.json` — that is the **only** source of
truth for paper tables. No legacy / scratch / hand-edited numbers are
admitted; this invariant is enforced by
`tests/test_canonical_consistency.py`.

| experiment_id | aliases | MAE_event | RMSE_event | SSIM_event | CSI@5 | CSI@10 | CSI@20 | CSI@30 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E0_persistence | I0 | 0.1380 | 0.5763 | 0.9775 | 0.399 | 0.300 | 0.292 | 0.165 |
| E1_plain_convlstm | I1 | 0.2376 | 0.8834 | 0.9363 | 0.000 | 0.000 | 0.000 | 0.000 |
| E2_resconvlstm | I2 | 0.1313 | 0.5036 | 0.9798 | 0.405 | 0.311 | 0.231 | 0.119 |
| B1_trajgru | *(none)* | 0.2332 | 0.8935 | 0.9380 | 0.243 | 0.146 | 0.099 | 0.018 |
| E3_resconvlstm_cma | I3 | 0.1427 | 0.5172 | 0.9789 | 0.425 | 0.330 | 0.262 | 0.100 |
| E4_static_terrain | I4 | 0.1405 | 0.5158 | 0.9794 | 0.418 | 0.324 | 0.196 | 0.094 |
| E5_terrain_geometry | I5 \| P0 | 0.1429 | 0.5188 | 0.9790 | 0.422 | 0.322 | 0.235 | 0.118 |
| P1_resconvlstm_smooth | P1 | 0.1227 | 0.4994 | 0.9806 | 0.415 | 0.316 | 0.254 | 0.121 |
| E6_terrain_extreme | P2 | 0.3532 | 0.8308 | 0.9013 | 0.359 | 0.268 | 0.300 | 0.153 |
| P3_resconvlstm_smooth_extreme | P3 | 0.2228 | 0.6708 | 0.9550 | 0.370 | 0.266 | 0.300 | 0.154 |

**Notes**:
- `I1` (plain ConvLSTM) emits zero categorical hits at every threshold;
  its continuous MAE/RMSE/SSIM are the worst of the trainable set. This
  is consistent with the documented limitation that a plain ConvLSTM
  without residual connections is unstable on this task; it is reported
  as **backbone sanity** only, never as an information effect.
- `B1_trajgru` is also backbone sanity. Categorical scores are non-zero
  but consistently the worst among trainable models. TrajGRU requires
  sequence-level flow features that this run did not supply in the same
  channel matrix; it is reported for completeness, not as a recommendation.
- `E6_terrain_extreme` (P2) is a stress-test variant. By construction
  the auxiliary extreme-rain penalty intentionally *trades continuous
  metric quality for extreme-rain detection sensitivity* (see §G.2 /
  categorical@τ tables). It is reported as an explicit trade-off, not as
  a continuous-metric improvement.

---

## F. Axis I formal contrasts (event-paired; positive Δ = improvement)

Three contrasts × (3 continuous metrics + 6 categorical metrics × 4
thresholds) = 81 rows. Selected high-signal rows below (n_pairs = 7 for
continuous, 4 for categorical@τ ≤ 10). Full table in
`tables/ablation_analysis/contrasts_long.csv`.

### F.1 I3 − I2 (add CMA storm-state channels)
| Metric@τ | mean Δ | CI95 | p (sign-flip) | Inferential |
|---|---:|---|---:|---|
| MAE_event (n=7) | −0.0114 | [−0.0212, +0.0005] | 0.125 | yes |
| RMSE_event (n=7) | −0.0136 | [−0.0342, +0.0091] | 0.453 | yes |
| SSIM_event_mean (n=7) | −0.0009 | [−0.0018, +0.0006] | 0.125 | yes |
| CSI@5 (n=4) | +0.0196 | [+0.0079, +0.0312] | 0.125 | yes |
| HSS@5 (n=4) | +0.0242 | [+0.0076, +0.0450] | 0.125 | yes |

**Reading** (improvement-delta convention: positive = I3 better):
Continuous metrics are slightly negative (I3 marginally *worse* than I2
on event-mean MAE/RMSE/SSIM, all with CI crossing zero). Categorical
metrics at τ = 5 mm/h show small positive Δ (CSI, HSS); the CI excludes
zero on the better side, indicating a possible low-threshold detection
gain.

### F.2 I4 − I3 (add static-terrain channel)
| Metric@τ | mean Δ | CI95 | p | Inferential |
|---|---:|---|---:|---|
| MAE_event | +0.0022 | [−0.0103, +0.0136] | 0.453 | yes |
| SSIM_event_mean | +0.0005 | [−0.0011, +0.0019] | 0.453 | yes |
| CSI@5 (n=4) | −0.0067 | [−0.0135, −0.0017] | 0.125 | yes |
| HSS@5 (n=4) | −0.0058 | [−0.0103, −0.0025] | 0.125 | yes |

**Reading**: Adding the static-terrain channel does NOT improve
performance vs I3 at this seed; continuous diffs cross zero on both
sides, and categorical@5 goes in the worse direction.

### F.3 I5 − I4 (add full 12-channel terrain geometry)
| Metric@τ | mean Δ | CI95 | p | Inferential |
|---|---:|---|---:|---|
| MAE_event | −0.0024 | [−0.0108, +0.0072] | 1.000 | yes |
| SSIM_event_mean | −0.0003 | [−0.0012, +0.0009] | 0.125 | yes |
| CSI@5 (n=4) | +0.0041 | [+0.0008, +0.0073] | 0.625 | yes |
| CSI@20 (n=3) | +0.0384 | [+0.0004, +0.0955] | 0.250 | **no** |

**Reading**: I5 vs I4 shows mixed signals. Continuous metrics are
indistinguishable (mean diff ≈ 0; both CIs cross zero). Categorical@5
is directionally positive (Δ CSI +0.004) but small. Categorical@20 is
positive but n=3 so descriptive-only.

---

## G. Axis II formal contrasts (improvement-delta = positive = candidate better)

### G.1 P1 − P0 (smoothness penalty on top of I5/P0)
| Metric@τ | mean Δ | CI95 | p | Inferential |
|---|---:|---|---:|---|
| MAE_event (n=7) | **+0.0202** | [+0.0142, +0.0279] | **0.0156** | yes |
| RMSE_event | +0.0193 | [+0.0058, +0.0334] | 0.125 | yes |
| SSIM_event_mean | **+0.0015** | [+0.0009, +0.0024] | **0.0156** | yes |
| CSI@5 | −0.0074 | [−0.0145, −0.0004] | 0.625 | yes |
| FAR@5 | +0.0194 | [+0.0132, +0.0285] | 0.125 | yes |

**Reading**: P1 (smoothness penalty on top of I5) **directionally
improves every continuous validation metric** at this seed. MAE_event
Δ = +0.020 mm/h, SSIM Δ = +0.0015, both with raw paired sign-flip
p = 0.0156 on n=7 events; the 95% bootstrap CI on MAE excludes zero on
the better side. The categorical@5 Δ is small and goes the other way
(CSI −0.007, FAR +0.019), so the trade-off between event-mean
continuous quality and 5 mm/h categorical specificity at this seed is
mixed. Holm-adjusted p = 0.094 across the 6-contrast MAE family (see
§J) — descriptive-significant but **not** inferentially significant at
the preregistered family gate. **Not publishable as an established
effect** at this seed; multi-seed confirmation is required before
any inductive-bias claim is made about the smoothness penalty.

### G.2 P2 − P0 (extreme-rain penalty on top of I5/P0)
| Metric@τ | mean Δ | CI95 | p | Inferential |
|---|---:|---|---:|---|
| MAE_event | **−0.2103** | [−0.4139, −0.0654] | **0.0156** | yes |
| RMSE_event | **−0.3121** | [−0.4691, −0.1695] | **0.0156** | yes |
| SSIM_event_mean | **−0.0778** | [−0.1295, −0.0347] | **0.0156** | yes |
| POD@5 | +0.1203 | [+0.0621, +0.1772] | 0.125 | yes |
| POD@10 | +0.1693 | [+0.0714, +0.2628] | 0.125 | yes |
| FAR@5 | +0.1767 | [+0.1162, +0.2469] | 0.125 | yes |
| FAR@10 | +0.3808 | [+0.2937, +0.4707] | 0.125 | yes |
| HSS@10 | −0.0577 | [−0.1018, −0.0147] | 0.125 | yes |

**Reading**: P2 **substantially worsens every continuous validation
metric** at this seed (MAE Δ = −0.21 mm/h, RMSE Δ = −0.31 mm/h, SSIM
Δ = −0.078), all three with raw p = 0.0156 (n=7) and 95% CIs excluding
zero on the worse side. **In trade-off**, P2 increases POD on
extreme-rain detection (POD@5 +0.120, POD@10 +0.169, both raw p = 0.125,
n=4) **at the cost of false alarms** (FAR@5 +0.177, FAR@10 +0.381, raw
p = 0.125, n=4). HSS@10 is negative (P2 is worse on the joint
Heidke score). This is an **explicit trade-off, not a continuous
improvement**. Holm-adjusted p = 0.094 across the 6-contrast MAE family
— descriptive-significant but **not** inferentially significant.

### G.3 P3 − P0 (smoothness + extreme combined)
| Metric@τ | mean Δ | CI95 | p | Inferential |
|---|---:|---|---:|---|
| MAE_event | −0.0799 | [−0.2224, +0.0074] | 1.000 | yes |
| RMSE_event | −0.1521 | [−0.3116, −0.0317] | 0.453 | yes |
| SSIM_event_mean | −0.0241 | [−0.0594, −0.0018] | 0.453 | yes |
| POD@5 | +0.0975 | [+0.0437, +0.1513] | 0.125 | yes |
| FAR@5 | +0.1688 | [+0.1087, +0.2319] | 0.125 | yes |
| CSI@5 | −0.0516 | [−0.1047, −0.0125] | 0.125 | yes |
| HSS@5 | −0.0528 | [−0.1080, −0.0112] | 0.125 | yes |

**Reading**: P3 has the **same trade-off direction as P2 but generally
less extreme**: continuous diffs are negative (P3 is worse on MAE/RMSE/
SSIM; CIs include zero on the worse side for MAE). POD gains are
smaller than P2 (+0.098 vs +0.120 at τ=5); FAR costs are similar. P3
shows no continuous-metric improvement over P0.

---

## H. Event-level consistency

The `per_event_differences.csv` file holds one row per `(axis, contrast,
metric, threshold, typhoon_id)` triple. **7 typhoon events** contribute
to every continuous contrast; **3–4 events** contribute to each
categorical@τ contrast (limited by the number of events with at least
one pixel ≥ τ in both baseline and candidate). The categorical@30
contrast in particular drops to n=2 (only 2 events had ≥30 mm/h pixels
on both sides); the analyzer explicitly marks `inferential=no` for those
rows.

No window is treated as a statistical unit. All 1266 windows are
collapsed into 7 event means before any paired operation.

---

## I. Bootstrap CIs

Bootstrap resamples are drawn **at the event level** with replacement
(n=10000, seed=42). The 95% CI is the equal-event percentile interval of
the resampled means of the candidate-minus-baseline differences. CIs
are reported per contrast × metric × threshold in `contrasts_long.csv`
and selected rows in §F / §G above.

A handful of contrasts have CIs that EXCLUDE zero on one side only:
these are the candidates for multi-seed confirmation. See §N.

---

## J. Sign-flip / Holm results

Exact two-sided sign-flip is the primary inferential test. Holm
correction is applied **per `(metric, threshold)` family** with
`family_size ≥ 3` and `n_pairs ≥ 4`. The full corrected table is in
`statistical_summary.csv`.

**No family reaches Holm-adjusted p < 0.05 at this seed.** The smallest
Holm-adjusted p-values are:

| Family | Contrast | p_raw | p_holm | Direction |
|---|---|---:|---:|---|
| MAE_event (continuous, n=7) | P1 − P0 | 0.0156 | 0.0938 | P1 better |
| MAE_event (continuous, n=7) | P2 − P0 | 0.0156 | 0.0938 | P2 worse |
| RMSE_event (continuous, n=7) | P2 − P0 | 0.0156 | 0.0938 | P2 worse |
| SSIM_event_mean (continuous, n=7) | P1 − P0 | 0.0156 | 0.0938 | P1 better |
| SSIM_event_mean (continuous, n=7) | P2 − P0 | 0.0156 | 0.0938 | P2 worse |

All five are at the **descriptive-significant** level (raw p < 0.05) but
**not inferentially significant** after Holm adjustment at the
`family_size = 6, n_pairs_min = 7` level. None of these can be
interpreted as a hypothesis rejection under the preregistered
`(metric, threshold)` family gate.

---

## K. Limitations

1. **Single seed (42).** Initialization robustness is NOT measured. The
   raw p < 0.05 in §G must be reconfirmed on additional seeds.
2. **n_event = 7 on validation.** Statistical power is bounded by 7
   paired observations; some categorical@τ contrasts collapse to n = 2–3
   and are descriptive-only.
3. **GPM 0.1° reprojection.** Sub-grid-scale terrain variation is
   smoothed; the I5 − I4 signal is bounded above by what the smoother
   version can carry.
4. **No held-out test.** The held-out 4 events / 707 windows remain
   sealed. No inference in this audit is a held-out test result.
5. **Categorical@τ metrics only.** We report event-level pooled
   categorical scores; we do NOT claim threshold-transfer across
   arbitrary thresholds beyond the frozen four (5, 10, 20, 30 mm/h).
6. **Single sign-flip test.** Bootstrap CIs are descriptive; the
   inferential gate is the Holm-adjusted sign-flip test.

---

## L. Interpretation boundaries

- The descriptive statements in §G are **per-seed observations**. They
  are written up because the analysis infra produced them; they MUST
  NOT be paraphrased in the paper as "P2 is robustly worse than I5 on
  validation" or "P1 is robustly better than I5 on validation". The
  paper text must read "On validation at seed 42, P1 improves MAE by
  Δ = +0.020 mm/h (raw p = 0.0156, Holm-adjusted p = 0.094 across the
  6-contrast MAE family; not inferentially significant at the
  preregistered gate)" and the corresponding P2 / P3 phrasings.
- The continuous-vs-categorical disagreement (P1 improves continuous but
  has mixed categorical@5; P2 worsens continuous but improves POD at the
  cost of FAR) is reported as-is; it is not adjudicated here.
- Any claim that "the smoothness penalty is helpful", "the extreme
  penalty is helpful", or "P2 has better categorical skill overall" is
  OUT OF SCOPE for this audit at this seed. The trade-off at τ = 5 / 10
  mm/h between POD and FAR is a categorical-only statement that does
  NOT license a paper claim.

---

## M. Test remains sealed

The held-out test split is `configs/splits_v1.yaml` (4 events, 707
windows). No `test/` path appears anywhere in `results/` or
`tables/ablation_analysis/`. `docs/FINAL_TEST_AUTHORIZATION.md` §3
remains empty. `FINAL_TEST_STATUS = NOT_AUTHORIZED`. **TEST_STATUS =
SEALED** is preserved by the archiver and the analyzer via the
`split == "val"` fail-fast contract on every `result_v2.json`.

---

## N. Recommendation for multi-seed confirmation

Before any inferential claim enters the paper, the following
descriptively-positive rows MUST be reconfirmed on additional seeds
(see `docs/MULTISEED_PROTOCOL_FREEZE.md` for the frozen multi-seed
protocol):

1. **P1 − P0 continuous metrics** (MAE_event p_raw = 0.0156, SSIM_event
   p_raw = 0.0156). Confirm the sign stays positive (P1 better on MAE
   and SSIM) across all five seeds; that the seed-averaged event-level
   Δ is positive on MAE and SSIM; that the 95% event-bootstrap CI on the
   seed-averaged Δ excludes zero; and that ≥ 4/5 seeds agree on
   direction. If all four hold, the smoothness penalty enters the paper
   as a continuous-metric improvement at seed-robust significance;
   otherwise it is reported as "directionally positive at one seed;
   multi-seed confirmation pending".
2. **P2 − P0 continuous metrics** (MAE_event p_raw = 0.0156, RMSE_event
   p_raw = 0.0156, SSIM_event p_raw = 0.0156). Confirm the sign stays
   negative (P2 worse on MAE/RMSE/SSIM) across seeds. If robust, the
   paper claims the **explicit trade-off** (continuous worse, POD
   better, FAR worse) at the same seed-robust gate.
3. **I3 − I2 categorical@5** (CSI, HSS, FAR all p_raw = 0.125 with n=4).
   This is a borderline signal at low-to-moderate rain thresholds; it
   must be confirmed with more event-paired samples before any claim
   is made. Categorical@τ signals in this band have limited power at
   n_event = 7 / τ ≥ 5 mm/h, so multi-seed confirmation requires
   preservation of the event as the primary independent unit per
   `docs/MULTISEED_PROTOCOL_FREEZE.md`.

Recommended multi-seed experiments: {42, 123, 2024, 7, 31415}. Five
seeds is the minimum to make a credible claim about initialization
robustness without overstating power. **Do NOT execute until
`docs/MULTISEED_PROTOCOL_FREEZE.md` is in place and the interpretation
bugs of this revision are verified.**

---

## Reproducibility

```bash
# Stage from a fresh clone on this branch.
pip install -r requirements.txt
pytest -q tests/

# Stage the RESULTS ONLY archive into the isolation area (SHA256 must
# match the value in §"Field"):
mkdir -p _local_artifacts/20260819_validation
tar -xzf Geo_AI_validation_results_ONLY_20260819.tar.gz \
    -C _local_artifacts/20260819_validation --strip-components=0 \
    --transform 's,Geo_AI_validation_results_ONLY_20260819/,,'

# Run the archive script (writes 10 canonical targets):
python scripts/archive_validation_results.py \
    --source-dir _local_artifacts/20260819_validation/outputs/backbone_gate \
    --source-dir _local_artifacts/20260819_validation/outputs/axis_i \
    --source-dir _local_artifacts/20260819_validation/outputs/axis_ii_c1 \
    --results-root results

# Run the analyzer:
python scripts/analyze_ablation_results.py \
    --results-dir results \
    --output-dir tables/ablation_analysis \
    --include-backbone-sanity
```

The full output of this audit is in `tables/ablation_analysis/`:
- `experiment_summary.csv` (10 canonical rows: B1 backbone sanity
  included; I2 deduped)
- `contrasts_long.csv` (one row per `axis × contrast × metric ×
  threshold`; positive `mean_diff` ⇒ candidate better)
- `per_event_differences.csv` (one row per `axis × contrast × metric ×
  threshold × typhoon_id`)
- `statistical_summary.csv` (Holm-adjusted p per `(metric, threshold)`
  family, family_size ≥ 3)
- `ABLATION_ANALYSIS.md` (human-readable summary, OBSERVATION /
  STATISTICAL SUMMARY / INTERPRETATION LIMIT / NOT YET A TEST-SET
  CONCLUSION)

The archive manifest is `results/ARCHIVE_MANIFEST.csv` (one row per
canonical experiment, including the I2 dedup annotation).

---

## Top 5 validation findings (single seed = 42)

1. **P1 (smoothness penalty) is directionally BETTER than I5 on every
   continuous validation metric** at seed 42: MAE Δ = +0.020, RMSE
   Δ = +0.019, SSIM Δ = +0.0015. MAE and SSIM both have raw paired
   sign-flip p = 0.0156 on n = 7 events; 95% bootstrap CI on MAE
   excludes zero on the better side. Holm-adjusted p = 0.094 across the
   6-contrast MAE family — descriptive-significant but **not**
   inferentially significant at the preregistered family gate.
   **Not publishable as an established effect** at this seed;
   multi-seed confirmation is required.
2. **P2 (extreme-rain penalty) substantially WORSENS continuous metrics
   at seed 42, while increasing extreme-rain POD at the cost of FAR**:
   MAE Δ = −0.21, RMSE Δ = −0.31, SSIM Δ = −0.078 (all raw p = 0.0156,
   CIs exclude zero on the worse side). Trade-off: POD@5 +0.120,
   POD@10 +0.169, but FAR@5 +0.177, FAR@10 +0.381 (raw p = 0.125, n=4).
   Holm-adjusted p = 0.094 — descriptive-significant, not
   inferentially significant. The paper may report the trade-off
   qualitatively only; an inductive-bias claim about P2 is OUT OF SCOPE
   until multi-seed confirmation.
3. **I3 (CMA) shows a low-threshold categorical gain at τ = 5 mm/h**:
   CSI Δ = +0.020, HSS Δ = +0.024, both p = 0.125 with n = 4 events. CIs
   exclude zero on the better side for CSI and HSS. Continuous metrics
   are slightly negative (I3 marginally worse than I2 on event-mean
   MAE/RMSE/SSIM).
4. **I5 (terrain geometry) is statistically indistinguishable from I4
   (static terrain)** on continuous metrics at this seed: MAE p = 1.0,
   RMSE p = 0.45, SSIM p = 0.125. The 12-channel geometry does not beat
   the single static-terrain channel under the current GPM
   reprojection.
5. **I2 (ResConvLSTM) deduplicates cleanly across `backbone_gate` and
   `axis_i` source dirs** — same `checkpoint_sha256`, same `git_commit`,
   same `dataset_sha256` etc. This validates that the validate-only
   pipeline is producing identical re-evaluations of a fixed
   checkpoint across two production code paths.

## Top 3 scientific risks

1. **Single-seed inference.** The five descriptively-significant
   findings (P1 better on continuous, P2 worse on continuous, P2 POD↑/
   FAR↑ trade-off, I3 categorical@5, I5≈I4) are all observed at seed 42
   only. At n_event = 7 on validation, the threshold for multi-seed
   confirmation is the dominant uncertainty source. Multi-seed runs are
   required before any of these enter the paper as an effect.
2. **Categorical n_pairs collapse at τ = 20 / 30.** Multiple contrasts
   lose 4–5 events because few validation events produce rain at
   τ ≥ 20 mm/h in both baseline and candidate. The I3 − I2 signal at
   τ = 20 mm/h (n=3, p=1.0) and I5 − I4 at τ = 30 mm/h (n=2, p=1.0) are
   not measurable here. Any claim that "categorical performance
   improves at extreme rain thresholds" is out of scope at this seed
   and this n_event.
3. **GPM resolution ceiling on terrain channels.** The I4 (static
   terrain) and I5 (12-channel geometry) channels are smoothed by the
   GPM 0.1° reprojection. A null result on I5 − I4 may reflect the
   smoother-resolution ceiling, not the underlying physics. A
   high-resolution regional reanalysis would be needed to disentangle
   these. This audit cannot make that call.

## Recommended multi-seed experiments

1. **Seeds: {42, 123, 2024, 7, 31415}** (5 seeds; minimum for credible
   initialization-robustness claim). Each seed re-runs the formal
   matrix (I0/I2/I3/I4/I5 + P1/P2/P3) and the I1/B1 backbone sanity.
2. **Per-seed protocol**: identical to the current run (frozen split,
   frozen normalization, frozen thresholds, frozen evaluator).
3. **Multi-seed analysis surface**: event-level pooled statistics across
   seeds; bootstrap on event differences, NOT on per-seed aggregates.
   Holm correction within the (metric, threshold, seed) family; this
   keeps the preregistered family size fixed and adds seed as a
   blocking factor. Do NOT treat `(seed, event)` cells as IID
   observations. See `docs/MULTISEED_PROTOCOL_FREEZE.md`.
4. **Stop condition for paper**: if any Axis II effect is Holm-adjusted
   p < 0.05 across the 5-seed pooled MAE / RMSE / SSIM families, the
   paper can claim the effect. Otherwise, the paper reports
   "descriptively positive at one seed; multi-seed confirmation
   pending".

---

## Bottom line

`REAL_GPU_ARTIFACT_INTEGRATION = PASS` (10 canonical from 11 sources, I2
deduplicated, I5 ≡ P0 verified, all 11 fingerprints pass contract,
zero `.pth` files copied, source `manifest.json` and `result_v2.json`
preserved byte-for-byte).

`SOURCE_ARTIFACTS_FOUND = 11`, `CANONICAL_EXPERIMENTS = 10`,
`DUPLICATE_I2_STATUS = collapsed to one canonical target`,
`I5_P0_IDENTITY = verified (single artifact, aliases=["I5","P0"])`,
`TEST_STATUS = SEALED`.

`SCIENTIFIC_SEMANTICS_CHANGED = NO`. No evaluator / config / split /
loss / threshold / channel / seed change. No held-out test evaluation.
No fabricated generalization claim.

`MULTI-SEED = NOT YET CONFIRMED`. The five seeds listed in §N are the
required next step before any descriptive finding in this audit enters
the paper as a published effect.