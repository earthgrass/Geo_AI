# Multi-Seed Validation Protocol — Frozen

> **Status:** FROZEN at HEAD of `research-analysis-infra` (R39).
> **Effective date:** 2026-08-19.
> **Authoritative for:** any future multi-seed re-evaluation of the
> formal validation matrix (I0 / I2 / I3 / I4 / I5 + P1 / P2 / P3) and
> backbone-sanity rows (I1 / B1).

> **Hard freeze.** This document prescribes the seed set, the
> independent statistical unit, the bootstrapping unit, the Holm
> correction family, and the acceptance rule. NO multi-seed run may
> deviate from this protocol without an explicit protocol amendment
> recorded at the bottom of this file.

---

## 1. Why this protocol exists

The single-seed (42) validation in
`deliverables/REAL_GPU_RESULT_AUDIT.md` (revision R39) produced five
descriptively-significant findings (P1 better on continuous, P2 worse
on continuous with POD↑/FAR↑ trade-off, I3 categorical@5 borderline,
I5 ≈ I4, I2 dedup verified) — none of which cross the preregistered
Holm-adjusted significance gate. Multi-seed confirmation is the
required next step before any of these may enter the paper as an
established effect.

This protocol prevents three specific anti-patterns:

1. Treating `n_seed × n_event` cells as IID observations. **They are
   not.** Within-seed event differences share a fixed initialization;
   pooling them with across-seed differences as one bag of
   "35 observations" over-counts degrees of freedom and inflates the
   false-positive rate. The event is the primary independent
   statistical unit.
2. Re-defining the Holm family after seeing multi-seed results. The
   family is fixed by `docs/EVALUATION_PROTOCOL_V2.md` §17.7 and is
   `(metric, threshold)`, not `(metric, threshold, seed)`. Across-seed
   information enters as a *secondary robustness dimension*, not as
   additional family members.
3. Quietly expanding the formal contrast set. The six contrasts
   `(I3 − I2), (I4 − I3), (I5 − I4), (P1 − P0), (P2 − P0), (P3 − P0)`
   are the entire family; per-seed re-runs of these contrasts are the
   only Holm-corrected rows. Anything else (P3 − P1 − P2 + P0, I3 − I1,
   etc.) is exploratory and is **not** subject to the preregistered
   family gate.

---

## 2. Seed set

```
seeds = {42, 123, 2024, 7, 31415}
```

Five seeds. The minimum for a credible initialization-robustness
claim without overstating power. Seed 42 is the canonical
single-seed run already on disk; the other four are net new.

**Do NOT** add additional seeds after seeing results. If a sixth seed
is later deemed necessary, an explicit protocol amendment (§6) is
required.

---

## 3. Independent statistical unit

The independent statistical unit is the **typhoon event**. The unit
is fixed by `docs/EVALUATION_PROTOCOL_V2.md` §17 and has been the
single-seed analysis unit all along. Multi-seed analysis MUST preserve
this.

### 3.1 Primary multi-seed estimand

For each formal contrast and each `(metric, threshold)` pair, the
multi-seed improvement-delta for a given event is the **mean across
seeds of the per-event improvement-delta on that seed**.

```
Δ_ms(event_e, metric, τ, seed-set)
  = mean over s in seed-set of
        Δ_per_seed(event_e, metric, τ, seed=s)
```

where `Δ_per_seed(event_e, metric, τ, seed=s)` is the
event-`e`-level improvement-delta computed by
`paired_event_differences` for the run at seed `s`. With 7 events,
this yields **7 independent event-level improvement-deltas** per
contrast × metric × τ.

### 3.2 Forbidden estimand

The following is **forbidden**:

```
Δ_wrong(event_e, seed_s) treated as one of 35 iid observations
```

The `(seed × event)` cell is NOT an independent observation. Within
a single seed, all 7 event deltas share the same initialization
randomness. Across seeds, the 7 events still share the data and split
fingerprint — only the initialization differs. Treating `7 × 5 = 35`
as IID inflates significance. **Do not do this.**

### 3.3 Bootstrap resampling unit

Bootstrap resamples (paired event bootstrap, n=10000, seed 42) are
drawn at the **event level** on the **multi-seed-aggregated** event
deltas (§3.1). Windows are never resampled as independent cases. The
bootstrap seed (`BOOTSTRAP_SEED_DEFAULT = 42`) is held constant across
seeds for reproducibility; the per-event deltas vary across seeds
through the seed-mean aggregation, NOT through bootstrap RNG.

### 3.4 Sign-flip and Holm family

- Sign-flip is **exact two-sided** on the 7 multi-seed-aggregated
  event deltas. Inferential only when `n_pairs ≥ 4`.
- Holm correction is applied **per `(metric, threshold)` family**
  with `family_size ≥ 3` and `n_pairs ≥ 4`. The family is unchanged
  from the single-seed protocol — it does NOT multiply by 5 seeds.
  Seed is a **blocking factor**, not a family axis.

### 3.5 Secondary robustness descriptors (NOT significance tests)

For each contrast × metric × τ, the following secondary descriptors
are reported alongside the primary multi-seed inferential test:

- `mean across seeds of seed-level event-aggregated delta`
- `SD across seeds`
- `min / max across seeds`
- `n_seeds_with_improvement_direction_agreeing_with_primary_estimate`

These are descriptive-only. They do not produce additional
significance claims. They are reported so the reader can judge whether
a near-significant primary effect is also seed-stable.

**Optional sensitivity**: a two-way clustered bootstrap (clustered on
event, on seed) may be reported as a sensitivity analysis. It is NOT
the primary inferential test.

---

## 4. Per-seed run set

### 4.1 CORE multi-seed run set (4 new seeds)

For each new seed `s ∈ {123, 2024, 7, 31415}`, run:

```
I2   train
I3   train
I4   train
I5   train     (used as P0 for Axis II)
P1   train     (smoothness on top of I5)
P2   train     (extreme on top of I5)
P3   train     (smooth+extreme on top of I5)
```

Per seed: **7 trainable experiments**. **Do NOT rerun I0 per seed.**
I0 is deterministic and non-trainable; its evaluation at seed 42 is
the I0 estimate for every seed.

Total CORE GPU work = **4 seeds × 7 trainable experiments = 28 runs**.

### 4.2 FULL multi-seed run set (CORE + backbone sanity)

Add to CORE for each new seed `s`:

```
I1   train (backbone sanity)
B1   train (backbone sanity, TrajGRU)
```

Per seed: **9 trainable experiments**.

Total FULL GPU work = **4 seeds × 9 trainable experiments = 36 runs**.

I1 / B1 are **secondary backbone-sanity experiments** that exist for
robustness on the I1 − I2 / B1 − I2 backbone comparison only. They are
NOT part of the Axis I information axis or the Axis II inductive-bias
formal contrasts. **Do NOT spend GPU on I1/B1 across all five seeds
unless the paper intends to make a robustness claim about the backbone
comparison** (e.g. "I2 reliably beats I1 across seeds on continuous
metrics"). If the paper does not intend to make such a claim, CORE is
sufficient.

### 4.3 Decision rule

- **Default: run CORE only.** Spine of the paper.
- **Run FULL only if** the paper intends a backbone-robustness claim.
  This must be decided **before** running seeds, not after.

---

## 5. Acceptance rule (preregistered)

A formal effect (e.g. "P1 improves continuous MAE on validation") is
called **seed-robust** if and only if **all four** of the following
hold:

1. **Seed-averaged event-level effect has the expected direction.**
   The §3.1 multi-seed-aggregated event-level Δ has the predicted sign
   (improvement-delta > 0 for "candidate better", < 0 for
   "candidate worse").
2. **At least 4 of 5 seeds show the same direction.** A 5/5
   agreement is preferred.
4. **95% event-bootstrap CI excludes 0** on the §3.1 multi-seed
   event-level Δ (CI on seed-averaged event deltas).
5. **Exact event-level sign-flip + Holm correction passes** at the
   preregistered `(metric, threshold)` family gate, **if** inferential
   significance is to be claimed.

If any of 1–3 fail, the effect is reported as
**"inconsistent / uncertain / descriptive only"** for that
contrast × metric × τ.

If 1–3 hold but 4 fails (CI includes 0), the effect is reported as
**"directionally seed-stable but not inferentially significant"**.

If 1–4 hold but 5 fails (Holm-adjusted p ≥ 0.05), the effect is
reported as **"raw-significant but not family-adjusted significant"**.

If 1–5 all hold, the effect enters the paper as an established
empirical claim at seed-robust significance.

**Categorical@τ contrasts with `n_pairs < 4`** are descriptive-only by
construction (per §17.6). Multi-seed confirmation can raise the number
of paired events only via the event-mean aggregation §3.1; if after
aggregation the multi-seed-aggregated event delta is still computed on
fewer than 4 events with non-NaN values, the contrast remains
descriptive-only.

---

## 6. Protocol amendment procedure

Any change to this protocol (seed set, estimand, family definition,
acceptance rule, GPU run set, threshold set) is an amendment. To
amend:

1. Open a PR against `research-analysis-infra` modifying this file.
2. In the PR description, state what changes, why, and which result
   rows in `deliverables/REAL_GPU_RESULT_AUDIT.md` are affected.
3. Append an entry at the bottom of this document with the PR number,
   date, and a one-line summary.
4. **No multi-seed run may proceed with an amendment that has not been
   merged.**

---

## 7. Stop conditions for final-test authorization

The held-out test split remains SEALED. `FINAL_TEST_STATUS` will be
moved to `AUTHORIZED` only after **all** of:

1. The interpretation bug fixed in R39 is verified by
   `tests/test_canonical_consistency.py` passing.
2. The canonical tables in `tables/ablation_analysis/` are regenerated
   from the canonical artifacts in `results/` (R39-E).
3. This multi-seed protocol is in force (this document).
4. The CORE multi-seed run set is complete (4 new seeds × 7 trainable
   experiments = 28 runs).
5. Accepted effects from the multi-seed analysis are frozen in
   `deliverables/MULTISEED_RESULT_AUDIT.md` with the preregistered
   acceptance rule applied.
6. No further model / loss / threshold / channel selection is planned
   (selection-after-results is forbidden).

Only after items 1–6 may `docs/FINAL_TEST_AUTHORIZATION.md` §3 be
opened and a single-shot final-test evaluation be authorized.

---

## 8. Amendment log

| Date | PR | Summary |
|---|---|---|
| 2026-08-19 | — | Initial freeze (R39). |