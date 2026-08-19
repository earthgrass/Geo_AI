# Multi-Seed Provenance Preflight — Normalization SHA256 Discrepancy Audit

> **Phase 2 Preflight — Provenance Audit.** Generated 2026-08-19 on
> `research-analysis-infra` @ `112f830` (HEAD). **No GPU training invoked.**
> Test split untouched. `TEST_STATUS = SEALED`, `FINAL_TEST_STATUS = NOT_AUTHORIZED`.
>
> Purpose: resolve the `normalization_sha256` discrepancy between the Phase 1
> execution plan and the seed-42 canonical artifacts, and decide whether
> seed-42 and the future multi-seed runs are strictly comparable.

---

## 0. TL;DR

The two hashes both describe **the same scientific normalization file**.
They differ because one was computed over the **LF** file (git blob / Linux
seed-42 host) and the other over the **CRLF** working-tree copy on this
Windows machine (`git core.autocrlf=true`).

- Parsed JSON scientific values: **byte-for-byte identical**.
- Classification: **CASE B** (file bytes differ — line endings only; parsed
  scientific numeric values identical).
- **Not** CASE D. No scientific value has changed. No blocker at the
  scientific-semantics level.
- Precondition for Phase 2: the training host must hash the LF-normalized
  file so every manifest records `normalization_sha256 = 92a553…`
  (== seed-42). On the same Linux GPU host as seed-42 this holds by
  default; on this Windows host it does not, and LF normalization is a
  **human decision**, not performed during this audit.

---

## A. Current normalization SHA (working tree, this Windows host)

| Item | Value |
|---|---|
| File | `configs/normalization_v1.json` |
| SHA-256 (raw bytes) | `f360272e4fc5186b4f69b16332783cc28f742d2bcc166d2bcaa899aff4e3f291` |
| Byte count | 1126 |
| CRLF pairs | 51 |
| Line ending | CRLF |

This matches the value recorded in
`deliverables/MULTISEED_EXECUTION_PLAN.md` §2 and
`outputs/multiseed/_plan/matrix.json` → `frozen_configs_sha256`.

## B. Seed-42 manifest normalization SHA (canonical artifacts)

| Source | normalization_sha256 |
|---|---|
| `results/I2_resconvlstm_seed42_v2/manifest.json` | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` |
| `results/I3_resconvlstm_cma_seed42/manifest.json` | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` |
| `results/I4_static_terrain_seed42/manifest.json` | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` |
| `results/I5_terrain_geometry_seed42/manifest.json` | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` |
| `results/P1_smooth_seed42/manifest.json` | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` |
| `results/P2_extreme_seed42/manifest.json` | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` |
| `results/P3_smooth_extreme_seed42/manifest.json` | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` |

All 7 trainable seed-42 manifests agree.

## C. Seed-42-era normalization file SHA (git blob)

The committed blob is identical at **all** relevant revisions:

| Revision | SHA-256 | Bytes | Line ending |
|---|---|---|---|
| `376b14e` (file first committed, 2026-08-14) | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` | 1075 | LF |
| `1391b2d` (seed-42 `git_commit`, 2026-08-18) | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` | 1075 | LF |
| `HEAD` (`112f830`) | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` | 1075 | LF |

Computed with `git show <rev>:configs/normalization_v1.json | sha256sum`.

`git log -- configs/normalization_v1.json` shows a **single** commit
(`376b14e`) — the file's committed content has never changed. The seed-42
host hashed the LF file on disk, matching the blob exactly.

## D. Hash implementation (what does `normalization_sha256` hash?)

**Answer: A — the raw file bytes of `configs/normalization_v1.json`.**

- `src/experiments/registry.py:206-208`

  ```python
  def fingerprint_file(path: Path) -> str:
      """SHA-256 of a file's bytes."""
      return hashlib.sha256(path.read_bytes()).hexdigest()
  ```

- `scripts/run_experiment.py:353` (canonical runner):

  ```python
  "normalization_sha256": fingerprint_file(Path(norm_path)),
  ```

  where `norm_path = cfg["data"]["normalization_path"]` (line 228).
  All 10 experiment YAMLs set it to `configs/normalization_v1.json`
  (verified: 10/10 identical).

- `scripts/run_multiseed_core.py:439-441` (`run_one_row`, Phase 2 training
  path) hashes the same `cfg["data"]["normalization_path"]`.
- `scripts/run_multiseed_core.py:491-492` (plan `frozen_configs_sha256`)
  hashes `Path(NORM_PATH)` = `configs/normalization_v1.json`.

No canonical/re-serialized JSON hash exists for the normalization file.
`config_fingerprint()` (registry.py:199-203) is a *different* function —
sorted-JSON over the **experiment YAML** config, not the normalization file.
The normalization hash is always raw bytes.

## E. Parsed JSON numeric comparison

| Check | Result |
|---|---|
| `json.loads(working tree)` vs `json.loads(git blob)` canonical-dumps equal | **TRUE** |
| `precipitation.mean` / `precipitation.std` | `0.3678` / `1.8848` — identical |
| `track_features.*`, `terrain.*`, `land_mask`, `log1p_precipitation` | identical |
| Only byte difference | 51 × CR before LF (CRLF vs LF) |

```text
working file: 1126 bytes, 51 x CRLF, sha256 = f36027...
blob (LF)   : 1075 bytes,  0 x CRLF, sha256 = 92a553...
CRLF -> LF normalization of working tree => sha256 = 92a553...  (EXACT match)
```

## F. Seed-42 canonical fingerprint table (all 7 trainable)

| Experiment | seed | git_commit | dataset_sha256 | split_sha256 | normalization_sha256 | protocol_id | test_status |
|---|---|---|---|---|---|---|---|
| I2 (`E2_resconvlstm`, validate-only reuse) | 42 | `1391b2d…` | `bb83be…` | `e46cb948…` | `92a553…` | `evaluation_v2` | `SEALED` |
| I3 (`E3_resconvlstm_cma`) | 42 | `1391b2d…` | `bb83be…` | `e46cb948…` | `92a553…` | `evaluation_v2` | `SEALED` |
| I4 (`E4_static_terrain`) | 42 | `1391b2d…` | `bb83be…` | `e46cb948…` | `92a553…` | `evaluation_v2` | `SEALED` |
| I5 (`E5_terrain_geometry`, aliases `[I5,P0]`) | 42 | `1391b2d…` | `bb83be…` | `e46cb948…` | `92a553…` | `evaluation_v2` | `SEALED` |
| P1 (`P1_resconvlstm_smooth`) | 42 | `1391b2d…` | `bb83be…` | `e46cb948…` | `92a553…` | `evaluation_v2` | `SEALED` |
| P2 (`E6_terrain_extreme`) | 42 | `1391b2d…` | `bb83be…` | `e46cb948…` | `92a553…` | `evaluation_v2` | `SEALED` |
| P3 (`P3_resconvlstm_smooth_extreme`) | 42 | `1391b2d…` | `bb83be…` | `e46cb948…` | `92a553…` | `evaluation_v2` | `SEALED` |

`bb83be… = bb83be4616f1f3a9399f98107bbc7d7c6cd4fc5bdaf33f27fb847703241c02ea`
`e46cb948… = e46cb948ecaf303910882b26a770e3ee15765e62fcfb995a003d48696d7f4a9e`
`92a553…   = 92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e`

**Seed-42 internal fingerprint consistency: PASS** — `dataset_sha256`,
`split_sha256`, `normalization_sha256` identical across all 7.

## G. Current Phase-2 fingerprint table (this working tree, HEAD `112f830`)

| Fingerprint | Seed-42 canonical | Current | Match |
|---|---|---|---|
| DATASET (h5, `ConvLSTM_Dataset_128.h5`, 1 256 405 151 B) | `bb83be…` | `bb83be4616f1f3a9399f98107bbc7d7c6cd4fc5bdaf33f27fb847703241c02ea` | **YES** |
| SPLIT (`configs/splits_v1.yaml`, LF) | `e46cb948…` | `e46cb948ecaf303910882b26a770e3ee15765e62fcfb995a003d48696d7f4a9e` | **YES** |
| NORMALIZATION (raw bytes) | `92a553…` (LF) | `f360272e4fc5186b4f69b16332783cc28f742d2bcc166d2bcaa899aff4e3f291` (CRLF) | **NO — line endings only** |
| NORMALIZATION (parsed values) | identical | identical | **YES** |
| EXPERIMENT CONFIG (`config_fingerprint`, 7/7) | per-manifest | all 7 re-computed | **YES (7/7)** |
| ALIAS REGISTRY (`configs/experiment_aliases_v2.yaml`, LF) | `a1e9c469…` | `a1e9c4696d4350bf65ca79dd246e86bf75ce1d3a936a97fa4cb109b017f58b19` | **YES** |
| CATEGORICAL THRESHOLDS `[5.0,10.0,20.0,30.0]` | `src/evaluation/metrics.py:36` `DEFAULT_THRESHOLDS` | same constant (also passed verbatim by runner) | **YES** |

Frozen common controls (all verified equal to `FROZEN_COMMON` and re-checked
by the runner's `validate_seed_only_override` before every row):

```text
data.seq_len = 11              model.hidden_dims = [64,128]   model.kernel_size = 3
batch_size = 4                 epochs = 20                    lr = 1e-4
weight_decay = 1e-4            lr_patience = 10               early_stopping_patience = 10
grad_clip_norm = 1.0           use_amp = auto                 checkpoint_selection_metric = rain_mse
physics_loss.enabled = True    lambda_smooth = 0.01           lambda_extreme = 0.5
extreme_threshold = 10.0       orographic.enabled = False
input channels / loss components: read from canonical YAML, identical to seed-42
seed: 42 -> {123, 2024, 7, 31415}  ← ONLY allowed change
```

Notes:

- `configs/evaluation_thresholds_v1.json` (working tree `76c0652b…`, LF blob
  `e8f409df…`) is **not consumed by the training/evaluation pipeline** — it is
  a provenance record written by `scripts/fit_eval_thresholds.py` and hashed by
  the plan matrix only. The thresholds that actually gate evaluation are the
  hard-coded `DEFAULT_THRESHOLDS = [5.0, 10.0, 20.0, 30.0]`. Its CRLF/LF split
  therefore has **no scientific effect**; it is the same class of line-ending
  artifact as the normalization file.
- No `evaluate_model_v2(split="test")` call site exists. `assert_test_seal()`
  passes at runner import.

## H. Root cause

**CASE B** — the file bytes differ, but the parsed scientific numeric values
are **identical**.

Specific mechanism:

1. `configs/normalization_v1.json` was committed once (`376b14e`) with **LF**
   line endings (1075 bytes). The blob has never changed.
2. `fingerprint_file()` hashes **raw bytes**. The seed-42 GPU host (Linux,
   `/root/autodl-tmp/Geo_AI/`) presented the LF file → hash `92a553…`, written
   into all 7 canonical manifests.
3. This Windows working tree was checked out with `git core.autocrlf = true`
   and no `.gitattributes`, so the file on disk has **CRLF** line endings
   (1126 bytes, 51 CR). `git` reports the file clean (it normalizes CRLF→LF
   for diff), but `fingerprint_file()` does **not** normalize → hash `f36027…`.
4. The Phase 1 plan document was generated on this Windows host, so it
   recorded the CRLF hash `f36027…` — hence the apparent "drift" from seed-42.

No hash-method difference (both paths use the identical `fingerprint_file`),
and no scientific-value change. This is **not** CASE D.

## I. Scientific comparability decision

**SCIENTIFIC_VALUES_IDENTICAL = YES.**

- dataset, split, normalization numeric values, all 7 experiment configs,
  loss hyperparameters, thresholds, channels, and every frozen common control
  are identical to seed-42. Only `training.seed` changes.
- Seed-42 internal fingerprint consistency: **PASS**.
- Therefore the multi-seed training is **strictly comparable** to seed-42 at
  the scientific level.

**Artifact-fingerprint caveat (must be resolved at the training host, human
decision):** the launch precondition (§6 of the orchestration doc,
"normalization SHA = seed42") is measured at the raw-byte level. On this
Windows working tree the file hashes to `f36027…`, so a Phase-2 run executed
here would write `normalization_sha256 = f36027…` into its manifests — the
same value across all seeds/experiments (self-consistent), but different from
seed-42's `92a553…` in the `normalization_sha256` field. On the same Linux GPU
host used for seed-42, the LF checkout yields `92a553…` with no action needed.

## J. Phase-2 authorization recommendation

**Recommendation: AUTHORIZE Phase 2A Canary (seed=123, I2) — with a hard
gate on the canary manifest's `normalization_sha256`.**

| Gate | Status |
|---|---|
| Root cause resolved | CASE B, benign (line endings only) |
| Scientific semantics changed | NO |
| Seed-42 internal fingerprint consistency | PASS |
| dataset / split / config / threshold match seed-42 | YES |
| pytest | PASS (196 passed in 14.71s, CPU) |
| TEST status | SEALED / NOT_AUTHORIZED |
| Canary precondition | `normalization_sha256` in the canary `manifest.json` must equal `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` |

**Decision path for the human:**

1. If Phase 2 runs on the **Linux GPU host** (same as seed-42): the LF file
   hashes to `92a553…` automatically. Proceed to the canary; verify the canary
   manifest records `92a553…` before Phase 2B.
2. If Phase 2 must run on a **Windows host**: the working-tree normalization
   file must first be made LF (e.g. `git config core.autocrlf false` +
   re-checkout, or add `.gitattributes` with `*.json text eol=lf`, or convert
   the single file in place). This changes **no scientific value** — it only
   realigns the raw-byte fingerprint to `92a553…`. This is a human decision
   and was **not** performed during this audit.
3. Do **not** launch `--execute` while the working tree would write
   `normalization_sha256 = f36027…` (unless the human explicitly accepts a
   recorded-fingerprint delta from seed-42 for the new seeds).

This audit made no file changes and ran no GPU training.

---

## Final block

```text
NORMALIZATION_SHA_DISCREPANCY_ROOT_CAUSE = CASE B: file bytes differ by LF/CRLF line endings only; parsed scientific values identical; hash method identical (fingerprint_file = sha256 raw bytes)

SEED42_NORMALIZATION_SHA          = 92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e
SEED42_FILE_SHA                   = 92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e  (git blob @ 376b14e / 1391b2d / HEAD; LF, 1075 B)
CURRENT_NORMALIZATION_SHA         = f360272e4fc5186b4f69b16332783cc28f742d2bcc166d2bcaa899aff4e3f291  (Windows working tree; CRLF, 1126 B)
   (CRLF->LF normalized current  = 92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e)

SCIENTIFIC_VALUES_IDENTICAL       = YES

DATASET_SHA_MATCH                 = YES
SPLIT_SHA_MATCH                   = YES
NORMALIZATION_MATCH               = NO   (raw bytes: CRLF vs LF; scientific values identical = YES)
THRESHOLD_MATCH                   = YES

SEED42_INTERNAL_FINGERPRINT_CONSISTENCY = PASS

MULTISEED_COMPARABILITY           = PASS   (scientific semantics identical; raw-byte fingerprint requires LF file at training host)

SCIENTIFIC_SEMANTICS_CHANGED      = NO
TEST_STATUS                       = SEALED
FINAL_TEST_STATUS                 = NOT_AUTHORIZED

RECOMMENDATION                    = AUTHORIZE Phase 2A Canary (seed=123, I2) on a host whose
                                    normalization file hashes to 92a553... ; hard-verify the canary
                                    manifest.normalization_sha256 == 92a553... before Phase 2B.
```
