# Phase 2A Canary — Official Runbook (seed=123, I2)

> **Human authorization 2026-08-19**: only the official canary
> `seed=123 / experiment=I2` is authorized. Phase 2B (full seed123),
> Phase 2C (remaining seeds), and the final test are **NOT** authorized.
> `TEST_STATUS` must stay `SEALED`.
>
> **Execution environment**: the canary MUST run on the Linux GPU host
> (the same AutoDL Linux type as seed-42). It must NOT be launched from
> the Windows CRLF working tree (the file hashes to `f36027…` there, and
> the gate below would fail).

---

## 0. Why this gate exists

`fingerprint_file()` hashes **raw file bytes**. The committed
`configs/normalization_v1.json` is LF. On the Linux GPU host (as on
seed-42) the checkout is LF and hashes to `92a553…`. On the Windows
working tree, `core.autocrlf=true` makes it CRLF and hash to `f36027…`.
The gate below enforces the LF hash on the GPU host before training.

## 1. Stage the repo on the Linux GPU host

Three paths — pick one (human decision):

| Path | How | Requirement |
|---|---|---|
| **A. Reuse the seed-42 AutoDL instance (recommended)** | Repo + dataset already present at `/root/autodl-tmp/Geo_AI/` (LF checkout). `git pull` / switch to `research-analysis-infra` @ `112f830`. | `scripts/run_multiseed_core.py` is **untracked** — copy it (and the two new scripts in §2) onto the host, or commit it first. |
| B. Fresh git clone | `git clone <remote> -b research-analysis-infra` | Requires committing `scripts/run_multiseed_core.py` + the Phase 2A tooling first; dataset h5 must be copied to host. |
| C. Manual working-tree copy | rsync/scp the working tree to the host. | Ensure dataset h5 present; verify no CRLF got introduced by the transfer. |

After staging, confirm on the host:

```bash
git rev-parse --short HEAD        # expect 112f830
sha256sum configs/normalization_v1.json   # must be 92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e
```

## 2. Tooling on the host

Copy these three files to the host (under the repo root, if not already
present):

```text
scripts/gpu_host_phase2a_preflight.sh   # provenance gate (read-only)
scripts/verify_phase2a_canary.py        # post-run artifact audit (read-only)
scripts/run_multiseed_core.py           # the Phase 2 runner (untracked — must be present)
```

If the `.sh` file was transferred from Windows, strip CR first:

```bash
sed -i 's/\r$//' scripts/gpu_host_phase2a_preflight.sh
```

## 3. STEP 1 — provenance gate (must be PASS)

```bash
bash scripts/gpu_host_phase2a_preflight.sh
```

Checks (7): normalization sha256 == `92a553…`, dataset == `bb83be…`,
split == `e46cb948…`, no test call site / no `--allow-test-eval`,
`TEST_STATUS = SEALED` + `FINAL_TEST_STATUS = NOT_AUTHORIZED`, pytest,
CUDA available.

- Output must end with `PHASE2A_PREFLIGHT = PASS`.
- Any `[FAIL]` ⇒ **STOP. Do not start training.**

## 4. STEP 2 — official canary run

Only after the gate is PASS:

```bash
python scripts/run_multiseed_core.py --execute --seeds 123 --experiments I2
```

- This is a REAL experiment in the official 28-run matrix — **not a smoke test**.
- Full frozen configuration is used. No epochs/batch/LR/loss/channels/
  checkpoint-selection override is allowed.
- Expected output dir: `outputs/multiseed/seed_123/I2/` with
  `manifest.json`, `result_v2.json`, `metrics_v2.csv`, `validation.md`,
  `models/ResConvLSTM_best.pth`, `logs/`.
- When the run finishes, **STOP immediately**. Do **not** continue to I3
  or the full seed123.

## 5. STEP 3 — artifact audit

Run the audit on the host:

```bash
python scripts/verify_phase2a_canary.py --run-dir outputs/multiseed/seed_123/I2
```

or sync `outputs/multiseed/seed_123/I2/` back to this repo and run the
same command here. The audit checks every gate from the human
authorization (see §8 in the authorization prompt) and prints
`CANARY_ACCEPTANCE = PASS/FAIL`.

## 6. Expected values (pre-verified on 2026-08-19)

| Field | Expected |
|---|---|
| manifest.seed | `123` |
| manifest.protocol_id | `evaluation_v2` |
| manifest.test_status | `SEALED` |
| manifest.smoke | `false` |
| result.protocol_id / split / test_status | `evaluation_v2` / `val` / `SEALED` |
| result.n_events / n_windows | `7` / `1266` |
| dataset_sha256 | `bb83be4616f1f3a9399f98107bbc7d7c6cd4fc5bdaf33f27fb847703241c02ea` |
| split_sha256 | `e46cb948ecaf303910882b26a770e3ee15765e62fcfb995a003d48696d7f4a9e` |
| normalization_sha256 | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` |
| config_sha256 (seed=123, computed) | `cbf9a515feffe3995da30c6748bd5b441d2293828ec8ab4cb1a97329102c7968` |
| checkpoint_sha256 | differs from seed-42 (re-trained) — expected |
| runtime_seconds | differs from seed-42 (fresh train) — expected |
| mode | `train` (seed-42 I2 was validate-only archive reuse; per MULTISEED_PROTOCOL §4.1 the per-seed I2 is trained fresh) |

Config equivalence vs seed-42 I2: equal in `experiment, model,
batch_size, epochs, learning_rate, weight_decay,
checkpoint_selection_metric, loss_components, protocol_id,
input_channel_indices, dataset/split/normalization sha` — only `seed`,
`checkpoint_sha256`, `runtime_seconds`, `mode` differ.

## 7. After the canary

- I will produce `deliverables/MULTISEED_CANARY_SEED123_I2_AUDIT.md` from
  the audit output and STOP.
- Phase 2B (full seed123) awaits a separate human authorization.
