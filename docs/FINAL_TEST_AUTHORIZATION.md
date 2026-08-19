# FINAL_TEST_AUTHORIZATION — Pre-Authorization Checklist

> **FINAL_TEST_STATUS = NOT_AUTHORIZED**
> **TEST_STATUS = SEALED**
> No code path in this repository currently permits a held-out test (2023–2024) evaluation. This file is **not** an executable test bypass. It is the only document that will ever gate a future test-set evaluation.

A test-set evaluation of any model in the frozen matrix is permitted only after **every item below** is satisfied and explicit authorization is recorded in `docs/FINAL_TEST_AUTHORIZATION.md` with append-only §"Authorizations" entries. Any failure of one item ⇒ NO TEST EVALUATION.

---

## 0. Hard preconditions

- [ ] **Validation matrix complete.** Every trainable row of `docs/RESEARCH_DESIGN_C_FREEZE.md` §3 / §4 / §5 has been trained on the GPU host, or reused from a pre-existing frozen artifact (`I2` reuses `E2_resconvlstm_seed42`). `results/README.md` lists an artifact per row.
- [ ] **Accepted experiment set frozen.** No new experiment ID has been added to `configs/experiment_aliases_v2.yaml` since the matrix was last accepted. Any addition requires a new freeze document.
- [ ] **Statistics frozen.** `scripts/analyze_ablation_results.py` produced `tables/ablation_analysis/ABLATION_ANALYSIS.md` and the four CSVs, and the statistical block in that Markdown is the only analysis surface for inferences. No new analyses may be added without re-freezing.
- [ ] **Paper table definitions frozen.** `paper/MANUSCRIPT_SKELETON.md` §"Results tables" lists the final per-axis tables; no new rows or columns may be added without re-freezing.
- [ ] **Multi-seed policy decided.** Either single-seed is declared sufficient and disclosed, or a multi-seed extension has been designed, frozen, and accepted.
- [ ] **Hashes archived.** Every `results/<id>_seed<N>/manifest.json` carries `git_commit`, `git_dirty`, `git_status_clean`, plus `config_sha256`, `dataset_sha256`, `split_sha256`, `normalization_sha256`, `checkpoint_sha256`. `python scripts/verify_experiment_artifact.py` reports `COMPATIBLE` for every accepted model.

## 1. Test-set policy (frozen in design)

Held-out test events: **`2306`**, **`2310`**, **`2402`**, **`2418`** (4 events, 707 windows). No code path in this repository currently constructs a test `Dataset` / `DataLoader` / inference loop / metric computation for these event IDs. `scripts/evaluate_checkpoint.py --split test` is refused; `src/evaluation/evaluate_models.py` is deprecated and fail-fast; `scripts/run_experiment.py` does not accept `--allow-test-eval`.

## 2. Authorization form (to be appended on each authorization)

```yaml
auth_id:               AUTH-YYYYMMDD-NN
date_submitted:        YYYY-MM-DD HH:MM
hardware:              <host>
runner_path:           <single command, one line>
metrics_requested:     [MAE, RMSE, SSIM, CSI_5mmh, CSI_10mmh, CSI_20mmh, CSI_30mmh, ...]
models_to_evaluate:    # whitelist; everything else refused
  - id: <experiment_id>
    checkpoint_sha256:        <SHA256>
    config_sha256:            <SHA256>
    manifest_path:            results/<id>_seed<N>/manifest.json
outputs:
  results_dir: results/final_test_<auth_id>/
  paper_table_csv: paper/tables/final_test_<auth_id>.csv
approvals_required:
  - role: Research Lead          sign: ___   date: ___
  - role: Independent Reviewer   sign: ___   date: ___
```

After execution, append to §3 here: outcome (PASS / ABORTED), per-model summary table, fingerprint reconciliation, post-execution reseal confirmation.

## 3. Authorizations (append-only)

_None. Any future entry must include the authorization form, fingerprint verification, post-execution reseal confirmation, and a per-model summary table._

---

> **This file is the *only* gate.** Any code change that admits test data into a normal runner before §0 is satisfied and §3 records an authorization is a violation of the research-design freeze and must be reverted.
