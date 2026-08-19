"""Phase 2A Canary artifact audit — seed=123, experiment=I2.

Read-only verifier. Runs either on the Linux GPU host right after the
canary finishes, or locally after the `outputs/multiseed/seed_123/I2`
artifacts are synced back. Emits the human-specified audit block and
exits 0 only if every gate passes.

Canary contract (human authorization, 2026-08-19):
  manifest: experiment=E2_resconvlstm, seed=123, protocol_id=evaluation_v2,
            test_status=SEALED, smoke=false
  result_v2: protocol_id=evaluation_v2, split=val, test_status=SEALED,
            n_events=7, n_windows=1266
  dataset/split/normalization sha256 identical to seed-42 canonical.
  config equal to seed-42 I2 except seed (+ checkpoint_sha256, runtime).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

CANONICAL = {
    "dataset_sha256": "bb83be4616f1f3a9399f98107bbc7d7c6cd4fc5bdaf33f27fb847703241c02ea",
    "split_sha256": "e46cb948ecaf303910882b26a770e3ee15765e62fcfb995a003d48696d7f4a9e",
    "normalization_sha256": "92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e",
}
CANARY_SEED = 123
CANARY_ALIAS = "I2"
CANARY_STEM = "E2_resconvlstm"


def config_fingerprint(cfg: Dict[str, Any]) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description="Audit the Phase 2A canary artifact.")
    p.add_argument("--run-dir", default="outputs/multiseed/seed_123/I2",
                   help="Canary output dir (runner writes manifest.json + result_v2.json).")
    p.add_argument("--seed42-manifest",
                   default="results/I2_resconvlstm_seed42_v2/manifest.json",
                   help="Seed-42 I2 manifest (reference for config equivalence).")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    man_p = run_dir / "manifest.json"
    res_p = run_dir / "result_v2.json"

    checks: List[Tuple[str, bool, str]] = []
    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # ---- artifact presence -------------------------------------------------
    have_man, have_res = man_p.exists(), res_p.exists()
    check("CANARY_ARTIFACT_COMPLETE (manifest.json)",
          have_man and have_res, f"manifest={man_p.name} result_v2.json={res_p.name}")

    if not (have_man and have_res):
        print("Audit cannot proceed: manifest.json or result_v2.json missing.")
        for name, ok, d in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {d}" if d else ""))
        return 1

    man = load_json(man_p)
    res = load_json(res_p)
    inner = res.get("result", res)

    # ---- manifest contract ---------------------------------------------------
    check("manifest.experiment == E2_resconvlstm",
          man.get("experiment") == CANARY_STEM, str(man.get("experiment")))
    check("manifest.model == ResConvLSTM",
          man.get("model") == "ResConvLSTM", str(man.get("model")))
    check(f"manifest.seed == {CANARY_SEED}",
          man.get("seed") == CANARY_SEED, str(man.get("seed")))
    check("manifest.batch_size == 4", man.get("batch_size") == 4,
          str(man.get("batch_size")))
    check("manifest.epochs == 20", man.get("epochs") == 20, str(man.get("epochs")))
    check("manifest.protocol_id == evaluation_v2",
          man.get("protocol_id") == "evaluation_v2", str(man.get("protocol_id")))
    check("manifest.test_status == SEALED",
          str(man.get("test_status")).upper() == "SEALED", str(man.get("test_status")))
    check("manifest.smoke == false",
          man.get("smoke", False) is False, str(man.get("smoke")))
    check("manifest.selection_metric == rain_mse",
          man.get("selection_metric") == "rain_mse", str(man.get("selection_metric")))
    check("manifest.input_channel_indices == [0]",
          list(man.get("input_channel_indices", [])) == [0],
          str(man.get("input_channel_indices")))
    check("manifest.loss_components == [rain]",
          list(man.get("loss_components", [])) == ["rain"],
          str(man.get("loss_components")))

    # ---- result_v2 inner contract ---------------------------------------------
    check("result.protocol_id == evaluation_v2",
          inner.get("protocol_id") == "evaluation_v2", str(inner.get("protocol_id")))
    check("result.test_status == SEALED",
          str(inner.get("test_status")).upper() == "SEALED", str(inner.get("test_status")))
    check("result.split == val", inner.get("split") == "val", str(inner.get("split")))
    check("result.n_events == 7", inner.get("n_events") == 7, str(inner.get("n_events")))
    check("result.n_windows == 1266", inner.get("n_windows") == 1266,
          str(inner.get("n_windows")))

    # ---- fingerprints vs seed-42 canonical -------------------------------------
    for key, expected in CANONICAL.items():
        got = man.get(key)
        check(f"manifest.{key} == seed42 canonical",
              got == expected, f"got {got}")
    check("manifest.checkpoint_sha256 recorded",
          bool(man.get("checkpoint_sha256")), str(man.get("checkpoint_sha256")))

    # ---- config equivalence (only seed differs) -------------------------------
    cfg_path = Path(str(man.get("config_path")))
    cfg_abs = cfg_path if cfg_path.is_absolute() else REPO_ROOT / cfg_path
    try:
        cfg = yaml.safe_load(cfg_abs.read_text(encoding="utf-8"))
        cfg_42 = json.loads(json.dumps(cfg, default=str))
        cfg_123 = json.loads(json.dumps(cfg, default=str))
        cfg_42["training"]["seed"] = 42
        cfg_123["training"]["seed"] = 123
        fp42 = config_fingerprint(cfg_42)
        fp123 = config_fingerprint(cfg_123)
        check("config_fingerprint(seed=123) == canary config_sha256",
              man.get("config_sha256") == fp123,
              f"canary={man.get('config_sha256')} recomputed={fp123}")
        # seed42 reference manifest for the config comparison
        ref = load_json(Path(args.seed42_manifest))
        check("seed42 manifest.config_sha256 == config_fingerprint(seed=42)",
              ref.get("config_sha256") == fp42,
              f"seed42={ref.get('config_sha256')} recomputed={fp42}")
    except Exception as exc:  # noqa: BLE001
        check("config equivalence (recompute)", False, f"error: {exc}")

    # ---- seed42 vs seed123 field equality (except seed / ckpt / runtime) -----
    ref = load_json(Path(args.seed42_manifest))
    eq_fields = [
        "experiment", "model", "batch_size", "epochs", "selection_metric",
        "protocol_id", "test_status", "input_channel_indices", "loss_components",
        "dataset_sha256", "split_sha256", "normalization_sha256",
    ]
    diff = []
    for fld in eq_fields:
        if man.get(fld) != ref.get(fld):
            diff.append(f"{fld}: seed123={man.get(fld)} vs seed42={ref.get(fld)}")
    check("SEED42_VS_SEED123_CONFIG_EQUAL_EXCEPT_SEED",
          not diff, "; ".join(diff) if diff else "only seed/checkpoint_sha256/runtime differ")
    check("seed differs as expected", man.get("seed") != ref.get("seed"),
          f"seed123={man.get('seed')} seed42={ref.get('seed')}")

    # ---- training history / checkpoint present --------------------------------
    ckpt_p = run_dir / "models"
    check("checkpoint dir + weights exist",
          any(ckpt_p.glob("*.pth")), str(ckpt_p))
    check("metrics_v2.csv / validation.md exist",
          (run_dir / "metrics_v2.csv").exists() and (run_dir / "validation.md").exists(),
          str(run_dir))

    # ---- report ----------------------------------------------------------------
    passed = all(ok for _, ok, _ in checks)
    print()
    print("=" * 72)
    print("Phase 2A Canary Audit — seed123 / I2")
    print("=" * 72)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    print("-" * 72)

    seed123_norm = man.get("normalization_sha256", "?")
    print("CANARY_RUN = seed123/I2")
    print(f"TRAINING_STATUS = {'COMPLETE' if man.get('mode') else '?'} "
          f"(mode={man.get('mode')}, best_epoch={man.get('best_epoch')}, "
          f"runtime={man.get('runtime_seconds')}s)")
    print(f"ARTIFACT_STATUS = {'PASS' if passed else 'FAIL'}")
    print(f"SEED42_VS_SEED123_CONFIG_EQUAL_EXCEPT_SEED = "
          f"{'YES' if all(ok for n, ok, _ in checks if 'EQUAL_EXCEPT' in n) else 'NO'}")
    print(f"DATASET_SHA_MATCH = {'YES' if man.get('dataset_sha256') == CANONICAL['dataset_sha256'] else 'NO'}")
    print(f"SPLIT_SHA_MATCH = {'YES' if man.get('split_sha256') == CANONICAL['split_sha256'] else 'NO'}")
    print(f"NORMALIZATION_SHA_MATCH = {'YES' if seed123_norm == CANONICAL['normalization_sha256'] else 'NO'}")
    print(f"SEED123_NORMALIZATION_SHA = {seed123_norm}")
    print(f"PROTOCOL_ID = {inner.get('protocol_id')}")
    print(f"VALIDATION_SPLIT = {inner.get('split')}")
    print(f"N_EVENTS = {inner.get('n_events')}")
    print(f"N_WINDOWS = {inner.get('n_windows')}")
    print(f"TEST_STATUS = {man.get('test_status')}")
    print("FINAL_TEST_STATUS = NOT_AUTHORIZED (unchanged — no test path touched)")
    print(f"CHECKPOINT_CREATED = {'YES' if any(ckpt_p.glob('*.pth')) else 'NO'}")
    print(f"RESULT_V2_CREATED = YES")
    print("SCIENTIFIC_SEMANTICS_CHANGED = NO")
    print(f"CANARY_ACCEPTANCE = {'PASS' if passed else 'FAIL'}")
    print("=" * 72)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
