#!/usr/bin/env bash
# ============================================================================
# Phase 2A Canary — GPU host provenance gate (RUN ON THE LINUX GPU HOST ONLY)
#
# The human-authorized launch gate for the official canary run
# (seed=123, experiment=I2). Must PASS on the Linux GPU host BEFORE
# `python scripts/run_multiseed_core.py --execute --seeds 123 --experiments I2`
#
# It does NOT start training. On any FAIL it exits non-zero and prints which
# check failed. Deliberately read-only (no file modification).
#
# Reference values (seed-42 canonical artifacts / git blob):
#   NORMALIZATION_SHA = 92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e
#   DATASET_SHA       = bb83be4616f1f3a9399f98107bbc7d7c6cd4fc5bdaf33f27fb847703241c02ea
#   SPLIT_SHA         = e46cb948ecaf303910882b26a770e3ee15765e62fcfb995a003d48696d7f4a9e
# ============================================================================
set -u

NORM_SHA="92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e"
DATASET_SHA="bb83be4616f1f3a9399f98107bbc7d7c6cd4fc5bdaf33f27fb847703241c02ea"
SPLIT_SHA="e46cb948ecaf303910882b26a770e3ee15765e62fcfb995a003d48696d7f4a9e"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FAIL: cannot cd to repo root"; exit 1; }

FAILED=0
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; FAILED=1; }

echo "Phase 2A Canary — GPU host provenance gate"
echo "repo root : $REPO_ROOT"
echo "host      : $(uname -s -r 2>/dev/null || echo unknown)"
echo ""

# --- 1. normalization sha256 (MANDATORY — hard gate) -----------------------
echo "[1/7] normalization file sha256"
if [ -f configs/normalization_v1.json ]; then
    actual="$(sha256sum configs/normalization_v1.json | awk '{print $1}')"
    echo "       expected $NORM_SHA"
    echo "       actual   $actual"
    if [ "$actual" = "$NORM_SHA" ]; then
        pass "normalization_v1.json sha256 == seed42 canonical (LF)"
    else
        fail "normalization_v1.json sha256 != seed42 canonical (likely CRLF checkout)"
    fi
else
    fail "configs/normalization_v1.json missing"
fi
echo ""

# --- 2. dataset sha256 -------------------------------------------------------
echo "[2/7] dataset sha256 (ConvLSTM_Dataset_128.h5)"
if [ -f ConvLSTM_Dataset_128.h5 ]; then
    actual="$(sha256sum ConvLSTM_Dataset_128.h5 | awk '{print $1}')"
    if [ "$actual" = "$DATASET_SHA" ]; then
        pass "dataset sha256 == seed42 canonical"
    else
        fail "dataset sha256 != seed42 canonical (expected $DATASET_SHA, got $actual)"
    fi
else
    echo "       (ConvLSTM_Dataset_128.h5 not present here — will be checked at run time by the runner's manifest)"
fi
echo ""

# --- 3. split sha256 ----------------------------------------------------------
echo "[3/7] split sha256 (configs/splits_v1.yaml)"
if [ -f configs/splits_v1.yaml ]; then
    actual="$(sha256sum configs/splits_v1.yaml | awk '{print $1}')"
    if [ "$actual" = "$SPLIT_SHA" ]; then
        pass "split sha256 == seed42 canonical"
    else
        fail "split sha256 != seed42 canonical (expected $SPLIT_SHA, got $actual)"
    fi
else
    fail "configs/splits_v1.yaml missing"
fi
echo ""

# --- 4. test seal (code-level) ------------------------------------------------
echo "[4/7] test seal code invariants"
leak=0
grep -rn --include="*.py" 'split="test"' scripts/run_multiseed_core.py && leak=1
grep -rn --include="*.py" 'split="test"' scripts/run_experiment.py && leak=1
grep -rn --include="*.py" -- '--allow-test-eval' scripts/ && leak=1
if [ "$leak" = "0" ]; then
    pass "no split=\"test\" call site; no --allow-test-eval flag"
else
    fail "test-seal code invariant violated — DO NOT START TRAINING"
fi
echo ""

# --- 5. frozen protocol status docs --------------------------------------------
echo "[5/7] frozen protocol status markers"
if grep -q "FINAL_TEST_STATUS = NOT_AUTHORIZED" docs/FINAL_TEST_AUTHORIZATION.md \
   && grep -q "TEST_STATUS = SEALED" docs/FINAL_TEST_AUTHORIZATION.md; then
    pass "TEST_STATUS = SEALED / FINAL_TEST_STATUS = NOT_AUTHORIZED"
else
    fail "test status markers not SEALED/NOT_AUTHORIZED in docs/FINAL_TEST_AUTHORIZATION.md"
fi
echo ""

# --- 6. pytest ------------------------------------------------------------------
echo "[6/7] pytest"
if python -m pytest tests/ -q >/tmp/phase2a_pytest.log 2>&1; then
    n="$(tail -1 /tmp/phase2a_pytest.log)"
    pass "pytest: $n"
else
    echo "       pytest FAILED (tail of /tmp/phase2a_pytest.log):"
    tail -20 /tmp/phase2a_pytest.log
    fail "pytest did not pass"
fi
echo ""

# --- 7. CUDA / GPU ---------------------------------------------------------------
echo "[7/7] CUDA available / GPU"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("       torch:", torch.__version__)
print("       cuda_available:", ok)
print("       device:", torch.cuda.get_device_name(0) if ok else "N/A")
if not ok:
    print("       [FAIL] CUDA NOT available")
    raise SystemExit(1)
print("       [PASS] CUDA available")
PY
if [ $? -ne 0 ]; then FAILED=1; fi
echo ""

# --- verdict -----------------------------------------------------------------------
echo "======================================================"
if [ "$FAILED" = "0" ]; then
    echo "PHASE2A_PREFLIGHT = PASS"
    echo "Next: run the canary:"
    echo "  python scripts/run_multiseed_core.py --execute --seeds 123 --experiments I2"
    echo "Then STOP. Do not continue to I3 or the full seed123."
    exit 0
else
    echo "PHASE2A_PREFLIGHT = FAIL"
    echo "STOP. Do not start training until every [FAIL] above is resolved."
    exit 1
fi
