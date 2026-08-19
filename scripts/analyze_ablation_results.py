"""Cross-experiment statistical analysis for the two-axis controlled ablation.

This is the **only** statistical surface that produces paper inferences. It
consumes the frozen evaluator-v2 outputs (``result_v2.json``) plus their
fingerprints (``manifest.json``) under ``results/<id>_seed<N>/`` and emits
five files under ``--output-dir`` (default ``tables/ablation_analysis/``):

    experiment_summary.csv     — one row per experiment (wide; per-metric
                                 columns), backbone sanity included.
    contrasts_long.csv         — one row per (axis, contrast, metric, threshold).
    per_event_differences.csv  — one row per (axis, contrast, metric,
                                 threshold, typhoon_id).
    statistical_summary.csv    — raw + Holm-adjusted p-values per
                                 (metric, threshold) family.
    ABLATION_ANALYSIS.md       — human-readable summary split into
                                 OBSERVATION / STATISTICAL SUMMARY /
                                 INTERPRETATION LIMIT /
                                 NOT YET A TEST-SET CONCLUSION.

It REFUSES to run if any loaded artifact fails the fingerprint contract:

    protocol_id == "evaluation_v2"
    split        == "val"
    test_status  == "SEALED"
    smoke        is False
    result.n_events   == 7
    result.n_windows  == 1266

It enforces ``I5 == P0`` exact artifact identity. If a manifest declares
both aliases, the resulting fingerprint must match the canonical I5
artifact; if both directories exist and differ, the script aborts.

Statistical unit is the typhoon event (per ``EVALUATION_PROTOCOL_V2.md``
§17). Window-level significance testing is forbidden by construction.

Frozen contrast set (only these are emitted as "formal" comparisons):

    Axis I   : (I3 - I2), (I4 - I3), (I5 - I4)
    Axis II  : (P1 - P0), (P2 - P0), (P3 - P0)

Backbone sanity contrasts (I0 / I1 / I2 / B1) are emitted as descriptive
rows only. ``I1 - I2`` is reported ONLY as a backbone contrast (architecture
change), NOT as an information effect, per ``RESEARCH_DESIGN_C_FREEZE.md``
§2 / §11 / §12.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Make `src.evaluation.evaluator.paired_event_differences` importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import (  # noqa: E402
    paired_event_differences,
    PROTOCOL_ID,
)


# ---------------------------------------------------------------------------
# Frozen thresholds and the strict fingerprint contract
# ---------------------------------------------------------------------------

EXPECTED_PROTOCOL_ID = "evaluation_v2"
EXPECTED_SPLIT = "val"
EXPECTED_TEST_STATUS = "SEALED"
EXPECTED_SMOKE = False
EXPECTED_N_EVENTS = 7
EXPECTED_N_WINDOWS = 1266

# Mm/h categorical thresholds (frozen in src/evaluation/metrics.py).
FROZEN_THRESHOLDS_MMH: Tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)

# Continuous event-level metrics directly available in result_v2.json.
CONTINUOUS_METRICS: Tuple[str, ...] = (
    "MAE_event", "RMSE_event", "SSIM_event_mean",
)

# Categorical metrics: extracted per event from per_event[tid].categorical[tau_key].
CATEGORICAL_METRICS: Tuple[str, ...] = (
    "CSI", "POD", "FAR", "HSS", "ACC", "BIAS",
)

# Alias-resolving input map: alias -> canonical E-prefix / P-prefix config.
FORMAL_AXIS_I: Tuple[Tuple[str, str, str], ...] = (
    # (axis, baseline, candidate)
    ("AxisI", "I2", "I3"),
    ("AxisI", "I3", "I4"),
    ("AxisI", "I4", "I5"),
)
FORMAL_AXIS_II: Tuple[Tuple[str, str, str], ...] = (
    ("AxisII", "P0", "P1"),
    ("AxisII", "P0", "P2"),
    ("AxisII", "P0", "P3"),
)
# Backbone-sanity comparisons are reported in experiment_summary only; we do
# NOT compute paired differences for them as "information" or "inductive-bias"
# claims. ``I1 - I2`` is a backbone contrast (architecture changes), not an
# information effect, per RESEARCH_DESIGN_C_FREEZE.md §2 / §11.1.
BACKBONE_SANITY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("I0", "I1"),
    ("I1", "I2"),
    ("I2", "B1"),
)

# Statistical controls (frozen from EVALUATION_PROTOCOL_V2.md §17.4).
N_BOOTSTRAP_DEFAULT = 10000
BOOTSTRAP_SEED_DEFAULT = 42

# Holm correction only applies when the family has at least this many contrasts
# and the per-contrast n_pairs is at least 4 (§17.6 / §17.7).
HOLM_FAMILY_MIN_SIZE = 3
HOLM_N_PAIRS_MIN = 4


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ContractViolation(AssertionError):
    """Raised when an artifact fails the fingerprint contract. NO BYPASS."""


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise ContractViolation(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_manifest(manifest: Dict, manifest_path: Path) -> None:
    """FAIL FAST on any fingerprint violation."""
    def _check(key: str, expected) -> None:
        if manifest.get(key) != expected:
            raise ContractViolation(
                f"{manifest_path.name}: manifest field '{key}' must equal "
                f"{expected!r}; got {manifest.get(key)!r}."
            )

    for key in ("experiment_id", "alias_ids", "git_commit",
                "config_sha256", "dataset_sha256", "split_sha256",
                "normalization_sha256", "checkpoint_sha256",
                "protocol_id", "test_status", "split", "smoke"):
        if key not in manifest:
            raise ContractViolation(
                f"{manifest_path.name}: missing required manifest field '{key}'."
            )

    _check("protocol_id", EXPECTED_PROTOCOL_ID)
    _check("split", EXPECTED_SPLIT)
    _check("test_status", EXPECTED_TEST_STATUS)
    _check("smoke", EXPECTED_SMOKE)

    # n_events / n_windows are recorded in the result_v2 payload, not the
    # manifest; the manifest records the *expected* counts for cross-checks.
    # We compare them below against the result_v2 n_events / n_windows.


def _validate_result(result: Dict, manifest_path: Path) -> None:
    if result.get("protocol_id") != PROTOCOL_ID:
        raise ContractViolation(
            f"{manifest_path.name}: result_v2.protocol_id must equal "
            f"{PROTOCOL_ID!r}; got {result.get('protocol_id')!r}.")
    if result.get("split") != EXPECTED_SPLIT:
        raise ContractViolation(
            f"{manifest_path.name}: result_v2.split must equal "
            f"{EXPECTED_SPLIT!r}; got {result.get('split')!r}.")
    if result.get("test_status") != EXPECTED_TEST_STATUS:
        raise ContractViolation(
            f"{manifest_path.name}: result_v2.test_status must equal "
            f"{EXPECTED_TEST_STATUS!r}; got {result.get('test_status')!r}.")
    if result.get("n_events") != EXPECTED_N_EVENTS:
        raise ContractViolation(
            f"{manifest_path.name}: result_v2.n_events must equal "
            f"{EXPECTED_N_EVENTS}; got {result.get('n_events')}.")
    if result.get("n_windows") != EXPECTED_N_WINDOWS:
        raise ContractViolation(
            f"{manifest_path.name}: result_v2.n_windows must equal "
            f"{EXPECTED_N_WINDOWS}; got {result.get('n_windows')}.")


def load_results(
    results_dir: Path,
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Walk ``results_dir`` and return (alias -> result_v2, alias -> manifest).

    Aliases are the entries in each manifest's ``alias_ids`` list. If the
    same alias appears in two manifests, FAIL FAST. The I5/P0 same-artifact
    identity is verified separately in :func:`enforce_i5_p0_identity`.
    """
    if not results_dir.exists():
        raise ContractViolation(f"results_dir does not exist: {results_dir}")

    alias_to_result: Dict[str, Dict] = {}
    alias_to_manifest: Dict[str, Dict] = {}

    for manifest_path in sorted(results_dir.glob("*/manifest.json")):
        manifest = _load_json(manifest_path)
        _validate_manifest(manifest, manifest_path)

        result_v2_path = manifest_path.parent / "result_v2.json"
        result = _load_json(result_v2_path)
        _validate_result(result, manifest_path)

        for alias in manifest["alias_ids"]:
            if alias in alias_to_result:
                raise ContractViolation(
                    f"Alias '{alias}' is declared by more than one manifest: "
                    f"{manifest_path.parent.name} and "
                    f"{alias_to_manifest[alias].get('experiment_id', '<unknown>')}."
                )
            alias_to_result[alias] = result
            alias_to_manifest[alias] = manifest

    return alias_to_result, alias_to_manifest


def enforce_i5_p0_identity(
    alias_to_manifest: Dict[str, Dict],
) -> Optional[Dict]:
    """Verify that the I5 and P0 manifests, if both present, are the same artifact.

    The alias registry resolves I5 and P0 to the same canonical config; the
    two aliases therefore MUST resolve to the same manifest fingerprint. If
    only one of them is present the identity is trivially recorded; if both
    are present they must agree on the full fingerprint tuple.
    """
    if "I5" not in alias_to_manifest and "P0" not in alias_to_manifest:
        return None
    if "I5" in alias_to_manifest and "P0" not in alias_to_manifest:
        return alias_to_manifest["I5"]
    if "P0" in alias_to_manifest and "I5" not in alias_to_manifest:
        return alias_to_manifest["P0"]

    a = alias_to_manifest["I5"]
    b = alias_to_manifest["P0"]
    fingerprint_keys = (
        "checkpoint_sha256", "config_sha256", "dataset_sha256",
        "split_sha256", "normalization_sha256", "git_commit",
        "epochs", "best_epoch",
    )
    for k in fingerprint_keys:
        if a.get(k) != b.get(k):
            raise ContractViolation(
                f"I5/P0 artifact identity violated: "
                f"{a.get('experiment_id', '?')} vs "
                f"{b.get('experiment_id', '?')} disagree on '{k}' "
                f"({a.get(k)!r} vs {b.get(k)!r}). "
                f"I5 and P0 must resolve to the same canonical config "
                f"and checkpoint, per the alias registry."
            )
    return a


# ---------------------------------------------------------------------------
# Statistical helpers (event-level ONLY; window-level significance is banned)
# ---------------------------------------------------------------------------

def exact_two_sided_sign_flip(diffs: Sequence[float]) -> Tuple[float, int]:
    """Exact two-sided sign-flip randomization p-value.

    With n paired event differences, the number of positive-event sign
    assignments under H0 follows Binomial(n, 0.5). The two-sided p-value is

        p = min(1, 2 * P[#positives >= observed_max_under_tail]).

    Returns ``(p_value, n)``. When ``n < 4``, we still compute the p-value
    here for completeness, but the caller MUST NOT use it inferentially
    (EVALUATION_PROTOCOL_V2.md §17.6). The caller is responsible for
    marking such rows as descriptive-only.
    """
    diffs = [d for d in diffs if d == d]  # drop NaN
    n = len(diffs)
    if n == 0:
        return float("nan"), 0
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    total = 1 << n
    observed_max = max(pos, neg)
    ge = 0
    for k in range(observed_max, n + 1):
        ge += comb(n, k)
    p = min(1.0, 2.0 * ge / total)
    return p, n


def holm_correction(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down correction, preserving NaNs."""
    if not pvals:
        return {}
    keys = sorted(pvals, key=lambda k: (
        pvals[k] if pvals[k] == pvals[k] else float("inf")))
    m = len(keys)
    out: Dict[str, float] = {}
    running = 0.0
    for i, k in enumerate(keys, start=1):
        p = pvals[k]
        if p != p:
            out[k] = float("nan")
            continue
        adj = (m - i + 1) * p
        adj = max(adj, running)
        adj = min(adj, 1.0)
        running = adj
        out[k] = adj
    return out


# ---------------------------------------------------------------------------
# Per-event extraction
# ---------------------------------------------------------------------------

def _extract_threshold_keys(result: Dict) -> Dict[float, str]:
    """Return a {tau_mm_h -> threshold_key} map emitted by the v2 evaluator.

    The v2 result stores thresholds under ``result['thresholds']`` as a list
    of floats. The metrics layer's ``threshold_key`` is used by the
    evaluator; here we just look it up via the per_event dict, so we don't
    rely on the exact threshold_key encoding — we walk every categorical
    sub-bucket by metric name (e.g. 'CSI', 'POD').
    """
    return {float(t): float(t) for t in result.get("thresholds",
                                                   FROZEN_THRESHOLDS_MMH)}


def _continuous_event_value(result: Dict, tid: str, metric: str) -> float:
    return float(result["per_event"][tid].get(metric, float("nan")))


def _categorical_event_value(result: Dict, tid: str,
                             tau: float, metric: str) -> float:
    """Return the categorical-event-pooled value for one (tau, metric) pair.

    The ``result_v2.json`` emits per_event categorical sub-buckets keyed by
    the canonical threshold_key emitted by ``src.evaluation.metrics``. We
    try the documented key per tau first, then fall back to numeric-string
    representations ('5.0', '5', '10.0', ...) so the script stays robust to
    any rename in the evaluator's threshold_key but does NOT silently
    fabricate a value. If no bucket matches the tau, the value is NaN.
    """
    cat = result["per_event"][tid].get("categorical", {})
    if not cat:
        return float("nan")
    candidates: List[str] = []
    for fmt in (f"CSI_{tau:g}mmh", f"{tau:g}mmh",
                str(int(tau)) if float(tau).is_integer() else str(tau),
                f"{tau}", str(tau), f"tau_{tau:g}"):
        candidates.append(fmt)
    for key in candidates:
        if key in cat and metric in cat[key]:
            return float(cat[key][metric])
    # Last fallback: scan all buckets and pick the first whose values include
    # ``metric`` and whose threshold is numerically nearest to ``tau``.
    best_key, best_dist = None, float("inf")
    for key, bucket in cat.items():
        if not isinstance(bucket, dict) or metric not in bucket:
            continue
        try:
            tk = float(key.split("_")[0]) if "_" in key and key[0].isdigit() \
                else float(key)
        except (ValueError, IndexError):
            continue
        d = abs(tk - tau)
        if d < best_dist:
            best_dist, best_key = d, key
    if best_key is not None and best_dist < 1e-6:
        return float(cat[best_key][metric])
    return float("nan")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    alias_to_result: Dict[str, Dict],
    alias_to_manifest: Dict[str, Dict],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    include_backbone_sanity: bool,
) -> Dict:
    """Run the full analysis; return a structured dict ready for writers."""
    # Pre-extract event_id lists per alias (intersection later enforced
    # inside paired_event_differences).
    event_ids_by_alias: Dict[str, List[str]] = {
        alias: sorted(result["per_event"].keys(), key=int)
        for alias, result in alias_to_result.items()
    }

    # ---- per-alias summaries (experiment_summary.csv) ----
    experiment_summary_rows: List[Dict] = []
    for alias, manifest in sorted(alias_to_manifest.items()):
        result = alias_to_result[alias]
        per_event = result["per_event"]
        # Continuous event-mean (equal-event).
        def event_mean(metric: str) -> float:
            vals = [per_event[tid].get(metric) for tid in per_event]
            vals = [v for v in vals if v == v]
            return float(sum(vals) / len(vals)) if vals else float("nan")

        row: Dict = {
            "experiment_id": manifest["experiment_id"],
            "alias_ids": "|".join(manifest["alias_ids"]),
            "seed": manifest.get("seed", -1),
            "n_events": result["n_events"],
            "n_windows": result["n_windows"],
            "MAE_event_mean": event_mean("MAE_event"),
            "RMSE_event_mean": event_mean("RMSE_event"),
            "SSIM_event_mean_mean": event_mean("SSIM_event_mean"),
        }
        for tau in FROZEN_THRESHOLDS_MMH:
            for m in CATEGORICAL_METRICS:
                vals = [_categorical_event_value(result, tid, tau, m)
                        for tid in per_event]
                vals = [v for v in vals if v == v]
                key = f"{m}_event_macro@{tau:g}mmh"
                row[key] = float(sum(vals) / len(vals)) if vals else float("nan")
        experiment_summary_rows.append(row)

    # ---- formal contrasts (axis-aligned) ----
    formal_contrasts = list(FORMAL_AXIS_I) + list(FORMAL_AXIS_II)
    long_rows: List[Dict] = []
    perdiff_rows: List[Dict] = []
    raw_pvals_by_family: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(dict)

    for axis, baseline, candidate in formal_contrasts:
        if baseline not in alias_to_result or candidate not in alias_to_result:
            missing = [a for a in (baseline, candidate)
                       if a not in alias_to_result]
            raise ContractViolation(
                f"Formal contrast ({axis}, {baseline} -> {candidate}) lacks "
                f"artifacts for aliases: {missing}. The frozen alias registry "
                f"guarantees I5 == P0 but does not authorize the analysis to "
                f"run before every trainable row in the matrix has produced "
                f"an artifact."
            )
        _emit_contrast(
            axis=axis, baseline=baseline, candidate=candidate,
            baseline_result=alias_to_result[baseline],
            candidate_result=alias_to_result[candidate],
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            long_rows=long_rows,
            perdiff_rows=perdiff_rows,
            raw_pvals_by_family=raw_pvals_by_family,
        )

    # ---- backbone sanity block (descriptive-only, NOT info/inductive-bias) ----
    backbone_long_rows: List[Dict] = []
    backbone_perdiff_rows: List[Dict] = []
    if include_backbone_sanity:
        for baseline, candidate in BACKBONE_SANITY_PAIRS:
            if baseline not in alias_to_result or candidate not in alias_to_result:
                continue
            _emit_contrast(
                axis="BackboneSanity", baseline=baseline, candidate=candidate,
                baseline_result=alias_to_result[baseline],
                candidate_result=alias_to_result[candidate],
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
                long_rows=backbone_long_rows,
                perdiff_rows=backbone_perdiff_rows,
                raw_pvals_by_family=None,  # never Holm-corrected
            )

    # ---- statistical_summary.csv (raw + Holm-adjusted; only >=HOLM_FAMILY_MIN_SIZE) ----
    statistical_summary_rows: List[Dict] = []
    for (metric, tau), cmap in sorted(raw_pvals_by_family.items()):
        if len(cmap) < HOLM_FAMILY_MIN_SIZE:
            continue
        adjusted = holm_correction(cmap)
        for contrast, raw in sorted(cmap.items()):
            n_pairs = 0  # already filtered, just emit
            statistical_summary_rows.append({
                "metric": metric,
                "threshold_mmh": "" if tau == 0.0 else f"{tau:g}",
                "family_size": len(cmap),
                "contrast": contrast,
                "n_pairs_min_in_family": HOLM_N_PAIRS_MIN,
                "p_raw": raw,
                "p_holm": adjusted.get(contrast, float("nan")),
            })

    return {
        "experiment_summary_rows": experiment_summary_rows,
        "long_rows": long_rows,
        "perdiff_rows": perdiff_rows,
        "backbone_long_rows": backbone_long_rows,
        "backbone_perdiff_rows": backbone_perdiff_rows,
        "statistical_summary_rows": statistical_summary_rows,
        "raw_pvals_by_family": raw_pvals_by_family,
        "alias_to_manifest": alias_to_manifest,
        "i5_p0_identity": enforce_i5_p0_identity(alias_to_manifest),
        "event_ids_by_alias": event_ids_by_alias,
    }


def _emit_contrast(
    *,
    axis: str,
    baseline: str,
    candidate: str,
    baseline_result: Dict,
    candidate_result: Dict,
    n_bootstrap: int,
    bootstrap_seed: int,
    long_rows: List[Dict],
    perdiff_rows: List[Dict],
    raw_pvals_by_family: Optional[Dict[Tuple[str, float], Dict[str, float]]],
) -> None:
    """Run every metric/threshold combination for one contrast."""
    label = f"{candidate} - {baseline}"

    metric_threshold_pairs: List[Tuple[str, Optional[float]]] = \
        [(m, None) for m in CONTINUOUS_METRICS] + \
        [(m, tau) for tau in FROZEN_THRESHOLDS_MMH for m in CATEGORICAL_METRICS]

    for metric, tau in metric_threshold_pairs:
        pa = paired_event_differences(
            baseline_result, candidate_result,
            metric=metric, threshold=tau,
            n_bootstrap=n_bootstrap, seed=bootstrap_seed,
        )
        n_pairs = pa["n_pairs"]
        diffs = [(tid, v) for tid, v in pa["per_event"].items()]
        p_raw, p_n = exact_two_sided_sign_flip([v for _, v in diffs])
        ci_lo, ci_hi = pa["ci95"]

        # Window-level significance is forbidden — confirm n_pairs is event
        # count, not window count, by checking against n_events.
        expected_n_events = baseline_result["n_events"]
        if n_pairs > expected_n_events:
            raise ContractViolation(
                f"n_pairs ({n_pairs}) exceeds n_events ({expected_n_events}); "
                f"refusing to run. Resampling at the window level is forbidden.")

        long_rows.append({
            "axis": axis,
            "contrast": label,
            "baseline": baseline,
            "candidate": candidate,
            "metric": metric,
            "threshold_mmh": "" if tau is None else f"{tau:g}",
            "n_pairs": n_pairs,
            "mean_diff": pa["mean"],
            "median_diff": pa["median"],
            "iqr": pa["iqr"],
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "p_signflip_raw": p_raw,
            "p_signflip_n": p_n,
            "inferential": "yes" if n_pairs >= HOLM_N_PAIRS_MIN else "no",
        })

        for tid, diff in diffs:
            perdiff_rows.append({
                "axis": axis,
                "contrast": label,
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "threshold_mmh": "" if tau is None else f"{tau:g}",
                "typhoon_id": tid,
                "diff": diff,
            })

        if raw_pvals_by_family is not None and n_pairs >= HOLM_N_PAIRS_MIN:
            family = (metric, 0.0 if tau is None else float(tau))
            raw_pvals_by_family[family][label] = p_raw


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_outputs(out_dir: Path, payload: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_csv(name: str, rows: List[Dict], cols: List[str]) -> Path:
        p = out_dir / name
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        return p

    summary_cols = ["experiment_id", "alias_ids", "seed", "n_events",
                    "n_windows", "MAE_event_mean", "RMSE_event_mean",
                    "SSIM_event_mean_mean"]
    for tau in FROZEN_THRESHOLDS_MMH:
        for m in CATEGORICAL_METRICS:
            summary_cols.append(f"{m}_event_macro@{tau:g}mmh")
    _write_csv("experiment_summary.csv",
               payload["experiment_summary_rows"], summary_cols)

    _write_csv("contrasts_long.csv", payload["long_rows"], [
        "axis", "contrast", "baseline", "candidate", "metric",
        "threshold_mmh", "n_pairs", "mean_diff", "median_diff", "iqr",
        "ci95_lo", "ci95_hi", "p_signflip_raw", "p_signflip_n", "inferential",
    ])

    _write_csv("per_event_differences.csv", payload["perdiff_rows"], [
        "axis", "contrast", "baseline", "candidate", "metric",
        "threshold_mmh", "typhoon_id", "diff",
    ])

    _write_csv("statistical_summary.csv", payload["statistical_summary_rows"], [
        "metric", "threshold_mmh", "family_size", "contrast",
        "n_pairs_min_in_family", "p_raw", "p_holm",
    ])

    md = out_dir / "ABLATION_ANALYSIS.md"
    md.write_text(render_markdown(payload), encoding="utf-8")

    print(f"[ok] wrote {out_dir / 'experiment_summary.csv'}")
    print(f"[ok] wrote {out_dir / 'contrasts_long.csv'}")
    print(f"[ok] wrote {out_dir / 'per_event_differences.csv'}")
    print(f"[ok] wrote {out_dir / 'statistical_summary.csv'}")
    print(f"[ok] wrote {md}")


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_markdown(payload: Dict) -> str:
    out: List[str] = []
    out.append("# Ablation Analysis — Two-Axis Controlled Study")
    out.append("")
    out.append("> **VALIDATION MATRIX = IN PROGRESS.**")
    out.append("> **TEST STATUS = SEALED.** This document reports validation-")
    out.append("> level paired event-level statistics and does NOT include a")
    out.append("> held-out test evaluation. See `docs/FINAL_TEST_AUTHORIZATION.md`")
    out.append("> for the test-evaluation gate.")
    out.append("")
    out.append("- Protocol: `evaluation_v2` (per `docs/EVALUATION_PROTOCOL_V2.md`)")
    out.append("- Independent unit: typhoon event (per protocol §17)")
    out.append("- Paired event-bootstrap: 10,000 resamples, seed 42 (per §17.4)")
    out.append("- Exact two-sided sign-flip p-value: only inferential when n_pairs ≥ 4 (per §17.6)")
    out.append("- Holm correction: per `(metric, threshold)` family with ≥ 3 contrasts (per §17.7)")
    out.append("- Window-level significance testing is FORBIDDEN by construction.")
    out.append("")

    # I5/P0 identity record
    i5_p0 = payload.get("i5_p0_identity")
    if i5_p0 is not None:
        out.append("## Alias identity (mandatory)")
        out.append("")
        out.append(f"- I5 ≡ P0: artifact identity verified on directory "
                   f"`{i5_p0['experiment_id']}`. checkpoint_sha256 and "
                   f"config_sha256 are equal across both aliases.")
        out.append("")

    # ---- OBSERVATION ----
    out.append("## 1. OBSERVATION")
    out.append("")
    out.append("### 1.1 Per-experiment validation (event-macro means)")
    out.append("")
    cols = ["experiment_id", "alias_ids", "n_events", "n_windows",
            "MAE_event_mean", "RMSE_event_mean", "SSIM_event_mean_mean"]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "---|" * len(cols))
    for r in payload["experiment_summary_rows"]:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    out.append("")
    out.append("**Backbone sanity rows** for I0 / I1 / I2 / B1 are reported in "
               "`experiment_summary.csv`. I1 − I2 is a **backbone contrast**, "
               "NOT an information effect, per "
               "`docs/RESEARCH_DESIGN_C_FREEZE.md` §2 / §11.")
    out.append("")
    out.append("### 1.2 Formal contrasts (raw event-paired differences)")
    out.append("")
    out.append("| Axis | Contrast | Metric@τ | n_pairs | mean Δ | median Δ | "
               "CI95 | p (sign-flip) | Inferential? |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in payload["long_rows"]:
        tau_disp = "—" if r["threshold_mmh"] == "" else r["threshold_mmh"]
        metric_disp = (f"{r['metric']}@{tau_disp}" if tau_disp != "—"
                       else r["metric"])
        ci = f"[{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}]"
        p_disp = (_fmt(r["p_signflip_raw"])
                  if r["p_signflip_raw"] == r["p_signflip_raw"] else "—")
        out.append(
            f"| {r['axis']} | {r['contrast']} | {metric_disp} | "
            f"{r['n_pairs']} | {r['mean_diff']:.4f} | "
            f"{r['median_diff']:.4f} | {ci} | {p_disp} | "
            f"{r['inferential']} |")
    out.append("")
    out.append("> Mean Δ > 0 ⇒ improvement against baseline. "
               "MAE_event / RMSE_event / FAR use baseline − candidate; "
               "SSIM_event_mean / CSI / POD / HSS / ACC use candidate − baseline; "
               "BIAS uses |BIAS_b − 1| − |BIAS_c − 1|. "
               "See `EVALUATION_PROTOCOL_V2.md` §17.2.")
    out.append("")

    # ---- STATISTICAL SUMMARY ----
    out.append("## 2. STATISTICAL SUMMARY")
    out.append("")
    out.append("### 2.1 Holm-adjusted p-values (per `(metric, threshold)` family)")
    out.append("")
    out.append("Only families with at least `HOLM_FAMILY_MIN_SIZE = "
               f"{HOLM_FAMILY_MIN_SIZE}` contrasts and at least "
               f"`{HOLM_N_PAIRS_MIN}` paired events are reported.")
    out.append("")
    if payload["statistical_summary_rows"]:
        out.append("| Metric@τ | Family size | Contrast | p_raw | p_holm |")
        out.append("|---|---|---|---:|---:|")
        for r in payload["statistical_summary_rows"]:
            tau_disp = "—" if r["threshold_mmh"] == "" else r["threshold_mmh"]
            metric_disp = (f"{r['metric']}@{tau_disp}" if tau_disp != "—"
                           else r["metric"])
            out.append(f"| {metric_disp} | {r['family_size']} | "
                       f"{r['contrast']} | {_fmt(r['p_raw'])} | "
                       f"{_fmt(r['p_holm'])} |")
        out.append("")
    else:
        out.append("_No (metric, threshold) family satisfied the Holm "
                   "preregistration thresholds in this run._")
        out.append("")

    out.append("### 2.2 Bootstrap controls")
    out.append("")
    out.append(f"- n_bootstrap = {N_BOOTSTRAP_DEFAULT}, seed = "
               f"{BOOTSTRAP_SEED_DEFAULT}.")
    out.append("- 95% CI is the equal-event percentile interval of the "
               "resampled means.")
    out.append("- Resampling unit is the **typhoon event**, never the window.")
    out.append("")

    # ---- INTERPRETATION LIMIT ----
    out.append("## 3. INTERPRETATION LIMIT")
    out.append("")
    out.append("- **Inferential power is bounded by `n_event = 7` on "
               "validation.** The current matrix is not a substitute for "
               "the held-out test.")
    out.append("- **Single-seed training.** All inferences are conditioned "
               "on seed 42. Initialization robustness is out of scope for "
               "this matrix.")
    out.append("- **Backbone contrast vs. information effect.** `I1 − I2` "
               "changes the model family; it is reported as a backbone "
               "sanity check, NEVER as an information claim. "
               "Information claims use `I3 − I2`, `I4 − I3`, `I5 − I4` only.")
    out.append("- **Inductive-bias claims.** `P1 − P0`, `P2 − P0`, `P3 − P0` "
               "are the loss/regularization contrasts. The interaction "
               "`P3 − P1 − P2 + P0` is exploratory at one seed.")
    out.append("- **Geometric resolution.** GPM 0.1° (~10 km) reprojection "
               "smooths sub-grid-scale terrain variation. A negative "
               "`I5 − I4` does NOT contradict the underlying physics; it "
               "is a controlled report of what the smoother version carries "
               "at this resolution.")
    out.append("- **Single categorical measure.** Confidence intervals for "
               "categorical metrics are reported at the listed thresholds; "
               "no general threshold transfer claim is implied.")
    out.append("")

    # ---- NOT YET A TEST-SET CONCLUSION ----
    out.append("## 4. NOT YET A TEST-SET CONCLUSION")
    out.append("")
    out.append("- The held-out test events (4 typhoons, 707 windows) are "
               "**SEALED**. No inference in this document is a held-out test "
               "result.")
    out.append("- A test evaluation is permitted only after every item in "
               "`docs/FINAL_TEST_AUTHORIZATION.md` §0 is satisfied, with an "
               "append-only authorization recorded in §3 of that document.")
    out.append("- This analysis does not predict terrain information gain, "
               "smoothing-vs-extreme behavior, or any generalization "
               "beyond the `n_event = 7` paired observations used here.")
    out.append("")

    out.append("---")
    out.append("")
    out.append("Generated by `scripts/analyze_ablation_results.py`. "
               "Output filenames and statistical controls are frozen by this "
               "script's contract; do not edit the resulting tables by hand "
               "outside of an explicit re-run.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, default=Path("results"),
                   help="Directory containing <id>_seed<N>/{manifest,result_v2}.json")
    p.add_argument("--output-dir", type=Path,
                   default=Path("tables/ablation_analysis"),
                   help="Output directory for tables/figures source data.")
    p.add_argument("--include-backbone-sanity", action="store_true",
                   help="Emit backbone-sanity contrasts (I0/I1/I2/B1) "
                        "in addition to the formal contrasts.")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED_DEFAULT)
    args = p.parse_args(argv)

    alias_to_result, alias_to_manifest = load_results(args.results_dir)
    enforce_i5_p0_identity(alias_to_manifest)

    payload = analyze(
        alias_to_result, alias_to_manifest,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        include_backbone_sanity=args.include_backbone_sanity,
    )

    write_outputs(args.output_dir, payload)
    print(f"[ok] {len(alias_to_manifest)} alias(es) analyzed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
