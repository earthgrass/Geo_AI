# `results/` — paper-asset directory

This directory holds **paper-grade experiment assets**, one folder per experiment
and per seed. Everything here is intended to be git-tracked **except** the
large binary checkpoints (which live on the GPU host with their SHA256
recorded inside each `manifest.json`).

## Layout (canonical)

```
results/
├── I0_persistence_seed42/
│   ├── manifest.json           # experiment identity + sha256 fingerprints
│   ├── result_v2.json          # evaluator v2 output (authoritative)
│   ├── validation.md           # markdown view of result_v2.json
│   ├── history.json            # per-epoch metrics (train/val curves)
│   ├── config.yaml             # config snapshot used for this run
│   └── events.csv              # per-event diagnostics (optional)
├── I1_plain_convlstm_seed42/
├── I2_resconvlstm_seed42_v2/   # v1 -> v2 re-evaluation (NO training)
│   └── ...
├── I3_resconvlstm_cma_seed42/
├── I4_static_terrain_seed42/
├── I5_terrain_geometry_seed42/ # ALSO aliased as P0
├── B1_trajgru_seed42/
├── P1_resconvlstm_smooth_seed42/
├── P2_resconvlstm_extreme_seed42/
├── P3_resconvlstm_smooth_extreme_seed42/
└── ...
```

I5 and P0 live in **the same folder**, because the alias registry resolves
both to the same artifact (`E5_terrain_geometry`). The directory name itself
does not encode the alias; the alias is recorded in `manifest.json`.

## What gets committed

| File | Committed? | Reason |
|---|---|---|
| `manifest.json` | ✅ | identity + fingerprint; ~10 KB |
| `result_v2.json` | ✅ | machine-readable v2 result; ~10–50 KB |
| `validation.md`  | ✅ | human-readable v2 view |
| `history.json`   | ✅ | per-epoch curves (no images) |
| `config.yaml`    | ✅ | configuration snapshot |
| `events.csv`     | ✅ | per-event diagnostics (optional) |
| `*.png` figures  | ❌  | generated; commit only if published |
| `*.pth` weights  | ❌  | large blob; SHA256 stored in manifest |
| intermediate / debug / sample dumps | ❌  | place under `outputs/` instead |

## Fingerprint contract (manifest.json)

Every `manifest.json` MUST contain:

```json
{
  "experiment_id": "I5_terrain_geometry",
  "alias_ids": ["I5", "P0"],
  "seed": 42,
  "git_commit": "...",
  "git_dirty": false,
  "git_status_clean": true,
  "config_sha256": "...",
  "dataset_sha256": "...",
  "split_sha256": "...",
  "normalization_sha256": "...",
  "checkpoint_sha256": "...",
  "checkpoint_path_on_host": "saved_models/<id>_seed42/<id>_seed42_best.pth",
  "epochs": 20,
  "best_epoch": <int>,
  "val_base_rain_mse": <float>,
  "param_count": <int>,
  "protocol_id": "evaluation_v2",
  "evaluator_v2_reeval_only": false,
  "notes": "..."
}
```

> `scripts/verify_experiment_artifact.py` reads each manifest and emits
> COMPATIBLE / INCOMPATIBLE. It is the only authoritative compatibility check.

## Quick commit helper

After a GPU run completes, from the project root:

```bash
git add results/<id>_seed<N>/{manifest.json,result_v2.json,validation.md,history.json,config.yaml}
git status
python scripts/verify_experiment_artifact.py --manifest results/<id>_seed<N>/manifest.json
git commit -m "results: <id> seed<N> val rain_mse=<float>"
```

> We deliberately do NOT auto-track `results/<id>/`. Authors commit
> files explicitly. This avoids silently committing partial runs.

## What NEVER goes here

- `outputs/...` — use `outputs/` for throwaway, raw, debugging artifacts only.
- `saved_models/*.pth` — checkpoints; recorded only by SHA256 in manifest.
- `TIFdata/` — raw DEM rasters; ignored by `.gitignore` (`*.tif`).
- `ConvLSTM_Dataset_128.h5` — ignored by `.gitignore` (`*.h5`).

## When you change an experiment config or normalization

If you edit `configs/experiments/*.yaml`, `configs/splits_v1.yaml`,
`configs/normalization_v1.json`, or `configs/evaluation_thresholds_v1.json`,
**all existing `results/<id>/manifest.json` entries are invalidated.** Re-run
the affected experiments and update `config_sha256` / `dataset_sha256` /
`split_sha256` / `normalization_sha256`. The artifact verifier will refuse to
mark old manifests COMPATIBLE unless you explicitly re-authorize.
