# Ablation Analysis — Two-Axis Controlled Study

> **VALIDATION MATRIX = COMPLETE (single seed = 42).**
> **MULTI-SEED = NOT YET CONFIRMED.**
> **TEST STATUS = SEALED.** This document reports validation-
> level paired event-level statistics and does NOT include a
> held-out test evaluation. See `docs/FINAL_TEST_AUTHORIZATION.md`
> for the test-evaluation gate.

- Protocol: `evaluation_v2` (per `docs/EVALUATION_PROTOCOL_V2.md`)
- Independent unit: typhoon event (per protocol §17)
- Paired event-bootstrap: 10,000 resamples, seed 42 (per §17.4)
- Exact two-sided sign-flip p-value: only inferential when n_pairs ≥ 4 (per §17.6)
- Holm correction: per `(metric, threshold)` family with ≥ 3 contrasts (per §17.7)
- Window-level significance testing is FORBIDDEN by construction.

## Alias identity (mandatory)

- I5 ≡ P0: artifact identity verified on directory `E5_terrain_geometry`. checkpoint_sha256 and config_sha256 are equal across both aliases.

## 1. OBSERVATION

### 1.1 Per-experiment validation (event-macro means)

| experiment_id | alias_ids | n_events | n_windows | MAE_event_mean | RMSE_event_mean | SSIM_event_mean_mean |
|---|---|---|---|---|---|---|
| B1_trajgru |  | 7 | 1266 | 0.2332 | 0.8935 | 0.9380 |
| E0_persistence | I0 | 7 | 1266 | 0.1380 | 0.5763 | 0.9775 |
| E1_plain_convlstm | I1 | 7 | 1266 | 0.2376 | 0.8831 | 0.9355 |
| E2_resconvlstm | I2 | 7 | 1266 | 0.1313 | 0.5036 | 0.9798 |
| E3_resconvlstm_cma | I3 | 7 | 1266 | 0.1427 | 0.5172 | 0.9789 |
| E4_static_terrain | I4 | 7 | 1266 | 0.1405 | 0.5158 | 0.9794 |
| E5_terrain_geometry | I5|P0 | 7 | 1266 | 0.1429 | 0.5188 | 0.9790 |
| P1_resconvlstm_smooth | P1 | 7 | 1266 | 0.1227 | 0.4994 | 0.9806 |
| E6_terrain_extreme | P2 | 7 | 1266 | 0.3532 | 0.8308 | 0.9013 |
| P3_resconvlstm_smooth_extreme | P3 | 7 | 1266 | 0.2228 | 0.6708 | 0.9550 |

**Backbone sanity rows** for I0 / I1 / I2 / B1 are reported in `experiment_summary.csv`. I1 − I2 is a **backbone contrast**, NOT an information effect, per `docs/RESEARCH_DESIGN_C_FREEZE.md` §2 / §11.

### 1.2 Formal contrasts (raw event-paired differences)

| Axis | Contrast | Metric@τ | n_pairs | mean Δ | median Δ | CI95 | p (sign-flip) | Inferential? |
|---|---|---|---:|---:|---:|---:|---:|---|
| AxisI | I3 - I2 | MAE_event | 7 | -0.0114 | -0.0163 | [-0.0212, 0.0005] | 0.1250 | yes |
| AxisI | I3 - I2 | RMSE_event | 7 | -0.0136 | -0.0176 | [-0.0342, 0.0091] | 0.4531 | yes |
| AxisI | I3 - I2 | SSIM_event_mean | 7 | -0.0009 | -0.0015 | [-0.0018, 0.0006] | 0.1250 | yes |
| AxisI | I3 - I2 | CSI@5 | 4 | 0.0196 | 0.0192 | [0.0079, 0.0312] | 0.1250 | yes |
| AxisI | I3 - I2 | POD@5 | 4 | 0.0141 | 0.0137 | [-0.0042, 0.0324] | 1.0000 | yes |
| AxisI | I3 - I2 | FAR@5 | 4 | 0.0971 | 0.0210 | [0.0144, 0.2540] | 0.1250 | yes |
| AxisI | I3 - I2 | HSS@5 | 4 | 0.0242 | 0.0165 | [0.0076, 0.0450] | 0.1250 | yes |
| AxisI | I3 - I2 | ACC@5 | 7 | 0.0003 | 0.0000 | [0.0000, 0.0006] | 1.0000 | yes |
| AxisI | I3 - I2 | BIAS@5 | 4 | -0.0335 | -0.0333 | [-0.0774, 0.0102] | 0.6250 | yes |
| AxisI | I3 - I2 | CSI@10 | 4 | 0.0182 | 0.0073 | [0.0021, 0.0452] | 0.6250 | yes |
| AxisI | I3 - I2 | POD@10 | 4 | 0.0171 | 0.0029 | [-0.0028, 0.0513] | 1.0000 | yes |
| AxisI | I3 - I2 | FAR@10 | 3 | 0.0184 | 0.0180 | [0.0171, 0.0201] | 0.2500 | no |
| AxisI | I3 - I2 | HSS@10 | 4 | 0.0165 | 0.0080 | [0.0025, 0.0391] | 0.6250 | yes |
| AxisI | I3 - I2 | ACC@10 | 7 | 0.0001 | 0.0000 | [0.0000, 0.0002] | 1.0000 | yes |
| AxisI | I3 - I2 | BIAS@10 | 4 | 0.0054 | -0.0054 | [-0.0276, 0.0492] | 1.0000 | yes |
| AxisI | I3 - I2 | CSI@20 | 3 | 0.0308 | 0.0030 | [-0.0243, 0.1136] | 1.0000 | no |
| AxisI | I3 - I2 | POD@20 | 3 | 0.0338 | -0.0046 | [-0.0301, 0.1361] | 1.0000 | no |
| AxisI | I3 - I2 | FAR@20 | 3 | 0.0175 | 0.0216 | [-0.0199, 0.0509] | 1.0000 | no |
| AxisI | I3 - I2 | HSS@20 | 3 | 0.0410 | 0.0037 | [-0.0328, 0.1521] | 1.0000 | no |
| AxisI | I3 - I2 | ACC@20 | 7 | 0.0000 | 0.0000 | [-0.0000, 0.0001] | 1.0000 | yes |
| AxisI | I3 - I2 | BIAS@20 | 3 | 0.0351 | -0.0331 | [-0.0364, 0.1749] | 1.0000 | no |
| AxisI | I3 - I2 | CSI@30 | 3 | -0.0193 | -0.0113 | [-0.0465, 0.0000] | 1.0000 | no |
| AxisI | I3 - I2 | POD@30 | 3 | -0.0261 | -0.0176 | [-0.0605, 0.0000] | 1.0000 | no |
| AxisI | I3 - I2 | FAR@30 | 2 | 0.0105 | 0.0105 | [0.0053, 0.0156] | 0.5000 | no |
| AxisI | I3 - I2 | HSS@30 | 3 | -0.0288 | -0.0164 | [-0.0699, 0.0000] | 1.0000 | no |
| AxisI | I3 - I2 | ACC@30 | 7 | 0.0000 | 0.0000 | [-0.0000, 0.0000] | 1.0000 | yes |
| AxisI | I3 - I2 | BIAS@30 | 3 | -0.0511 | -0.0427 | [-0.1107, 0.0000] | 1.0000 | no |
| AxisI | I4 - I3 | MAE_event | 7 | 0.0022 | 0.0057 | [-0.0103, 0.0136] | 0.4531 | yes |
| AxisI | I4 - I3 | RMSE_event | 7 | 0.0013 | 0.0058 | [-0.0138, 0.0151] | 1.0000 | yes |
| AxisI | I4 - I3 | SSIM_event_mean | 7 | 0.0005 | 0.0012 | [-0.0011, 0.0019] | 0.4531 | yes |
| AxisI | I4 - I3 | CSI@5 | 4 | -0.0067 | -0.0038 | [-0.0135, -0.0017] | 0.1250 | yes |
| AxisI | I4 - I3 | POD@5 | 4 | -0.0073 | -0.0059 | [-0.0226, 0.0065] | 1.0000 | yes |
| AxisI | I4 - I3 | FAR@5 | 4 | -0.0340 | -0.0060 | [-0.0991, 0.0065] | 1.0000 | yes |
| AxisI | I4 - I3 | HSS@5 | 4 | -0.0058 | -0.0043 | [-0.0103, -0.0025] | 0.1250 | yes |
| AxisI | I4 - I3 | ACC@5 | 7 | -0.0001 | -0.0000 | [-0.0004, -0.0000] | 1.0000 | yes |
| AxisI | I4 - I3 | BIAS@5 | 4 | 0.0037 | -0.0011 | [-0.0332, 0.0416] | 1.0000 | yes |
| AxisI | I4 - I3 | CSI@10 | 4 | -0.0057 | -0.0038 | [-0.0124, 0.0006] | 1.0000 | yes |
| AxisI | I4 - I3 | POD@10 | 4 | -0.0048 | -0.0064 | [-0.0213, 0.0135] | 1.0000 | yes |
| AxisI | I4 - I3 | FAR@10 | 3 | -0.0045 | 0.0010 | [-0.0241, 0.0097] | 1.0000 | no |
| AxisI | I4 - I3 | HSS@10 | 4 | -0.0054 | -0.0043 | [-0.0113, 0.0005] | 1.0000 | yes |
| AxisI | I4 - I3 | ACC@10 | 7 | -0.0001 | 0.0000 | [-0.0002, -0.0000] | 1.0000 | yes |
| AxisI | I4 - I3 | BIAS@10 | 4 | 0.0004 | -0.0113 | [-0.0339, 0.0463] | 1.0000 | yes |
| AxisI | I4 - I3 | CSI@20 | 3 | -0.0656 | -0.0083 | [-0.1916, 0.0030] | 1.0000 | no |
| AxisI | I4 - I3 | POD@20 | 3 | -0.0771 | -0.0160 | [-0.2268, 0.0113] | 1.0000 | no |
| AxisI | I4 - I3 | FAR@20 | 3 | -0.0057 | -0.0141 | [-0.0333, 0.0303] | 1.0000 | no |
| AxisI | I4 - I3 | HSS@20 | 3 | -0.0943 | -0.0114 | [-0.2749, 0.0035] | 1.0000 | no |
| AxisI | I4 - I3 | ACC@20 | 7 | -0.0000 | 0.0000 | [-0.0000, 0.0000] | 1.0000 | yes |
| AxisI | I4 - I3 | BIAS@20 | 3 | -0.1103 | -0.0473 | [-0.3197, 0.0360] | 1.0000 | no |
| AxisI | I4 - I3 | CSI@30 | 3 | -0.0052 | -0.0032 | [-0.0124, 0.0000] | 1.0000 | no |
| AxisI | I4 - I3 | POD@30 | 3 | -0.0072 | -0.0057 | [-0.0158, 0.0000] | 1.0000 | no |
| AxisI | I4 - I3 | FAR@30 | 2 | 0.0122 | 0.0122 | [-0.0062, 0.0306] | 1.0000 | no |
| AxisI | I4 - I3 | HSS@30 | 3 | -0.0078 | -0.0050 | [-0.0184, 0.0000] | 1.0000 | no |
| AxisI | I4 - I3 | ACC@30 | 7 | -0.0000 | 0.0000 | [-0.0000, 0.0000] | 1.0000 | yes |
| AxisI | I4 - I3 | BIAS@30 | 3 | -0.0158 | -0.0227 | [-0.0248, 0.0000] | 1.0000 | no |
| AxisI | I5 - I4 | MAE_event | 7 | -0.0024 | -0.0014 | [-0.0108, 0.0072] | 1.0000 | yes |
| AxisI | I5 - I4 | RMSE_event | 7 | -0.0029 | -0.0042 | [-0.0171, 0.0137] | 0.4531 | yes |
| AxisI | I5 - I4 | SSIM_event_mean | 7 | -0.0003 | -0.0007 | [-0.0012, 0.0009] | 0.1250 | yes |
| AxisI | I5 - I4 | CSI@5 | 4 | 0.0041 | 0.0046 | [0.0008, 0.0073] | 0.6250 | yes |
| AxisI | I5 - I4 | POD@5 | 4 | 0.0064 | 0.0080 | [-0.0059, 0.0186] | 1.0000 | yes |
| AxisI | I5 - I4 | FAR@5 | 4 | -0.0099 | -0.0111 | [-0.0271, 0.0086] | 0.6250 | yes |
| AxisI | I5 - I4 | HSS@5 | 4 | 0.0033 | 0.0041 | [0.0004, 0.0060] | 0.6250 | yes |
| AxisI | I5 - I4 | ACC@5 | 7 | 0.0001 | 0.0000 | [-0.0000, 0.0004] | 1.0000 | yes |
| AxisI | I5 - I4 | BIAS@5 | 4 | 0.0214 | 0.0319 | [-0.0056, 0.0379] | 0.6250 | yes |
| AxisI | I5 - I4 | CSI@10 | 4 | -0.0022 | -0.0022 | [-0.0058, 0.0015] | 1.0000 | yes |
| AxisI | I5 - I4 | POD@10 | 4 | -0.0054 | 0.0005 | [-0.0199, 0.0031] | 1.0000 | yes |
| AxisI | I5 - I4 | FAR@10 | 3 | 0.0024 | 0.0016 | [-0.0175, 0.0230] | 1.0000 | no |
| AxisI | I5 - I4 | HSS@10 | 4 | -0.0017 | -0.0021 | [-0.0051, 0.0017] | 1.0000 | yes |
| AxisI | I5 - I4 | ACC@10 | 7 | 0.0001 | 0.0000 | [-0.0000, 0.0002] | 1.0000 | yes |
| AxisI | I5 - I4 | BIAS@10 | 4 | -0.0128 | 0.0025 | [-0.0549, 0.0139] | 1.0000 | yes |
| AxisI | I5 - I4 | CSI@20 | 3 | 0.0384 | 0.0193 | [0.0004, 0.0955] | 0.2500 | no |
| AxisI | I5 - I4 | POD@20 | 3 | 0.0454 | 0.0313 | [-0.0031, 0.1080] | 1.0000 | no |
| AxisI | I5 - I4 | FAR@20 | 3 | 0.0018 | 0.0080 | [-0.0298, 0.0274] | 1.0000 | no |
| AxisI | I5 - I4 | HSS@20 | 3 | 0.0584 | 0.0265 | [0.0005, 0.1481] | 0.2500 | no |
| AxisI | I5 - I4 | ACC@20 | 7 | 0.0000 | 0.0000 | [-0.0000, 0.0000] | 1.0000 | yes |
| AxisI | I5 - I4 | BIAS@20 | 3 | 0.0700 | 0.0735 | [-0.0148, 0.1512] | 1.0000 | no |
| AxisI | I5 - I4 | CSI@30 | 3 | 0.0237 | 0.0333 | [0.0000, 0.0377] | 1.0000 | no |
| AxisI | I5 - I4 | POD@30 | 3 | 0.0321 | 0.0453 | [0.0000, 0.0511] | 1.0000 | no |
| AxisI | I5 - I4 | FAR@30 | 2 | -0.0216 | -0.0216 | [-0.0429, -0.0002] | 0.5000 | no |
| AxisI | I5 - I4 | HSS@30 | 3 | 0.0353 | 0.0484 | [0.0000, 0.0574] | 1.0000 | no |
| AxisI | I5 - I4 | ACC@30 | 7 | 0.0000 | 0.0000 | [-0.0000, 0.0000] | 1.0000 | yes |
| AxisI | I5 - I4 | BIAS@30 | 3 | 0.0644 | 0.0825 | [0.0000, 0.1107] | 1.0000 | no |
| AxisII | P1 - P0 | MAE_event | 7 | 0.0202 | 0.0178 | [0.0142, 0.0279] | 0.0156 | yes |
| AxisII | P1 - P0 | RMSE_event | 7 | 0.0193 | 0.0123 | [0.0058, 0.0334] | 0.1250 | yes |
| AxisII | P1 - P0 | SSIM_event_mean | 7 | 0.0015 | 0.0011 | [0.0009, 0.0024] | 0.0156 | yes |
| AxisII | P1 - P0 | CSI@5 | 4 | -0.0074 | -0.0078 | [-0.0145, -0.0004] | 0.6250 | yes |
| AxisII | P1 - P0 | POD@5 | 4 | -0.0194 | -0.0215 | [-0.0312, -0.0070] | 0.6250 | yes |
| AxisII | P1 - P0 | FAR@5 | 4 | 0.0194 | 0.0159 | [0.0132, 0.0285] | 0.1250 | yes |
| AxisII | P1 - P0 | HSS@5 | 4 | -0.0060 | -0.0056 | [-0.0121, 0.0002] | 0.6250 | yes |
| AxisII | P1 - P0 | ACC@5 | 7 | 0.0001 | 0.0000 | [-0.0000, 0.0002] | 1.0000 | yes |
| AxisII | P1 - P0 | BIAS@5 | 4 | -0.0465 | -0.0430 | [-0.0615, -0.0351] | 0.1250 | yes |
| AxisII | P1 - P0 | CSI@10 | 4 | -0.0052 | -0.0039 | [-0.0112, 0.0007] | 1.0000 | yes |
| AxisII | P1 - P0 | POD@10 | 4 | -0.0105 | -0.0110 | [-0.0198, -0.0012] | 0.6250 | yes |
| AxisII | P1 - P0 | FAR@10 | 3 | 0.0079 | 0.0063 | [0.0004, 0.0172] | 0.2500 | no |
| AxisII | P1 - P0 | HSS@10 | 4 | -0.0050 | -0.0044 | [-0.0107, 0.0007] | 1.0000 | yes |
| AxisII | P1 - P0 | ACC@10 | 7 | 0.0000 | 0.0000 | [-0.0000, 0.0001] | 1.0000 | yes |
| AxisII | P1 - P0 | BIAS@10 | 4 | -0.0222 | -0.0189 | [-0.0411, -0.0060] | 0.6250 | yes |
| AxisII | P1 - P0 | CSI@20 | 3 | 0.0190 | 0.0061 | [0.0005, 0.0503] | 0.2500 | no |
| AxisII | P1 - P0 | POD@20 | 3 | 0.0230 | 0.0107 | [-0.0021, 0.0605] | 1.0000 | no |
| AxisII | P1 - P0 | FAR@20 | 3 | 0.0067 | 0.0065 | [-0.0012, 0.0148] | 1.0000 | no |
| AxisII | P1 - P0 | HSS@20 | 3 | 0.0256 | 0.0072 | [0.0007, 0.0689] | 0.2500 | no |
| AxisII | P1 - P0 | ACC@20 | 7 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | yes |
| AxisII | P1 - P0 | BIAS@20 | 3 | 0.0298 | 0.0201 | [-0.0148, 0.0842] | 1.0000 | no |
| AxisII | P1 - P0 | CSI@30 | 3 | 0.0030 | 0.0012 | [0.0000, 0.0077] | 1.0000 | no |
| AxisII | P1 - P0 | POD@30 | 3 | 0.0047 | 0.0038 | [0.0000, 0.0104] | 1.0000 | no |
| AxisII | P1 - P0 | FAR@30 | 2 | -0.0073 | -0.0073 | [-0.0169, 0.0024] | 1.0000 | no |
| AxisII | P1 - P0 | HSS@30 | 3 | 0.0042 | 0.0018 | [0.0000, 0.0108] | 1.0000 | no |
| AxisII | P1 - P0 | ACC@30 | 7 | 0.0000 | 0.0000 | [-0.0000, 0.0000] | 1.0000 | yes |
| AxisII | P1 - P0 | BIAS@30 | 3 | 0.0117 | 0.0171 | [0.0000, 0.0180] | 1.0000 | no |
| AxisII | P2 - P0 | MAE_event | 7 | -0.2103 | -0.1342 | [-0.4139, -0.0654] | 0.0156 | yes |
| AxisII | P2 - P0 | RMSE_event | 7 | -0.3121 | -0.3257 | [-0.4691, -0.1695] | 0.0156 | yes |
| AxisII | P2 - P0 | SSIM_event_mean | 7 | -0.0778 | -0.0657 | [-0.1295, -0.0347] | 0.0156 | yes |
| AxisII | P2 - P0 | CSI@5 | 4 | -0.0629 | -0.0467 | [-0.1423, 0.0004] | 0.6250 | yes |
| AxisII | P2 - P0 | POD@5 | 4 | 0.1203 | 0.1416 | [0.0592, 0.1601] | 0.1250 | yes |
| AxisII | P2 - P0 | FAR@5 | 4 | -0.1767 | -0.1894 | [-0.2862, -0.0545] | 0.6250 | yes |
| AxisII | P2 - P0 | HSS@5 | 4 | -0.0604 | -0.0434 | [-0.1511, 0.0133] | 0.6250 | yes |
| AxisII | P2 - P0 | ACC@5 | 7 | -0.0085 | -0.0000 | [-0.0232, -0.0001] | 1.0000 | yes |
| AxisII | P2 - P0 | BIAS@5 | 4 | -0.4091 | -0.3000 | [-0.9789, 0.0516] | 0.6250 | yes |
| AxisII | P2 - P0 | CSI@10 | 4 | -0.0537 | -0.0543 | [-0.0918, -0.0151] | 0.6250 | yes |
| AxisII | P2 - P0 | POD@10 | 4 | 0.1693 | 0.2014 | [0.0610, 0.2590] | 0.6250 | yes |
| AxisII | P2 - P0 | FAR@10 | 3 | -0.2814 | -0.2799 | [-0.2880, -0.2764] | 0.2500 | no |
| AxisII | P2 - P0 | HSS@10 | 4 | -0.0577 | -0.0565 | [-0.1018, -0.0147] | 0.1250 | yes |
| AxisII | P2 - P0 | ACC@10 | 7 | -0.0027 | -0.0000 | [-0.0067, -0.0001] | 1.0000 | yes |
| AxisII | P2 - P0 | BIAS@10 | 4 | -0.2803 | -0.6548 | [-0.7683, 0.5687] | 0.6250 | yes |
| AxisII | P2 - P0 | CSI@20 | 3 | 0.0655 | 0.0351 | [0.0234, 0.1382] | 0.2500 | no |
| AxisII | P2 - P0 | POD@20 | 3 | 0.1811 | 0.1404 | [0.0983, 0.3045] | 0.2500 | no |
| AxisII | P2 - P0 | FAR@20 | 3 | -0.1565 | -0.1703 | [-0.2181, -0.0811] | 0.2500 | no |
| AxisII | P2 - P0 | HSS@20 | 3 | 0.0829 | 0.0458 | [0.0267, 0.1764] | 0.2500 | no |
| AxisII | P2 - P0 | ACC@20 | 7 | -0.0001 | 0.0000 | [-0.0003, -0.0000] | 1.0000 | yes |
| AxisII | P2 - P0 | BIAS@20 | 3 | 0.4913 | 0.5313 | [0.3076, 0.6350] | 0.2500 | no |
| AxisII | P2 - P0 | CSI@30 | 3 | 0.0354 | 0.0343 | [0.0000, 0.0719] | 1.0000 | no |
| AxisII | P2 - P0 | POD@30 | 3 | 0.0777 | 0.0889 | [0.0000, 0.1441] | 1.0000 | no |
| AxisII | P2 - P0 | FAR@30 | 2 | -0.1093 | -0.1093 | [-0.1425, -0.0761] | 0.5000 | no |
| AxisII | P2 - P0 | HSS@30 | 3 | 0.0482 | 0.0491 | [-0.0000, 0.0956] | 1.0000 | no |
| AxisII | P2 - P0 | ACC@30 | 7 | -0.0000 | 0.0000 | [-0.0001, -0.0000] | 1.0000 | yes |
| AxisII | P2 - P0 | BIAS@30 | 3 | 0.3381 | 0.3368 | [0.3077, 0.3698] | 0.2500 | no |
| AxisII | P3 - P0 | MAE_event | 7 | -0.0799 | -0.0070 | [-0.2224, 0.0074] | 1.0000 | yes |
| AxisII | P3 - P0 | RMSE_event | 7 | -0.1521 | -0.0793 | [-0.3116, -0.0317] | 0.4531 | yes |
| AxisII | P3 - P0 | SSIM_event_mean | 7 | -0.0241 | -0.0074 | [-0.0594, -0.0018] | 0.4531 | yes |
| AxisII | P3 - P0 | CSI@5 | 4 | -0.0516 | -0.0355 | [-0.1047, -0.0125] | 0.1250 | yes |
| AxisII | P3 - P0 | POD@5 | 4 | 0.0975 | 0.1237 | [0.0312, 0.1375] | 0.6250 | yes |
| AxisII | P3 - P0 | FAR@5 | 4 | -0.1688 | -0.1655 | [-0.2361, -0.1050] | 0.1250 | yes |
| AxisII | P3 - P0 | HSS@5 | 4 | -0.0528 | -0.0333 | [-0.1080, -0.0112] | 0.1250 | yes |
| AxisII | P3 - P0 | ACC@5 | 7 | -0.0059 | -0.0000 | [-0.0157, -0.0001] | 1.0000 | yes |
| AxisII | P3 - P0 | BIAS@5 | 4 | -0.2813 | -0.2094 | [-0.6916, 0.0571] | 0.6250 | yes |
| AxisII | P3 - P0 | CSI@10 | 4 | -0.0560 | -0.0551 | [-0.0992, -0.0138] | 0.6250 | yes |
| AxisII | P3 - P0 | POD@10 | 4 | 0.1708 | 0.2100 | [0.0598, 0.2513] | 0.6250 | yes |
| AxisII | P3 - P0 | FAR@10 | 3 | -0.2861 | -0.2883 | [-0.2916, -0.2784] | 0.2500 | no |
| AxisII | P3 - P0 | HSS@10 | 4 | -0.0609 | -0.0583 | [-0.1075, -0.0169] | 0.1250 | yes |
| AxisII | P3 - P0 | ACC@10 | 7 | -0.0030 | -0.0000 | [-0.0076, -0.0001] | 1.0000 | yes |
| AxisII | P3 - P0 | BIAS@10 | 4 | -0.4375 | -0.6473 | [-0.8603, 0.1837] | 0.6250 | yes |
| AxisII | P3 - P0 | CSI@20 | 3 | 0.0656 | 0.0373 | [0.0249, 0.1347] | 0.2500 | no |
| AxisII | P3 - P0 | POD@20 | 3 | 0.2062 | 0.1566 | [0.1251, 0.3369] | 0.2500 | no |
| AxisII | P3 - P0 | FAR@20 | 3 | -0.1778 | -0.1805 | [-0.2497, -0.1030] | 0.2500 | no |
| AxisII | P3 - P0 | HSS@20 | 3 | 0.0831 | 0.0486 | [0.0283, 0.1724] | 0.2500 | no |
| AxisII | P3 - P0 | ACC@20 | 7 | -0.0002 | 0.0000 | [-0.0005, -0.0000] | 1.0000 | yes |
| AxisII | P3 - P0 | BIAS@20 | 3 | 0.4126 | 0.4838 | [0.2462, 0.5078] | 0.2500 | no |
| AxisII | P3 - P0 | CSI@30 | 3 | 0.0360 | 0.0334 | [0.0000, 0.0747] | 1.0000 | no |
| AxisII | P3 - P0 | POD@30 | 3 | 0.0810 | 0.0899 | [0.0000, 0.1532] | 1.0000 | no |
| AxisII | P3 - P0 | FAR@30 | 2 | -0.1150 | -0.1150 | [-0.1484, -0.0817] | 0.5000 | no |
| AxisII | P3 - P0 | HSS@30 | 3 | 0.0489 | 0.0478 | [-0.0000, 0.0990] | 1.0000 | no |
| AxisII | P3 - P0 | ACC@30 | 7 | -0.0000 | 0.0000 | [-0.0001, -0.0000] | 1.0000 | yes |
| AxisII | P3 - P0 | BIAS@30 | 3 | 0.3518 | 0.3491 | [0.3077, 0.3985] | 0.2500 | no |

> Mean Δ > 0 ⇒ improvement against baseline. MAE_event / RMSE_event / FAR use baseline − candidate; SSIM_event_mean / CSI / POD / HSS / ACC use candidate − baseline; BIAS uses |BIAS_b − 1| − |BIAS_c − 1|. See `EVALUATION_PROTOCOL_V2.md` §17.2.

## 2. STATISTICAL SUMMARY

### 2.1 Holm-adjusted p-values (per `(metric, threshold)` family)

Only families with at least `HOLM_FAMILY_MIN_SIZE = 3` contrasts and at least `4` paired events are reported.

| Metric@τ | Family size | Contrast | p_raw | p_holm |
|---|---|---|---:|---:|
| ACC@5 | 6 | I3 - I2 | 1.0000 | 1.0000 |
| ACC@5 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| ACC@5 | 6 | I5 - I4 | 1.0000 | 1.0000 |
| ACC@5 | 6 | P1 - P0 | 1.0000 | 1.0000 |
| ACC@5 | 6 | P2 - P0 | 1.0000 | 1.0000 |
| ACC@5 | 6 | P3 - P0 | 1.0000 | 1.0000 |
| ACC@10 | 6 | I3 - I2 | 1.0000 | 1.0000 |
| ACC@10 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| ACC@10 | 6 | I5 - I4 | 1.0000 | 1.0000 |
| ACC@10 | 6 | P1 - P0 | 1.0000 | 1.0000 |
| ACC@10 | 6 | P2 - P0 | 1.0000 | 1.0000 |
| ACC@10 | 6 | P3 - P0 | 1.0000 | 1.0000 |
| ACC@20 | 6 | I3 - I2 | 1.0000 | 1.0000 |
| ACC@20 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| ACC@20 | 6 | I5 - I4 | 1.0000 | 1.0000 |
| ACC@20 | 6 | P1 - P0 | 1.0000 | 1.0000 |
| ACC@20 | 6 | P2 - P0 | 1.0000 | 1.0000 |
| ACC@20 | 6 | P3 - P0 | 1.0000 | 1.0000 |
| ACC@30 | 6 | I3 - I2 | 1.0000 | 1.0000 |
| ACC@30 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| ACC@30 | 6 | I5 - I4 | 1.0000 | 1.0000 |
| ACC@30 | 6 | P1 - P0 | 1.0000 | 1.0000 |
| ACC@30 | 6 | P2 - P0 | 1.0000 | 1.0000 |
| ACC@30 | 6 | P3 - P0 | 1.0000 | 1.0000 |
| BIAS@5 | 6 | I3 - I2 | 0.6250 | 1.0000 |
| BIAS@5 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| BIAS@5 | 6 | I5 - I4 | 0.6250 | 1.0000 |
| BIAS@5 | 6 | P1 - P0 | 0.1250 | 0.7500 |
| BIAS@5 | 6 | P2 - P0 | 0.6250 | 1.0000 |
| BIAS@5 | 6 | P3 - P0 | 0.6250 | 1.0000 |
| BIAS@10 | 6 | I3 - I2 | 1.0000 | 1.0000 |
| BIAS@10 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| BIAS@10 | 6 | I5 - I4 | 1.0000 | 1.0000 |
| BIAS@10 | 6 | P1 - P0 | 0.6250 | 1.0000 |
| BIAS@10 | 6 | P2 - P0 | 0.6250 | 1.0000 |
| BIAS@10 | 6 | P3 - P0 | 0.6250 | 1.0000 |
| CSI@5 | 6 | I3 - I2 | 0.1250 | 0.7500 |
| CSI@5 | 6 | I4 - I3 | 0.1250 | 0.7500 |
| CSI@5 | 6 | I5 - I4 | 0.6250 | 1.0000 |
| CSI@5 | 6 | P1 - P0 | 0.6250 | 1.0000 |
| CSI@5 | 6 | P2 - P0 | 0.6250 | 1.0000 |
| CSI@5 | 6 | P3 - P0 | 0.1250 | 0.7500 |
| CSI@10 | 6 | I3 - I2 | 0.6250 | 1.0000 |
| CSI@10 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| CSI@10 | 6 | I5 - I4 | 1.0000 | 1.0000 |
| CSI@10 | 6 | P1 - P0 | 1.0000 | 1.0000 |
| CSI@10 | 6 | P2 - P0 | 0.6250 | 1.0000 |
| CSI@10 | 6 | P3 - P0 | 0.6250 | 1.0000 |
| FAR@5 | 6 | I3 - I2 | 0.1250 | 0.7500 |
| FAR@5 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| FAR@5 | 6 | I5 - I4 | 0.6250 | 1.0000 |
| FAR@5 | 6 | P1 - P0 | 0.1250 | 0.7500 |
| FAR@5 | 6 | P2 - P0 | 0.6250 | 1.0000 |
| FAR@5 | 6 | P3 - P0 | 0.1250 | 0.7500 |
| HSS@5 | 6 | I3 - I2 | 0.1250 | 0.7500 |
| HSS@5 | 6 | I4 - I3 | 0.1250 | 0.7500 |
| HSS@5 | 6 | I5 - I4 | 0.6250 | 1.0000 |
| HSS@5 | 6 | P1 - P0 | 0.6250 | 1.0000 |
| HSS@5 | 6 | P2 - P0 | 0.6250 | 1.0000 |
| HSS@5 | 6 | P3 - P0 | 0.1250 | 0.7500 |
| HSS@10 | 6 | I3 - I2 | 0.6250 | 1.0000 |
| HSS@10 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| HSS@10 | 6 | I5 - I4 | 1.0000 | 1.0000 |
| HSS@10 | 6 | P1 - P0 | 1.0000 | 1.0000 |
| HSS@10 | 6 | P2 - P0 | 0.1250 | 0.7500 |
| HSS@10 | 6 | P3 - P0 | 0.1250 | 0.7500 |
| MAE_event | 6 | I3 - I2 | 0.1250 | 0.5000 |
| MAE_event | 6 | I4 - I3 | 0.4531 | 1.0000 |
| MAE_event | 6 | I5 - I4 | 1.0000 | 1.0000 |
| MAE_event | 6 | P1 - P0 | 0.0156 | 0.0938 |
| MAE_event | 6 | P2 - P0 | 0.0156 | 0.0938 |
| MAE_event | 6 | P3 - P0 | 1.0000 | 1.0000 |
| POD@5 | 6 | I3 - I2 | 1.0000 | 1.0000 |
| POD@5 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| POD@5 | 6 | I5 - I4 | 1.0000 | 1.0000 |
| POD@5 | 6 | P1 - P0 | 0.6250 | 1.0000 |
| POD@5 | 6 | P2 - P0 | 0.1250 | 0.7500 |
| POD@5 | 6 | P3 - P0 | 0.6250 | 1.0000 |
| POD@10 | 6 | I3 - I2 | 1.0000 | 1.0000 |
| POD@10 | 6 | I4 - I3 | 1.0000 | 1.0000 |
| POD@10 | 6 | I5 - I4 | 1.0000 | 1.0000 |
| POD@10 | 6 | P1 - P0 | 0.6250 | 1.0000 |
| POD@10 | 6 | P2 - P0 | 0.6250 | 1.0000 |
| POD@10 | 6 | P3 - P0 | 0.6250 | 1.0000 |
| RMSE_event | 6 | I3 - I2 | 0.4531 | 1.0000 |
| RMSE_event | 6 | I4 - I3 | 1.0000 | 1.0000 |
| RMSE_event | 6 | I5 - I4 | 0.4531 | 1.0000 |
| RMSE_event | 6 | P1 - P0 | 0.1250 | 0.6250 |
| RMSE_event | 6 | P2 - P0 | 0.0156 | 0.0938 |
| RMSE_event | 6 | P3 - P0 | 0.4531 | 1.0000 |
| SSIM_event_mean | 6 | I3 - I2 | 0.1250 | 0.5000 |
| SSIM_event_mean | 6 | I4 - I3 | 0.4531 | 0.9062 |
| SSIM_event_mean | 6 | I5 - I4 | 0.1250 | 0.5000 |
| SSIM_event_mean | 6 | P1 - P0 | 0.0156 | 0.0938 |
| SSIM_event_mean | 6 | P2 - P0 | 0.0156 | 0.0938 |
| SSIM_event_mean | 6 | P3 - P0 | 0.4531 | 0.9062 |

### 2.2 Bootstrap controls

- n_bootstrap = 10000, seed = 42.
- 95% CI is the equal-event percentile interval of the resampled means.
- Resampling unit is the **typhoon event**, never the window.

## 3. INTERPRETATION LIMIT

- **Inferential power is bounded by `n_event = 7` on validation.** The current matrix is not a substitute for the held-out test.
- **Single-seed training.** All inferences are conditioned on seed 42. Initialization robustness is out of scope for this matrix.
- **Backbone contrast vs. information effect.** `I1 − I2` changes the model family; it is reported as a backbone sanity check, NEVER as an information claim. Information claims use `I3 − I2`, `I4 − I3`, `I5 − I4` only.
- **Inductive-bias claims.** `P1 − P0`, `P2 − P0`, `P3 − P0` are the loss/regularization contrasts. The interaction `P3 − P1 − P2 + P0` is exploratory at one seed.
- **Geometric resolution.** GPM 0.1° (~10 km) reprojection smooths sub-grid-scale terrain variation. A negative `I5 − I4` does NOT contradict the underlying physics; it is a controlled report of what the smoother version carries at this resolution.
- **Single categorical measure.** Confidence intervals for categorical metrics are reported at the listed thresholds; no general threshold transfer claim is implied.

## 4. NOT YET A TEST-SET CONCLUSION

- The held-out test events (4 typhoons, 707 windows) are **SEALED**. No inference in this document is a held-out test result.
- A test evaluation is permitted only after every item in `docs/FINAL_TEST_AUTHORIZATION.md` §0 is satisfied, with an append-only authorization recorded in §3 of that document.
- This analysis does not predict terrain information gain, smoothing-vs-extreme behavior, or any generalization beyond the `n_event = 7` paired observations used here.

---

Generated by `scripts/analyze_ablation_results.py`. Output filenames and statistical controls are frozen by this script's contract; do not edit the resulting tables by hand outside of an explicit re-run.