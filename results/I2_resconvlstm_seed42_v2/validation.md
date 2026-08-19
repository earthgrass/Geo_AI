## Experiment: ResConvLSTM

- protocol_id: evaluation_v2
- split: val
- test_status: SEALED
- n_events: 7
- n_windows: 1266
- n_pixels: 20742144
- mode: validate-only
- smoke: False
- n_val_windows: 1266

### Overall validation metrics (pooled / protocol v2)

| Metric | Value |
|---|---|
| MAE_global | 0.237219 |
| RMSE_global | 1.264393 |
| peak_error_global | 31.602982 |
| 10mmh.ACC | 0.994758 |
| 10mmh.BIAS | 0.884865 |
| 10mmh.CSI | 0.417616 |
| 10mmh.FAR | 0.372488 |
| 10mmh.HSS | 0.586553 |
| 10mmh.POD | 0.555264 |
| 10mmh.a_hits | 77974 |
| 10mmh.b_false_alarms | 46285 |
| 10mmh.c_misses | 62453 |
| 10mmh.d_correct_negatives | 20555432 |
| 10mmh.n_total | 20742144 |
| 20mmh.ACC | 0.998141 |
| 20mmh.BIAS | 0.662068 |
| 20mmh.CSI | 0.290320 |
| 20mmh.FAR | 0.435159 |
| 20mmh.HSS | 0.449105 |
| 20mmh.POD | 0.373963 |
| 20mmh.a_hits | 15776 |
| 20mmh.b_false_alarms | 12154 |
| 20mmh.c_misses | 26410 |
| 20mmh.d_correct_negatives | 20687804 |
| 20mmh.n_total | 20742144 |
| 30mmh.ACC | 0.999113 |
| 30mmh.BIAS | 0.390715 |
| 30mmh.CSI | 0.179382 |
| 30mmh.FAR | 0.458619 |
| 30mmh.HSS | 0.303838 |
| 30mmh.POD | 0.211525 |
| 30mmh.a_hits | 4023 |
| 30mmh.b_false_alarms | 3408 |
| 30mmh.c_misses | 14996 |
| 30mmh.d_correct_negatives | 20719717 |
| 30mmh.n_total | 20742144 |
| 5mmh.ACC | 0.986997 |
| 5mmh.BIAS | 0.974856 |
| 5mmh.CSI | 0.516691 |
| 5mmh.FAR | 0.309874 |
| 5mmh.HSS | 0.674704 |
| 5mmh.POD | 0.672774 |
| 5mmh.a_hits | 288330 |
| 5mmh.b_false_alarms | 129463 |
| 5mmh.c_misses | 140239 |
| 5mmh.d_correct_negatives | 20184112 |
| 5mmh.n_total | 20742144 |

### Per-event validation metrics (pooled within event)

| typhoon_id | MAE_event | RMSE_event | SSIM_event_mean | CSI_5mmh | POD_5mmh | FAR_5mmh | HSS_5mmh | ACC_5mmh | BIAS_5mmh | CSI_10mmh | POD_10mmh | FAR_10mmh | HSS_10mmh | ACC_10mmh | BIAS_10mmh | CSI_20mmh | POD_20mmh | FAR_20mmh | HSS_20mmh | ACC_20mmh | BIAS_20mmh | CSI_30mmh | POD_30mmh | FAR_30mmh | HSS_30mmh | ACC_30mmh | BIAS_30mmh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2203 | 0.600649 | 2.138749 | 0.911648 | 0.526277 | 0.689202 | 0.309958 | 0.671932 | 0.966529 | 0.998783 | 0.429389 | 0.574355 | 0.370202 | 0.593576 | 0.985777 | 0.911968 | 0.296284 | 0.384970 | 0.437422 | 0.454505 | 0.994588 | 0.684295 | 0.179757 | 0.212199 | 0.459605 | 0.303650 | 0.997323 | 0.392674 |
| 2205 | 0.268486 | 0.963852 | 0.953913 | 0.460249 | 0.588389 | 0.321197 | 0.623225 | 0.985907 | 0.866804 | 0.318587 | 0.413707 | 0.419177 | 0.481178 | 0.995804 | 0.712277 | 0.230671 | 0.273193 | 0.402897 | 0.374502 | 0.999145 | 0.457530 | 0.176710 | 0.205298 | 0.440722 | 0.300234 | 0.999714 | 0.367077 |
| 2208 | 0.000413 | 0.004091 | 0.999999 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2210 | 0.000413 | 0.004091 | 0.999999 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2212 | 0.000413 | 0.004091 | 0.999999 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2214 | 0.002360 | 0.038788 | 0.999753 | 0.000000 | 0.000000 | 1.000000 | -0.000003 | 0.999988 | 0.187500 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999999 | 0.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2217 | 0.046093 | 0.371410 | 0.993043 | 0.633719 | 0.732304 | 0.175213 | 0.774842 | 0.998088 | 0.887870 | 0.497191 | 0.580117 | 0.223306 | 0.663591 | 0.998829 | 0.746905 | 0.166667 | 0.183585 | 0.356061 | 0.285608 | 0.999691 | 0.285097 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999981 | 0.000000 |

### Window diagnostics (mean of per-window values)

| Metric | Value |
|---|---|
| MAE_window_mean | 0.237219 |
| RMSE_window_mean | 0.700905 |
| SSIM_window_mean | 0.964002 |
| peak_error_window_mean | 7.574324 |
| n_defined_windows | 1266 |

- n_negative_roundoff_clamped: 0