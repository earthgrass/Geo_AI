## Experiment: PlainConvLSTM

- protocol_id: evaluation_v2
- split: val
- test_status: SEALED
- n_events: 7
- n_windows: 1266
- n_pixels: 20742144
- mode: train
- smoke: False
- n_val_windows: 1266

### Overall validation metrics (pooled / protocol v2)

| Metric | Value |
|---|---|
| MAE_global | 0.430058 |
| RMSE_global | 2.130754 |
| peak_error_global | 94.401100 |
| 10mmh.ACC | 0.993230 |
| 10mmh.BIAS | 0.000000 |
| 10mmh.CSI | 0.000000 |
| 10mmh.FAR | nan |
| 10mmh.HSS | 0.000000 |
| 10mmh.POD | 0.000000 |
| 10mmh.a_hits | 0 |
| 10mmh.b_false_alarms | 0 |
| 10mmh.c_misses | 140427 |
| 10mmh.d_correct_negatives | 20601717 |
| 10mmh.n_total | 20742144 |
| 20mmh.ACC | 0.997966 |
| 20mmh.BIAS | 0.000000 |
| 20mmh.CSI | 0.000000 |
| 20mmh.FAR | nan |
| 20mmh.HSS | 0.000000 |
| 20mmh.POD | 0.000000 |
| 20mmh.a_hits | 0 |
| 20mmh.b_false_alarms | 0 |
| 20mmh.c_misses | 42186 |
| 20mmh.d_correct_negatives | 20699958 |
| 20mmh.n_total | 20742144 |
| 30mmh.ACC | 0.999083 |
| 30mmh.BIAS | 0.000000 |
| 30mmh.CSI | 0.000000 |
| 30mmh.FAR | nan |
| 30mmh.HSS | 0.000000 |
| 30mmh.POD | 0.000000 |
| 30mmh.a_hits | 0 |
| 30mmh.b_false_alarms | 0 |
| 30mmh.c_misses | 19019 |
| 30mmh.d_correct_negatives | 20723125 |
| 30mmh.n_total | 20742144 |
| 5mmh.ACC | 0.979338 |
| 5mmh.BIAS | 0.000000 |
| 5mmh.CSI | 0.000000 |
| 5mmh.FAR | nan |
| 5mmh.HSS | 0.000000 |
| 5mmh.POD | 0.000000 |
| 5mmh.a_hits | 0 |
| 5mmh.b_false_alarms | 0 |
| 5mmh.c_misses | 428569 |
| 5mmh.d_correct_negatives | 20313575 |
| 5mmh.n_total | 20742144 |

### Per-event validation metrics (pooled within event)

| typhoon_id | MAE_event | RMSE_event | SSIM_event_mean | CSI_5mmh | POD_5mmh | FAR_5mmh | HSS_5mmh | ACC_5mmh | BIAS_5mmh | CSI_10mmh | POD_10mmh | FAR_10mmh | HSS_10mmh | ACC_10mmh | BIAS_10mmh | CSI_20mmh | POD_20mmh | FAR_20mmh | HSS_20mmh | ACC_20mmh | BIAS_20mmh | CSI_30mmh | POD_30mmh | FAR_30mmh | HSS_30mmh | ACC_30mmh | BIAS_30mmh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2203 | 1.094666 | 3.571599 | 0.725144 | 0.000000 | 0.000000 | nan | 0.000000 | 0.946048 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.981366 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.994081 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.997236 | 0.000000 |
| 2205 | 0.476873 | 1.716042 | 0.846902 | 0.000000 | 0.000000 | nan | 0.000000 | 0.979577 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.995258 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999062 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999701 | 0.000000 |
| 2208 | 0.000000 | 0.000000 | 1.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2210 | 0.000000 | 0.000000 | 1.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2212 | 0.000000 | 0.000000 | 1.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2214 | 0.002356 | 0.051224 | 0.999400 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999989 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999999 | 0.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2217 | 0.089564 | 0.843148 | 0.977073 | 0.000000 | 0.000000 | nan | 0.000000 | 0.995483 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.998005 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999664 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999981 | 0.000000 |

### Window diagnostics (mean of per-window values)

| Metric | Value |
|---|---|
| MAE_window_mean | 0.430058 |
| RMSE_window_mean | 1.253459 |
| SSIM_window_mean | 0.886296 |
| peak_error_window_mean | 18.142125 |
| n_defined_windows | 1266 |

- n_negative_roundoff_clamped: 0