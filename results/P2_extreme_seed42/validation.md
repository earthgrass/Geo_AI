## Experiment: ResConvLSTM

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
| MAE_global | 0.562080 |
| RMSE_global | 1.687437 |
| peak_error_global | 16.584038 |
| 10mmh.ACC | 0.989747 |
| 10mmh.BIAS | 1.950992 |
| 10mmh.CSI | 0.321716 |
| 10mmh.FAR | 0.631831 |
| 10mmh.HSS | 0.482180 |
| 10mmh.POD | 0.718295 |
| 10mmh.a_hits | 100868 |
| 10mmh.b_false_alarms | 173104 |
| 10mmh.c_misses | 39559 |
| 10mmh.d_correct_negatives | 20428613 |
| 10mmh.n_total | 20742144 |
| 20mmh.ACC | 0.997924 |
| 20mmh.BIAS | 0.980847 |
| 20mmh.CSI | 0.319960 |
| 20mmh.FAR | 0.510464 |
| 20mmh.HSS | 0.483762 |
| 20mmh.POD | 0.480159 |
| 20mmh.a_hits | 20256 |
| 20mmh.b_false_alarms | 21122 |
| 20mmh.c_misses | 21930 |
| 20mmh.d_correct_negatives | 20678836 |
| 20mmh.n_total | 20742144 |
| 30mmh.ACC | 0.999041 |
| 30mmh.BIAS | 0.771439 |
| 30mmh.CSI | 0.257455 |
| 30mmh.FAR | 0.529853 |
| 30mmh.HSS | 0.409014 |
| 30mmh.POD | 0.362690 |
| 30mmh.a_hits | 6898 |
| 30mmh.b_false_alarms | 7774 |
| 30mmh.c_misses | 12121 |
| 30mmh.d_correct_negatives | 20715351 |
| 30mmh.n_total | 20742144 |
| 5mmh.ACC | 0.970058 |
| 5mmh.BIAS | 2.115151 |
| 5mmh.CSI | 0.365002 |
| 5mmh.FAR | 0.606178 |
| 5mmh.HSS | 0.521372 |
| 5mmh.POD | 0.832993 |
| 5mmh.a_hits | 356995 |
| 5mmh.b_false_alarms | 549493 |
| 5mmh.c_misses | 71574 |
| 5mmh.d_correct_negatives | 19764082 |
| 5mmh.n_total | 20742144 |

### Per-event validation metrics (pooled within event)

| typhoon_id | MAE_event | RMSE_event | SSIM_event_mean | CSI_5mmh | POD_5mmh | FAR_5mmh | HSS_5mmh | ACC_5mmh | BIAS_5mmh | CSI_10mmh | POD_10mmh | FAR_10mmh | HSS_10mmh | ACC_10mmh | BIAS_10mmh | CSI_20mmh | POD_20mmh | FAR_20mmh | HSS_20mmh | ACC_20mmh | BIAS_20mmh | CSI_30mmh | POD_30mmh | FAR_30mmh | HSS_30mmh | ACC_30mmh | BIAS_30mmh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2203 | 1.385453 | 2.830050 | 0.696375 | 0.355385 | 0.852907 | 0.621411 | 0.485992 | 0.916534 | 2.252860 | 0.325897 | 0.723133 | 0.627642 | 0.478764 | 0.972128 | 1.942034 | 0.326107 | 0.486889 | 0.503136 | 0.488831 | 0.994045 | 0.979923 | 0.261226 | 0.368142 | 0.526463 | 0.412821 | 0.997122 | 0.777431 |
| 2205 | 0.515962 | 1.309970 | 0.852629 | 0.409808 | 0.731381 | 0.517581 | 0.570804 | 0.978488 | 1.516070 | 0.274246 | 0.654529 | 0.679338 | 0.426796 | 0.991787 | 2.041180 | 0.252431 | 0.398795 | 0.592490 | 0.402551 | 0.998892 | 0.978614 | 0.199055 | 0.279092 | 0.590278 | 0.331859 | 0.999665 | 0.681173 |
| 2208 | 0.182379 | 0.450018 | 0.913472 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2210 | 0.029294 | 0.121926 | 0.989842 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2212 | 0.140542 | 0.372170 | 0.937876 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2214 | 0.024491 | 0.113405 | 0.991977 | 0.047619 | 0.062500 | 0.833333 | 0.090904 | 0.999987 | 0.375000 | 0.000000 | 0.000000 | 1.000000 | -0.000001 | 0.999999 | 1.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2217 | 0.194508 | 0.618251 | 0.926800 | 0.623783 | 0.897040 | 0.328112 | 0.767106 | 0.997556 | 1.335103 | 0.471434 | 0.895484 | 0.501116 | 0.639859 | 0.997997 | 1.794975 | 0.322314 | 0.505400 | 0.529175 | 0.487321 | 0.999643 | 1.073434 | 0.000000 | 0.000000 | 1.000000 | -0.000009 | 0.999975 | 0.307692 |

### Window diagnostics (mean of per-window values)

| Metric | Value |
|---|---|
| MAE_window_mean | 0.562080 |
| RMSE_window_mean | 1.134885 |
| SSIM_window_mean | 0.863438 |
| peak_error_window_mean | 5.153300 |
| n_defined_windows | 1266 |

- n_negative_roundoff_clamped: 0