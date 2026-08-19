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
| MAE_global | 0.245161 |
| RMSE_global | 1.251270 |
| peak_error_global | 27.723694 |
| 10mmh.ACC | 0.994785 |
| 10mmh.BIAS | 0.909234 |
| 10mmh.CSI | 0.425060 |
| 10mmh.FAR | 0.373673 |
| 10mmh.HSS | 0.593932 |
| 10mmh.POD | 0.569477 |
| 10mmh.a_hits | 79970 |
| 10mmh.b_false_alarms | 47711 |
| 10mmh.c_misses | 60457 |
| 10mmh.d_correct_negatives | 20554006 |
| 10mmh.n_total | 20742144 |
| 20mmh.ACC | 0.998158 |
| 20mmh.BIAS | 0.656474 |
| 20mmh.CSI | 0.293068 |
| 20mmh.FAR | 0.428107 |
| 20mmh.HSS | 0.452409 |
| 20mmh.POD | 0.375433 |
| 20mmh.a_hits | 15838 |
| 20mmh.b_false_alarms | 11856 |
| 20mmh.c_misses | 26348 |
| 20mmh.d_correct_negatives | 20688102 |
| 20mmh.n_total | 20742144 |
| 30mmh.ACC | 0.999113 |
| 30mmh.BIAS | 0.319680 |
| 30mmh.CSI | 0.154242 |
| 30mmh.FAR | 0.448355 |
| 30mmh.HSS | 0.266936 |
| 30mmh.POD | 0.176350 |
| 30mmh.a_hits | 3354 |
| 30mmh.b_false_alarms | 2726 |
| 30mmh.c_misses | 15665 |
| 30mmh.d_correct_negatives | 20720399 |
| 30mmh.n_total | 20742144 |
| 5mmh.ACC | 0.987189 |
| 5mmh.BIAS | 0.974385 |
| 5mmh.CSI | 0.522027 |
| 5mmh.FAR | 0.305021 |
| 5mmh.HSS | 0.679425 |
| 5mmh.POD | 0.677177 |
| 5mmh.a_hits | 290217 |
| 5mmh.b_false_alarms | 127374 |
| 5mmh.c_misses | 138352 |
| 5mmh.d_correct_negatives | 20186201 |
| 5mmh.n_total | 20742144 |

### Per-event validation metrics (pooled within event)

| typhoon_id | MAE_event | RMSE_event | SSIM_event_mean | CSI_5mmh | POD_5mmh | FAR_5mmh | HSS_5mmh | ACC_5mmh | BIAS_5mmh | CSI_10mmh | POD_10mmh | FAR_10mmh | HSS_10mmh | ACC_10mmh | BIAS_10mmh | CSI_20mmh | POD_20mmh | FAR_20mmh | HSS_20mmh | ACC_20mmh | BIAS_20mmh | CSI_30mmh | POD_30mmh | FAR_30mmh | HSS_30mmh | ACC_30mmh | BIAS_30mmh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2203 | 0.607525 | 2.114984 | 0.912116 | 0.531539 | 0.697861 | 0.309573 | 0.676581 | 0.966817 | 1.010767 | 0.436633 | 0.590948 | 0.374239 | 0.600627 | 0.985792 | 0.944367 | 0.302332 | 0.391662 | 0.430006 | 0.461698 | 0.994650 | 0.687134 | 0.155967 | 0.178802 | 0.450197 | 0.268855 | 0.997325 | 0.325212 |
| 2205 | 0.266745 | 0.958290 | 0.955426 | 0.463226 | 0.572257 | 0.291436 | 0.626338 | 0.986457 | 0.807629 | 0.319547 | 0.406555 | 0.401106 | 0.482343 | 0.995895 | 0.678844 | 0.198056 | 0.227108 | 0.392425 | 0.330287 | 0.999137 | 0.373795 | 0.127053 | 0.139073 | 0.404858 | 0.225372 | 0.999715 | 0.233680 |
| 2208 | 0.006139 | 0.019082 | 0.999817 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2210 | 0.007871 | 0.021724 | 0.999689 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2212 | 0.032476 | 0.085449 | 0.996104 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2214 | 0.012947 | 0.048346 | 0.999102 | 0.027778 | 0.031250 | 0.800000 | 0.054051 | 0.999989 | 0.156250 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999999 | 0.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2217 | 0.049675 | 0.363057 | 0.993222 | 0.649297 | 0.735682 | 0.153148 | 0.786466 | 0.998205 | 0.868726 | 0.538924 | 0.620175 | 0.195560 | 0.699869 | 0.998941 | 0.770940 | 0.088660 | 0.092873 | 0.338462 | 0.162809 | 0.999679 | 0.140389 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999981 | 0.000000 |

### Window diagnostics (mean of per-window values)

| Metric | Value |
|---|---|
| MAE_window_mean | 0.245161 |
| RMSE_window_mean | 0.706187 |
| SSIM_window_mean | 0.963928 |
| peak_error_window_mean | 7.776965 |
| n_defined_windows | 1266 |

- n_negative_roundoff_clamped: 0