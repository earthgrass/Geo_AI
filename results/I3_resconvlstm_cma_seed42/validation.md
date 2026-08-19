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
| MAE_global | 0.240324 |
| RMSE_global | 1.246955 |
| peak_error_global | 28.543831 |
| 10mmh.ACC | 0.994934 |
| 10mmh.BIAS | 0.853212 |
| 10mmh.CSI | 0.424698 |
| 10mmh.FAR | 0.352521 |
| 10mmh.HSS | 0.593660 |
| 10mmh.POD | 0.552436 |
| 10mmh.a_hits | 77577 |
| 10mmh.b_false_alarms | 42237 |
| 10mmh.c_misses | 62850 |
| 10mmh.d_correct_negatives | 20559480 |
| 10mmh.n_total | 20742144 |
| 20mmh.ACC | 0.998183 |
| 20mmh.BIAS | 0.630944 |
| 20mmh.CSI | 0.292294 |
| 20mmh.FAR | 0.415336 |
| 20mmh.HSS | 0.451501 |
| 20mmh.POD | 0.368890 |
| 20mmh.a_hits | 15562 |
| 20mmh.b_false_alarms | 11055 |
| 20mmh.c_misses | 26624 |
| 20mmh.d_correct_negatives | 20688903 |
| 20mmh.n_total | 20742144 |
| 30mmh.ACC | 0.999119 |
| 30mmh.BIAS | 0.344287 |
| 30mmh.CSI | 0.166165 |
| 30mmh.FAR | 0.443647 |
| 30mmh.HSS | 0.284641 |
| 30mmh.POD | 0.191545 |
| 30mmh.a_hits | 3643 |
| 30mmh.b_false_alarms | 2905 |
| 30mmh.c_misses | 15376 |
| 30mmh.d_correct_negatives | 20720220 |
| 30mmh.n_total | 20742144 |
| 5mmh.ACC | 0.987457 |
| 5mmh.BIAS | 0.945663 |
| 5mmh.CSI | 0.524394 |
| 5mmh.FAR | 0.292231 |
| 5mmh.HSS | 0.681608 |
| 5mmh.POD | 0.669311 |
| 5mmh.a_hits | 286846 |
| 5mmh.b_false_alarms | 118436 |
| 5mmh.c_misses | 141723 |
| 5mmh.d_correct_negatives | 20195139 |
| 5mmh.n_total | 20742144 |

### Per-event validation metrics (pooled within event)

| typhoon_id | MAE_event | RMSE_event | SSIM_event_mean | CSI_5mmh | POD_5mmh | FAR_5mmh | HSS_5mmh | ACC_5mmh | BIAS_5mmh | CSI_10mmh | POD_10mmh | FAR_10mmh | HSS_10mmh | ACC_10mmh | BIAS_10mmh | CSI_20mmh | POD_20mmh | FAR_20mmh | HSS_20mmh | ACC_20mmh | BIAS_20mmh | CSI_30mmh | POD_30mmh | FAR_30mmh | HSS_30mmh | ACC_30mmh | BIAS_30mmh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2203 | 0.581814 | 2.105741 | 0.915012 | 0.533209 | 0.685280 | 0.293874 | 0.678459 | 0.967633 | 0.970478 | 0.435334 | 0.568723 | 0.350126 | 0.599633 | 0.986254 | 0.875127 | 0.299312 | 0.380361 | 0.415860 | 0.458194 | 0.994730 | 0.651147 | 0.168412 | 0.194581 | 0.444002 | 0.287253 | 0.997344 | 0.349967 |
| 2205 | 0.268565 | 0.964047 | 0.952928 | 0.469196 | 0.583962 | 0.295211 | 0.631898 | 0.986508 | 0.828562 | 0.327151 | 0.419428 | 0.402090 | 0.491023 | 0.995910 | 0.701490 | 0.206341 | 0.243072 | 0.422747 | 0.341728 | 0.999123 | 0.421084 | 0.130213 | 0.144749 | 0.435424 | 0.230328 | 0.999711 | 0.256386 |
| 2208 | 0.029428 | 0.048966 | 0.997663 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2210 | 0.014767 | 0.040178 | 0.998460 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2212 | 0.016987 | 0.052017 | 0.998621 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2214 | 0.018634 | 0.056394 | 0.998216 | 0.029412 | 0.031250 | 0.666667 | 0.057141 | 0.999989 | 0.093750 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999999 | 0.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2217 | 0.068594 | 0.352953 | 0.991289 | 0.666760 | 0.765766 | 0.162414 | 0.799201 | 0.998271 | 0.914254 | 0.555521 | 0.648580 | 0.205266 | 0.713744 | 0.998965 | 0.816096 | 0.280303 | 0.319654 | 0.305164 | 0.437751 | 0.999724 | 0.460043 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999981 | 0.000000 |

### Window diagnostics (mean of per-window values)

| Metric | Value |
|---|---|
| MAE_window_mean | 0.240324 |
| RMSE_window_mean | 0.706689 |
| SSIM_window_mean | 0.964061 |
| peak_error_window_mean | 7.628709 |
| n_defined_windows | 1266 |

- n_negative_roundoff_clamped: 0