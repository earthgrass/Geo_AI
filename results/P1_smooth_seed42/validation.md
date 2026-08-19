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
| MAE_global | 0.222160 |
| RMSE_global | 1.251911 |
| peak_error_global | 30.539848 |
| 10mmh.ACC | 0.994958 |
| 10mmh.BIAS | 0.828808 |
| 10mmh.CSI | 0.421257 |
| 10mmh.FAR | 0.345984 |
| 10mmh.HSS | 0.590281 |
| 10mmh.POD | 0.542054 |
| 10mmh.a_hits | 76119 |
| 10mmh.b_false_alarms | 40268 |
| 10mmh.c_misses | 64308 |
| 10mmh.d_correct_negatives | 20561449 |
| 10mmh.n_total | 20742144 |
| 20mmh.ACC | 0.998179 |
| 20mmh.BIAS | 0.668445 |
| 20mmh.CSI | 0.301546 |
| 20mmh.FAR | 0.421717 |
| 20mmh.HSS | 0.462490 |
| 20mmh.POD | 0.386550 |
| 20mmh.a_hits | 16307 |
| 20mmh.b_false_alarms | 11892 |
| 20mmh.c_misses | 25879 |
| 20mmh.d_correct_negatives | 20688066 |
| 20mmh.n_total | 20742144 |
| 30mmh.ACC | 0.999123 |
| 30mmh.BIAS | 0.420737 |
| 30mmh.CSI | 0.195091 |
| 30mmh.FAR | 0.448763 |
| 30mmh.HSS | 0.326121 |
| 30mmh.POD | 0.231926 |
| 30mmh.a_hits | 4411 |
| 30mmh.b_false_alarms | 3591 |
| 30mmh.c_misses | 14608 |
| 30mmh.d_correct_negatives | 20719534 |
| 30mmh.n_total | 20742144 |
| 5mmh.ACC | 0.987619 |
| 5mmh.BIAS | 0.903350 |
| 5mmh.CSI | 0.521119 |
| 5mmh.FAR | 0.278168 |
| 5mmh.HSS | 0.678881 |
| 5mmh.POD | 0.652068 |
| 5mmh.a_hits | 279456 |
| 5mmh.b_false_alarms | 107692 |
| 5mmh.c_misses | 149113 |
| 5mmh.d_correct_negatives | 20205883 |
| 5mmh.n_total | 20742144 |

### Per-event validation metrics (pooled within event)

| typhoon_id | MAE_event | RMSE_event | SSIM_event_mean | CSI_5mmh | POD_5mmh | FAR_5mmh | HSS_5mmh | ACC_5mmh | BIAS_5mmh | CSI_10mmh | POD_10mmh | FAR_10mmh | HSS_10mmh | ACC_10mmh | BIAS_10mmh | CSI_20mmh | POD_20mmh | FAR_20mmh | HSS_20mmh | ACC_20mmh | BIAS_20mmh | CSI_30mmh | POD_30mmh | FAR_30mmh | HSS_30mmh | ACC_30mmh | BIAS_30mmh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2203 | 0.563629 | 2.115929 | 0.916109 | 0.532406 | 0.671033 | 0.279553 | 0.678113 | 0.968204 | 0.931411 | 0.433467 | 0.561684 | 0.344957 | 0.597863 | 0.986321 | 0.857476 | 0.308818 | 0.399318 | 0.423258 | 0.469334 | 0.994710 | 0.692368 | 0.197012 | 0.234500 | 0.447959 | 0.328065 | 0.997358 | 0.424788 |
| 2205 | 0.249895 | 0.962324 | 0.954924 | 0.454962 | 0.553883 | 0.281894 | 0.618610 | 0.986448 | 0.771310 | 0.314812 | 0.391001 | 0.382320 | 0.476947 | 0.995965 | 0.633015 | 0.217926 | 0.256325 | 0.407382 | 0.357500 | 0.999137 | 0.432530 | 0.165992 | 0.193945 | 0.464752 | 0.284609 | 0.999709 | 0.362346 |
| 2208 | 0.000087 | 0.002091 | 0.999999 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2210 | 0.000430 | 0.005184 | 0.999986 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2212 | 0.000699 | 0.011762 | 0.999989 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2214 | 0.001792 | 0.038715 | 0.999738 | 0.027778 | 0.031250 | 0.800000 | 0.054051 | 0.999989 | 0.156250 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999999 | 0.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2217 | 0.042494 | 0.360109 | 0.993352 | 0.643151 | 0.728925 | 0.154664 | 0.781914 | 0.998173 | 0.862291 | 0.517231 | 0.601238 | 0.212685 | 0.681258 | 0.998880 | 0.763656 | 0.234496 | 0.261339 | 0.304598 | 0.379792 | 0.999713 | 0.375810 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999981 | 0.000000 |

### Window diagnostics (mean of per-window values)

| Metric | Value |
|---|---|
| MAE_window_mean | 0.222160 |
| RMSE_window_mean | 0.692272 |
| SSIM_window_mean | 0.965586 |
| peak_error_window_mean | 7.808270 |
| n_defined_windows | 1266 |

- n_negative_roundoff_clamped: 0