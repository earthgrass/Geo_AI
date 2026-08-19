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
| MAE_global | 0.410866 |
| RMSE_global | 1.594965 |
| peak_error_global | 16.107910 |
| 10mmh.ACC | 0.989128 |
| 10mmh.BIAS | 2.078489 |
| 10mmh.CSI | 0.314378 |
| 10mmh.FAR | 0.645740 |
| 10mmh.HSS | 0.473555 |
| 10mmh.POD | 0.736326 |
| 10mmh.a_hits | 103400 |
| 10mmh.b_false_alarms | 188476 |
| 10mmh.c_misses | 37027 |
| 10mmh.d_correct_negatives | 20413241 |
| 10mmh.n_total | 20742144 |
| 20mmh.ACC | 0.997828 |
| 20mmh.BIAS | 1.080216 |
| 20mmh.CSI | 0.321587 |
| 20mmh.FAR | 0.531402 |
| 20mmh.HSS | 0.485581 |
| 20mmh.POD | 0.506187 |
| 20mmh.a_hits | 21354 |
| 20mmh.b_false_alarms | 24216 |
| 20mmh.c_misses | 20832 |
| 20mmh.d_correct_negatives | 20675742 |
| 20mmh.n_total | 20742144 |
| 30mmh.ACC | 0.999031 |
| 30mmh.BIAS | 0.799201 |
| 30mmh.CSI | 0.260043 |
| 30mmh.FAR | 0.535395 |
| 30mmh.HSS | 0.412274 |
| 30mmh.POD | 0.371313 |
| 30mmh.a_hits | 7062 |
| 30mmh.b_false_alarms | 8138 |
| 30mmh.c_misses | 11957 |
| 30mmh.d_correct_negatives | 20714987 |
| 30mmh.n_total | 20742144 |
| 5mmh.ACC | 0.975603 |
| 5mmh.BIAS | 1.799589 |
| 5mmh.CSI | 0.406694 |
| 5mmh.FAR | 0.550232 |
| 5mmh.HSS | 0.566717 |
| 5mmh.POD | 0.809398 |
| 5mmh.a_hits | 346883 |
| 5mmh.b_false_alarms | 424365 |
| 5mmh.c_misses | 81686 |
| 5mmh.d_correct_negatives | 19889210 |
| 5mmh.n_total | 20742144 |

### Per-event validation metrics (pooled within event)

| typhoon_id | MAE_event | RMSE_event | SSIM_event_mean | CSI_5mmh | POD_5mmh | FAR_5mmh | HSS_5mmh | ACC_5mmh | BIAS_5mmh | CSI_10mmh | POD_10mmh | FAR_10mmh | HSS_10mmh | ACC_10mmh | BIAS_10mmh | CSI_20mmh | POD_20mmh | FAR_20mmh | HSS_20mmh | ACC_20mmh | BIAS_20mmh | CSI_30mmh | POD_30mmh | FAR_30mmh | HSS_30mmh | ACC_30mmh | BIAS_30mmh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2203 | 1.094281 | 2.692710 | 0.786128 | 0.401604 | 0.828539 | 0.561994 | 0.540638 | 0.933394 | 1.891614 | 0.318251 | 0.744969 | 0.642836 | 0.469474 | 0.970263 | 2.085791 | 0.327637 | 0.513684 | 0.525040 | 0.490429 | 0.993760 | 1.081530 | 0.263998 | 0.377230 | 0.532056 | 0.416278 | 0.997093 | 0.806144 |
| 2205 | 0.356756 | 1.228695 | 0.924807 | 0.420780 | 0.710821 | 0.492312 | 0.582371 | 0.980016 | 1.400113 | 0.267260 | 0.649881 | 0.687786 | 0.418064 | 0.991552 | 2.081526 | 0.254666 | 0.415060 | 0.602767 | 0.405381 | 0.998860 | 1.044880 | 0.198126 | 0.280038 | 0.596180 | 0.330563 | 0.999661 | 0.693472 |
| 2208 | 0.011877 | 0.072785 | 0.998518 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2210 | 0.002318 | 0.033481 | 0.999959 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2212 | 0.018398 | 0.125771 | 0.991713 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2214 | 0.002743 | 0.047043 | 0.999540 | 0.023810 | 0.031250 | 0.909091 | 0.046507 | 0.999987 | 0.343750 | 0.000000 | 0.000000 | 1.000000 | -0.000000 | 0.999999 | 0.500000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2217 | 0.073402 | 0.495186 | 0.984034 | 0.635257 | 0.881757 | 0.305587 | 0.775818 | 0.997713 | 1.269788 | 0.476835 | 0.884559 | 0.491522 | 0.644853 | 0.998064 | 1.739621 | 0.318822 | 0.537797 | 0.560847 | 0.483304 | 0.999613 | 1.224622 | 0.000000 | 0.000000 | 1.000000 | -0.000009 | 0.999975 | 0.307692 |

### Window diagnostics (mean of per-window values)

| Metric | Value |
|---|---|
| MAE_window_mean | 0.410866 |
| RMSE_window_mean | 0.984516 |
| SSIM_window_mean | 0.918375 |
| peak_error_window_mean | 4.833524 |
| n_defined_windows | 1266 |

- n_negative_roundoff_clamped: 0