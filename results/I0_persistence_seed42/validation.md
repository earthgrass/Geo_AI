## Experiment: Persistence

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
| MAE_global | 0.249059 |
| RMSE_global | 1.441949 |
| peak_error_global | 0.021309 |
| 10mmh.ACC | 0.994224 |
| 10mmh.BIAS | 0.996432 |
| 10mmh.CSI | 0.401240 |
| 10mmh.FAR | 0.426282 |
| 10mmh.HSS | 0.569785 |
| 10mmh.POD | 0.571671 |
| 10mmh.a_hits | 80278 |
| 10mmh.b_false_alarms | 59648 |
| 10mmh.c_misses | 60149 |
| 10mmh.d_correct_negatives | 20542069 |
| 10mmh.n_total | 20742144 |
| 20mmh.ACC | 0.997929 |
| 20mmh.BIAS | 0.985493 |
| 20mmh.CSI | 0.322032 |
| 20mmh.FAR | 0.509237 |
| 20mmh.HSS | 0.486140 |
| 20mmh.POD | 0.483644 |
| 20mmh.a_hits | 20403 |
| 20mmh.b_false_alarms | 21171 |
| 20mmh.c_misses | 21783 |
| 20mmh.d_correct_negatives | 20678787 |
| 20mmh.n_total | 20742144 |
| 30mmh.ACC | 0.998960 |
| 30mmh.BIAS | 0.979126 |
| 30mmh.CSI | 0.271355 |
| 30mmh.FAR | 0.568575 |
| 30mmh.HSS | 0.426355 |
| 30mmh.POD | 0.422420 |
| 30mmh.a_hits | 8034 |
| 30mmh.b_false_alarms | 10588 |
| 30mmh.c_misses | 10985 |
| 30mmh.d_correct_negatives | 20712537 |
| 30mmh.n_total | 20742144 |
| 5mmh.ACC | 0.985933 |
| 5mmh.BIAS | 0.997485 |
| 5mmh.CSI | 0.491596 |
| 5mmh.FAR | 0.340014 |
| 5mmh.HSS | 0.651973 |
| 5mmh.POD | 0.658326 |
| 5mmh.a_hits | 282138 |
| 5mmh.b_false_alarms | 145353 |
| 5mmh.c_misses | 146431 |
| 5mmh.d_correct_negatives | 20168222 |
| 5mmh.n_total | 20742144 |

### Per-event validation metrics (pooled within event)

| typhoon_id | MAE_event | RMSE_event | SSIM_event_mean | CSI_5mmh | POD_5mmh | FAR_5mmh | HSS_5mmh | ACC_5mmh | BIAS_5mmh | CSI_10mmh | POD_10mmh | FAR_10mmh | HSS_10mmh | ACC_10mmh | BIAS_10mmh | CSI_20mmh | POD_20mmh | FAR_20mmh | HSS_20mmh | ACC_20mmh | BIAS_20mmh | CSI_30mmh | POD_30mmh | FAR_30mmh | HSS_30mmh | ACC_30mmh | BIAS_30mmh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2203 | 0.626881 | 2.430651 | 0.904937 | 0.504128 | 0.670433 | 0.329782 | 0.651522 | 0.964421 | 1.000320 | 0.415240 | 0.585836 | 0.412209 | 0.578979 | 0.984627 | 0.996675 | 0.329264 | 0.491602 | 0.500727 | 0.492427 | 0.994072 | 0.984637 | 0.276340 | 0.428022 | 0.561865 | 0.431466 | 0.996901 | 0.976918 |
| 2205 | 0.290632 | 1.132405 | 0.945751 | 0.424969 | 0.593799 | 0.400854 | 0.588085 | 0.983590 | 0.991076 | 0.296845 | 0.458164 | 0.542572 | 0.455211 | 0.994854 | 1.001609 | 0.246805 | 0.395482 | 0.603682 | 0.395333 | 0.998868 | 0.997892 | 0.200790 | 0.336802 | 0.667910 | 0.334229 | 0.999600 | 1.014191 |
| 2208 | 0.000000 | 0.000000 | 1.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2210 | 0.000000 | 0.000000 | 1.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2212 | 0.000000 | 0.000000 | 1.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2214 | 0.001850 | 0.043427 | 0.999726 | 0.048387 | 0.093750 | 0.909091 | 0.092298 | 0.999981 | 1.031250 | 0.000000 | 0.000000 | 1.000000 | -0.000001 | 0.999998 | 1.500000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2217 | 0.046940 | 0.427617 | 0.992180 | 0.617143 | 0.729730 | 0.200000 | 0.762226 | 0.997955 | 0.912162 | 0.489450 | 0.642025 | 0.326842 | 0.656554 | 0.998664 | 0.953751 | 0.301429 | 0.455724 | 0.529018 | 0.463050 | 0.999645 | 0.967603 | 0.018868 | 0.038462 | 0.964286 | 0.037018 | 0.999962 | 1.076923 |

### Window diagnostics (mean of per-window values)

| Metric | Value |
|---|---|
| MAE_window_mean | 0.249059 |
| RMSE_window_mean | 0.791295 |
| SSIM_window_mean | 0.960450 |
| peak_error_window_mean | 3.955544 |
| n_defined_windows | 1266 |

- n_negative_roundoff_clamped: 0