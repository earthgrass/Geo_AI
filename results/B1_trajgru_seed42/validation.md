## Experiment: TrajGRU

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
| MAE_global | 0.377054 |
| RMSE_global | 1.546354 |
| peak_error_global | 38.072262 |
| 10mmh.ACC | 0.994789 |
| 10mmh.BIAS | 0.570930 |
| 10mmh.CSI | 0.342272 |
| 10mmh.FAR | 0.298376 |
| 10mmh.HSS | 0.507565 |
| 10mmh.POD | 0.400578 |
| 10mmh.a_hits | 56252 |
| 10mmh.b_false_alarms | 23922 |
| 10mmh.c_misses | 84175 |
| 10mmh.d_correct_negatives | 20577795 |
| 10mmh.n_total | 20742144 |
| 20mmh.ACC | 0.998157 |
| 20mmh.BIAS | 0.581141 |
| 20mmh.CSI | 0.271289 |
| 20mmh.FAR | 0.419400 |
| 20mmh.HSS | 0.425936 |
| 20mmh.POD | 0.337411 |
| 20mmh.a_hits | 14234 |
| 20mmh.b_false_alarms | 10282 |
| 20mmh.c_misses | 27952 |
| 20mmh.d_correct_negatives | 20689676 |
| 20mmh.n_total | 20742144 |
| 30mmh.ACC | 0.999116 |
| 30mmh.BIAS | 0.390189 |
| 30mmh.CSI | 0.181148 |
| 30mmh.FAR | 0.453578 |
| 30mmh.HSS | 0.306375 |
| 30mmh.POD | 0.213208 |
| 30mmh.a_hits | 4055 |
| 30mmh.b_false_alarms | 3366 |
| 30mmh.c_misses | 14964 |
| 30mmh.d_correct_negatives | 20719759 |
| 30mmh.n_total | 20742144 |
| 5mmh.ACC | 0.985165 |
| 5mmh.BIAS | 0.448579 |
| 5mmh.CSI | 0.337216 |
| 5mmh.FAR | 0.185652 |
| 5mmh.HSS | 0.497931 |
| 5mmh.POD | 0.365299 |
| 5mmh.a_hits | 156556 |
| 5mmh.b_false_alarms | 35691 |
| 5mmh.c_misses | 272013 |
| 5mmh.d_correct_negatives | 20277884 |
| 5mmh.n_total | 20742144 |

### Per-event validation metrics (pooled within event)

| typhoon_id | MAE_event | RMSE_event | SSIM_event_mean | CSI_5mmh | POD_5mmh | FAR_5mmh | HSS_5mmh | ACC_5mmh | BIAS_5mmh | CSI_10mmh | POD_10mmh | FAR_10mmh | HSS_10mmh | ACC_10mmh | BIAS_10mmh | CSI_20mmh | POD_20mmh | FAR_20mmh | HSS_20mmh | ACC_20mmh | BIAS_20mmh | CSI_30mmh | POD_30mmh | FAR_30mmh | HSS_30mmh | ACC_30mmh | BIAS_30mmh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2203 | 0.863242 | 2.509826 | 0.779870 | 0.355480 | 0.386426 | 0.183856 | 0.507429 | 0.962200 | 0.473478 | 0.355952 | 0.420260 | 0.300643 | 0.518283 | 0.985831 | 0.600923 | 0.279350 | 0.351066 | 0.422390 | 0.434174 | 0.994639 | 0.607791 | 0.183681 | 0.216882 | 0.454571 | 0.309269 | 0.997335 | 0.397636 |
| 2205 | 0.443224 | 1.339685 | 0.861975 | 0.239019 | 0.253359 | 0.191452 | 0.379776 | 0.983526 | 0.313350 | 0.233744 | 0.260072 | 0.302207 | 0.377315 | 0.995958 | 0.372706 | 0.194750 | 0.218976 | 0.362281 | 0.325686 | 0.999151 | 0.343373 | 0.139712 | 0.156102 | 0.429066 | 0.245074 | 0.999713 | 0.273415 |
| 2208 | 0.051986 | 0.433968 | 0.988494 | 0.000000 | nan | 1.000000 | 0.000000 | 0.999939 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2210 | 0.051986 | 0.433968 | 0.988494 | 0.000000 | nan | 1.000000 | 0.000000 | 0.999939 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2212 | 0.051986 | 0.433968 | 0.988494 | 0.000000 | nan | 1.000000 | 0.000000 | 0.999939 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2214 | 0.054203 | 0.436513 | 0.987990 | 0.000000 | 0.000000 | 1.000000 | -0.000018 | 0.999925 | 6.156250 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999999 | 0.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2217 | 0.115777 | 0.666831 | 0.970774 | 0.456889 | 0.479086 | 0.092073 | 0.626045 | 0.997428 | 0.527671 | 0.367507 | 0.392935 | 0.149724 | 0.536900 | 0.998651 | 0.462127 | 0.052854 | 0.053996 | 0.285714 | 0.100359 | 0.999674 | 0.075594 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999981 | 0.000000 |

### Window diagnostics (mean of per-window values)

| Metric | Value |
|---|---|
| MAE_window_mean | 0.377054 |
| RMSE_window_mean | 1.134949 |
| SSIM_window_mean | 0.900401 |
| peak_error_window_mean | 10.814739 |
| n_defined_windows | 1266 |

- n_negative_roundoff_clamped: 0