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
| MAE_global | 0.246680 |
| RMSE_global | 1.254074 |
| peak_error_global | 23.440796 |
| 10mmh.ACC | 0.994909 |
| 10mmh.BIAS | 0.845749 |
| 10mmh.CSI | 0.421053 |
| 10mmh.FAR | 0.353367 |
| 10mmh.HSS | 0.590050 |
| 10mmh.POD | 0.546889 |
| 10mmh.a_hits | 76798 |
| 10mmh.b_false_alarms | 41968 |
| 10mmh.c_misses | 63629 |
| 10mmh.d_correct_negatives | 20559749 |
| 10mmh.n_total | 20742144 |
| 20mmh.ACC | 0.998174 |
| 20mmh.BIAS | 0.650429 |
| 20mmh.CSI | 0.295325 |
| 20mmh.FAR | 0.421480 |
| 20mmh.HSS | 0.455112 |
| 20mmh.POD | 0.376286 |
| 20mmh.a_hits | 15874 |
| 20mmh.b_false_alarms | 11565 |
| 20mmh.c_misses | 26312 |
| 20mmh.d_correct_negatives | 20688393 |
| 20mmh.n_total | 20742144 |
| 30mmh.ACC | 0.999120 |
| 30mmh.BIAS | 0.403596 |
| 30mmh.CSI | 0.187764 |
| 30mmh.FAR | 0.450234 |
| 30mmh.HSS | 0.315803 |
| 30mmh.POD | 0.221883 |
| 30mmh.a_hits | 4220 |
| 30mmh.b_false_alarms | 3456 |
| 30mmh.c_misses | 14799 |
| 30mmh.d_correct_negatives | 20719669 |
| 30mmh.n_total | 20742144 |
| 5mmh.ACC | 0.987462 |
| 5mmh.BIAS | 0.947966 |
| 5mmh.CSI | 0.524958 |
| 5mmh.FAR | 0.292616 |
| 5mmh.HSS | 0.682095 |
| 5mmh.POD | 0.670576 |
| 5mmh.a_hits | 287388 |
| 5mmh.b_false_alarms | 118881 |
| 5mmh.c_misses | 141181 |
| 5mmh.d_correct_negatives | 20194694 |
| 5mmh.n_total | 20742144 |

### Per-event validation metrics (pooled within event)

| typhoon_id | MAE_event | RMSE_event | SSIM_event_mean | CSI_5mmh | POD_5mmh | FAR_5mmh | HSS_5mmh | ACC_5mmh | BIAS_5mmh | CSI_10mmh | POD_10mmh | FAR_10mmh | HSS_10mmh | ACC_10mmh | BIAS_10mmh | CSI_20mmh | POD_20mmh | FAR_20mmh | HSS_20mmh | ACC_20mmh | BIAS_20mmh | CSI_30mmh | POD_30mmh | FAR_30mmh | HSS_30mmh | ACC_30mmh | BIAS_30mmh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2203 | 0.604101 | 2.119168 | 0.912038 | 0.533964 | 0.686094 | 0.293416 | 0.679131 | 0.967693 | 0.971001 | 0.432141 | 0.564116 | 0.351227 | 0.596496 | 0.986187 | 0.869511 | 0.302704 | 0.388589 | 0.422015 | 0.462172 | 0.994702 | 0.672317 | 0.189317 | 0.224074 | 0.450356 | 0.317269 | 0.997347 | 0.407672 |
| 2205 | 0.264922 | 0.963256 | 0.953754 | 0.469932 | 0.588278 | 0.299764 | 0.632540 | 0.986448 | 0.840115 | 0.322552 | 0.410667 | 0.399477 | 0.485792 | 0.995910 | 0.683850 | 0.217380 | 0.258434 | 0.422222 | 0.356755 | 0.999127 | 0.447289 | 0.164754 | 0.190161 | 0.447802 | 0.282790 | 0.999712 | 0.344371 |
| 2208 | 0.019176 | 0.038597 | 0.998915 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2210 | 0.026162 | 0.054870 | 0.997994 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2212 | 0.011390 | 0.046457 | 0.999148 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2214 | 0.014329 | 0.051035 | 0.998949 | 0.027027 | 0.031250 | 0.833333 | 0.052628 | 0.999988 | 0.187500 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999999 | 0.000000 | nan | nan | nan | nan | 1.000000 | nan | nan | nan | nan | nan | 1.000000 | nan |
| 2217 | 0.060338 | 0.357893 | 0.992479 | 0.657123 | 0.756918 | 0.167109 | 0.792195 | 0.998216 | 0.908784 | 0.531796 | 0.621267 | 0.213100 | 0.693804 | 0.998909 | 0.789512 | 0.184158 | 0.200864 | 0.311111 | 0.310932 | 0.999701 | 0.291577 | 0.000000 | 0.000000 | nan | 0.000000 | 0.999981 | 0.000000 |

### Window diagnostics (mean of per-window values)

| Metric | Value |
|---|---|
| MAE_window_mean | 0.246680 |
| RMSE_window_mean | 0.710874 |
| SSIM_window_mean | 0.963467 |
| peak_error_window_mean | 7.285779 |
| n_defined_windows | 1266 |

- n_negative_roundoff_clamped: 0