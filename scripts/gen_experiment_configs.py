"""Generate the frozen Design-C experiment-matrix YAML configs.

Produces the canonical E0-E6 + B1 configs AND the Axis-II P1/P3 configs, all
with the frozen 20-epoch common block (docs/MINIMAX_IMPLEMENTATION_SPEC.md §6).
It never generates P4/P5 (BLOCKED_BY_ENVIRONMENTAL_WIND_DATA).

Canonical channels:
    0 precipitation | 1 center_wind_speed | 2 center_pressure | 3 r_norm
    4 dx_norm | 5 dy_norm | 6 u_move | 7 v_move | 8 dem | 9 dh_dx | 10 dh_dy | 11 land_mask

Existing configs are overwritten ONLY when regenerating the same canonical
names; provenance fields (comments describing aliases) are preserved in the
template.
"""

from pathlib import Path

COMMON = """data:
  h5_path: "ConvLSTM_Dataset_128.h5"
  seq_len: 11
  split_path: "configs/splits_v1.yaml"
  normalization_path: "configs/normalization_v1.json"
  precip_vmax: 100.0

model:
  name: "{model}"
  input_channel_indices: {channels}
  hidden_dims: [64, 128]
  kernel_size: 3

training:
  batch_size: 4
  epochs: 20
  learning_rate: 0.0001
  weight_decay: 0.0001
  lr_patience: 10
  early_stopping_patience: 10
  grad_clip_norm: 1.0
  seed: 42
  use_amp: auto
  checkpoint_selection_metric: rain_mse

physics_loss:
  enabled: true
  lambda_smooth: 0.01
  lambda_extreme: 0.5
  extreme_threshold: 10.0
  components: [{components}]
  orographic:
    enabled: false

output:
  checkpoint_dir: "outputs/experiments/{slug}/models"
  log_dir: "outputs/experiments/{slug}/logs"
"""

# (slug, header, model, channels, components)
EXPS = [
    ("E0_persistence", "Persistence (E0) — Axis I alias: I0",
     "Persistence", [0], ["rain"]),
    ("E1_plain_convlstm", "Plain ConvLSTM (E1) — Axis I alias: I1",
     "PlainConvLSTM", [0], ["rain"]),
    ("E2_resconvlstm", "Residual ConvLSTM (E2) — Axis I alias: I2 (reuse only)",
     "ResConvLSTM", [0], ["rain"]),
    ("E3_resconvlstm_cma", "ResConvLSTM + CMA storm state (E3) — Axis I alias: I3",
     "ResConvLSTM", [0, 1, 2, 3, 4, 5, 6, 7], ["rain"]),
    ("E4_static_terrain", "Static Terrain (E4) — Axis I alias: I4",
     "ResConvLSTM", [0, 1, 2, 3, 4, 5, 6, 7, 8, 11], ["rain"]),
    ("E5_terrain_geometry", "Terrain Geometry (E5) — Axis I alias: I5 AND Axis II alias: P0",
     "ResConvLSTM", list(range(12)), ["rain"]),
    ("E6_terrain_extreme", "Terrain + Extreme Loss (E6) — Axis II alias: P2",
     "ResConvLSTM", list(range(12)), ["rain", "extreme"]),
    ("B1_trajgru", "TrajGRU baseline (B1)",
     "TrajGRU", [0], ["rain"]),
    ("P1_resconvlstm_smooth", "ResConvLSTM + Smoothness (P1) — Axis II",
     "ResConvLSTM", list(range(12)), ["rain", "smooth"]),
    ("P3_resconvlstm_smooth_extreme", "ResConvLSTM + Smoothness + Extreme (P3) — Axis II",
     "ResConvLSTM", list(range(12)), ["rain", "smooth", "extreme"]),
]


def main():
    out_dir = Path("configs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    for slug, header, model, channels, components in EXPS:
        content = (
            f"# Experiment: {header}\n"
            f"# Channels (canonical indices): {channels}\n"
            f"# Loss components: {components}\n"
            f"# Frozen common controls: seed 42 / batch 4 / 20 epochs / AMP auto\n"
            + COMMON.format(
                model=model,
                channels=channels,
                components=", ".join(components),
                slug=slug,
            )
        )
        (out_dir / f"{slug}.yaml").write_text(content, encoding="utf-8")
        print(f"wrote {slug}.yaml")
    print("NOTE: P4/P5 are BLOCKED_BY_ENVIRONMENTAL_WIND_DATA and are not generated.")


if __name__ == "__main__":
    main()
