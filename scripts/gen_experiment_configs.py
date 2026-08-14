"""Generate the frozen experiment-matrix YAML configs (E0-E6 + B1).

Canonical channels:
    0 precipitation | 1 center_wind_speed | 2 center_pressure | 3 r_norm
    4 dx_norm | 5 dy_norm | 6 u_move | 7 v_move | 8 dem | 9 dh_dx | 10 dh_dy | 11 land_mask
"""

from pathlib import Path

TEMPLATE = """# Experiment: {name}
# Channels (canonical indices): {channels}
# Loss components: {components}
data:
  h5_path: "ConvLSTM_Dataset_128.h5"
  seq_len: 11

model:
  name: "{model}"
  input_channel_indices: {channels}
  hidden_dims: [64, 128]
  kernel_size: 3

training:
  batch_size: 4
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  lr_patience: 10
  early_stopping_patience: 10
  grad_clip_norm: 1.0
  seed: 42

physics_loss:
  lambda_extreme: 0.5
  extreme_threshold: 10.0
  components: [{components}]

output:
  checkpoint_dir: "outputs/experiments/{slug}/models"
  log_dir: "outputs/experiments/{slug}/logs"
"""

EXPS = [
    ("E0_persistence", "Persistence (E0)", "Persistence", [0], ["rain"]),
    ("E1_plain_convlstm", "Plain ConvLSTM (E1)", "PlainConvLSTM", [0], ["rain"]),
    ("E2_resconvlstm", "Residual ConvLSTM (E2)", "ResConvLSTM", [0], ["rain"]),
    ("E3_resconvlstm_cma", "ResConvLSTM + CMA (E3)", "ResConvLSTM",
     [0, 1, 2, 3, 4, 5, 6, 7], ["rain"]),
    ("E4_static_terrain", "Static Terrain (E4)", "ResConvLSTM",
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 11], ["rain"]),
    ("E5_terrain_geometry", "Terrain Geometry (E5)", "ResConvLSTM",
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["rain"]),
    ("E6_terrain_extreme", "Terrain + Extreme Loss (E6)", "ResConvLSTM",
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["rain", "extreme"]),
    ("B1_trajgru", "TrajGRU baseline (B1)", "TrajGRU", [0], ["rain"]),
]


def main():
    out_dir = Path("configs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    for slug, name, model, channels, components in EXPS:
        content = TEMPLATE.format(
            name=name,
            model=model,
            channels=channels,
            components=", ".join(components),
            slug=slug,
        )
        (out_dir / f"{slug}.yaml").write_text(content, encoding="utf-8")
        print(f"wrote {slug}.yaml")


if __name__ == "__main__":
    main()
