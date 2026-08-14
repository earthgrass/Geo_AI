"""Generate the preliminary experiment-manifest YAML configs (Phase 9).

These are a SPEC for the training phase; channel-subset selection is a
training-phase TODO (the dataset currently reconstructs all 12 channels).
"""

from pathlib import Path

TEMPLATE = """# Experiment: {name}
# Input channels: {channels}
# Loss components: {components}
# NOTE: channel-subset selection is a training-phase TODO; the dataset currently
# reconstructs all 12 channels. `input_channels` documents the intended subset.
data:
  h5_path: "ConvLSTM_Dataset_128.h5"
  seq_len: 11
  precip_channel_idx: 0
  num_workers: 4
  normalize: true
  precip_vmax: 100.0

model:
  name: "{model}"
  input_channels: {nch}
  precip_channel_idx: 0
  hidden_dims: [64, 128]
  kernel_size: 3
  use_attention: false

training:
  batch_size: 8
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  lr_patience: 10
  early_stopping_patience: 20
  grad_clip_norm: 1.0
  use_amp: true
  seed: 42

physics_loss:
  use_physics_loss: true
  lambda_smooth: 0.01
  lambda_extreme: 0.5
  lambda_oro: 0.1
  extreme_threshold: 10.0
  orographic:
    enabled: false
    u_channel: null
    v_channel: null
    dh_dx_channel: 9
    dh_dy_channel: 10
  components: [{components}]

output:
  checkpoint_dir: "outputs/models/{slug}"
  log_dir: "outputs/logs/{slug}"
"""

EXPS = [
    ("01_persistence", "Persistence (E0)", "precipitation only", "Persistence", 0, "rain"),
    ("02_plain_convlstm", "Plain ConvLSTM (E1)", "precipitation only", "PlainConvLSTM", 1, "rain"),
    ("03_resconvlstm_precip", "ResConvLSTM precip-only (E2)", "precipitation only", "ResConvLSTM", 1, "rain"),
    ("04_resconvlstm_cma", "ResConvLSTM + CMA (E3)",
     "precip + center_wind_speed + center_pressure + r_norm + dx_norm + dy_norm + u_move + v_move (8ch)",
     "ResConvLSTM", 8, "rain"),
    ("05_resconvlstm_terrain", "Terrain-aware ResConvLSTM (E4)",
     "all 12 channels (precip + CMA + dem/dh_dx/dh_dy/land_mask)",
     "ResConvLSTM", 12, "rain"),
    ("06_resconvlstm_extreme", "Terrain-aware + Extreme Loss (E5)",
     "all 12 channels", "ResConvLSTM", 12, "rain, extreme"),
]


def main():
    out_dir = Path("configs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    for slug, name, channels, model, nch, comp in EXPS:
        content = TEMPLATE.format(
            name=name, channels=channels, components=comp,
            model=model, nch=nch, slug=slug,
        )
        (out_dir / f"{slug}.yaml").write_text(content, encoding="utf-8")
        print(f"wrote {slug}.yaml")


if __name__ == "__main__":
    main()
