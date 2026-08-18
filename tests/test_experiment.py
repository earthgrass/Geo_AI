"""Tests for the frozen Design-C pipeline: channel subset, normalization,
model channel counts, TrajGRU absolute semantics, checkpoint selection,
test sealing, config fidelity, E2 no-retrain, and I5==P0 identity."""

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_v2_h5(path, n=4, H=8, W=8):
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("precip/input",
                         data=np.random.rand(n, 11, H, W).astype("float32"))
        f.create_dataset("precip/target",
                         data=np.random.rand(n, 1, H, W).astype("float32"))
        f.create_dataset("terrain",
                         data=np.random.rand(n, 4, H, W).astype("float32"))
        f.create_dataset("track",
                         data=np.random.rand(n, 11, 6).astype("float32"))
        g = f.create_group("meta")
        g.create_dataset("year", data=np.arange(2020, 2020 + n))
        g.create_dataset("typhoon_id", data=np.arange(1, 1 + n))
    return str(path)


def test_channel_subset_shapes():
    import torch  # noqa: F401
    from src.data.dataset import TyphoonDataset

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = _make_v2_h5(Path(tmp) / "d.h5")
        # E1/E2: precipitation only -> 1 channel.
        ds1 = TyphoonDataset(p, channel_indices=[0])
        assert ds1[0][0].shape == (11, 1, 8, 8)
        # E3: 8 channels.
        ds3 = TyphoonDataset(p, channel_indices=[0, 1, 2, 3, 4, 5, 6, 7])
        assert ds3[0][0].shape == (11, 8, 8, 8)
        # E4: 10 channels.
        ds4 = TyphoonDataset(p, channel_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
        assert ds4[0][0].shape == (11, 10, 8, 8)
        # E5/E6/P1-P3: all 12.
        ds5 = TyphoonDataset(p, channel_indices=list(range(12)))
        assert ds5[0][0].shape == (11, 12, 8, 8)


def test_canonical_channel_semantics_preserved():
    import h5py
    from src.data.dataset import TyphoonDataset

    ch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]  # I4 10-channel subset
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = _make_v2_h5(Path(tmp) / "d.h5")
        ds = TyphoonDataset(p, typhoon_ids=[1], channel_indices=ch)
        X, _Y, _meta = ds[0]
        with h5py.File(p, "r") as f:
            # Canonical 0 (precipitation) sits at subset position 0.
            precip = f["precip/input"][0]
            assert np.allclose(X[:, 0].numpy(), precip, atol=1e-6)
            # I4 subset is [0..8, 11]: canonical 8 (dem) at position 8, and
            # canonical 11 (land_mask) at position 9 (the last channel).
            dem = f["terrain"][0][0]
            land = f["terrain"][0][3]
            assert np.allclose(X[:, 8].numpy(), dem, atol=1e-6)
            assert np.allclose(X[:, 9].numpy(), land, atol=1e-6)


def test_channel_normalize():
    import torch
    from src.data.transforms import ChannelNormalize

    stats = {
        "track_features": {
            "center_wind_speed": {"mean": 20.0, "std": 10.0},
            "center_pressure": {"mean": 990.0, "std": 20.0},
            "u_move": {"mean": 0.0, "std": 10.0},
            "v_move": {"mean": 0.0, "std": 10.0},
        },
        "terrain": {
            "dem": {"mean": 50.0, "std": 200.0},
            "dh_dx": {"mean": 0.0, "std": 5.0},
            "dh_dy": {"mean": 0.0, "std": 5.0},
        },
    }
    idx = [0, 1, 8, 11]  # precip, wind, dem, land_mask
    t = ChannelNormalize(stats, channel_indices=idx, precip_vmax=100.0)

    X = torch.zeros(1, 4, 4, 4)
    X[:, 0] = 100.0  # precip -> 1.0
    X[:, 1] = 30.0   # wind -> (30-20)/10 = 1.0
    X[:, 2] = 250.0  # dem -> (250-50)/200 = 1.0
    X[:, 3] = 1.0    # land_mask -> unchanged
    Y = torch.full((1, 1, 4, 4), 50.0)
    meta = {"P_prev": torch.full((1, 1, 4, 4), 100.0)}

    X, Y, meta = t(X, Y, meta)
    assert abs(X[0, 0, 0, 0].item() - 1.0) < 1e-6
    assert abs(X[0, 1, 0, 0].item() - 1.0) < 1e-6
    assert abs(X[0, 2, 0, 0].item() - 1.0) < 1e-6
    assert abs(X[0, 3, 0, 0].item() - 1.0) < 1e-6  # land_mask unchanged
    assert abs(Y[0, 0, 0, 0].item() - 0.5) < 1e-6
    assert abs(meta["P_prev"][0, 0, 0, 0].item() - 1.0) < 1e-6


def test_model_input_channel_count():
    import torch
    from src.models.baselines import ResConvLSTM

    for n_ch in (1, 8, 10, 12):
        m = ResConvLSTM(input_channels=n_ch, hidden_dims=[8, 16])
        X = torch.randn(2, 11, n_ch, 8, 8)
        out = m.compute_prediction(X)
        assert out.shape == (2, 1, 8, 8)


def test_trajgru_forward():
    import torch
    from src.models.trajgru import TrajGRU

    m = TrajGRU(input_channels=1, hidden_dims=[8, 16])
    X = torch.randn(2, 11, 1, 8, 8)
    out = m(X)
    assert out.shape == (2, 1, 8, 8)
    assert torch.isfinite(out).all()


def test_trajgru_train_eval_semantics_match():
    import torch
    import tempfile
    from src.models.trajgru import TrajGRU
    from src.training.trainer import Trainer
    from src.evaluation.evaluator import predict_absolute

    class _DS:
        def __len__(self):
            return 1

    class _L:
        def __init__(self):
            self.dataset = _DS()

    model = TrajGRU(input_channels=1, hidden_dims=[4, 8])
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = {
            "model_name": "TrajGRU",
            "use_physics_loss": False,
            "normalize_precip": False,
            "precip_vmax": 100.0,
            "learning_rate": 1e-4, "weight_decay": 1e-4,
            "lr_patience": 10, "early_stopping_patience": 10,
            "grad_clip_norm": 1.0, "use_amp": False,
            "checkpoint_selection_metric": "rain_mse",
            "checkpoint_dir": str(Path(tmp) / "models"),
            "log_dir": str(Path(tmp) / "logs"),
        }
        trainer = Trainer(model, _L(), _L(), cfg)
        X = torch.randn(2, 11, 1, 8, 8)
        P_prev = torch.rand(2, 1, 8, 8)

        out_train = trainer._predict_precipitation(X, P_prev)
        out_eval = predict_absolute(model, X)
        assert torch.allclose(out_train, out_eval)
        # Both are the model's ABSOLUTE output (already ReLU'd), not a residual
        # that is then added to P_prev.
        assert torch.allclose(out_train, model(X))
        assert not torch.allclose(out_train, torch.relu(P_prev + model(X)))


def _trainer_dummy_loaders():
    class _DS:
        def __len__(self):
            return 1

    class _L:
        def __init__(self):
            self.dataset = _DS()

    return _L(), _L()


def test_checkpoint_selection_uses_val_base_mse():
    import torch  # noqa: F401
    import tempfile
    from src.models.baselines import ResConvLSTM
    from src.training.trainer import Trainer

    train_l, val_l = _trainer_dummy_loaders()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = {
            "model_name": "ResConvLSTM",
            "use_physics_loss": False,
            "normalize_precip": False,
            "precip_vmax": 100.0,
            "learning_rate": 1e-4, "weight_decay": 1e-4,
            "lr_patience": 10, "early_stopping_patience": 10,
            "grad_clip_norm": 1.0, "use_amp": False,
            "checkpoint_selection_metric": "rain_mse",
            "checkpoint_dir": str(Path(tmp) / "models"),
            "log_dir": str(Path(tmp) / "logs"),
        }
        model = ResConvLSTM(input_channels=1, hidden_dims=[2, 4])
        trainer = Trainer(model, train_l, val_l, cfg)

        # Epochs where the composite total ranks differently from base rain MSE.
        seq = [
            {"rain": 0.5, "total": 0.05},   # epoch 0
            {"rain": 0.4, "total": 0.90},   # epoch 1 -> best by RAIN
            {"rain": 0.6, "total": 0.01},   # epoch 2 -> best by TOTAL, not RAIN
        ]
        it = iter(seq)
        trainer.train_epoch = lambda: {"rain": 0.5, "total": 0.5}
        trainer.validate = lambda: next(it)

        trainer.train(epochs=3)

        assert trainer.selection_metric == "rain_mse"
        assert trainer.best_val_mse == 0.4      # epoch 1 selected by rain MSE
        assert trainer.best_epoch == 1          # 0-indexed
        assert trainer.best_val_total == 0.90   # composite recorded, not used


def _load_runner_source():
    return (REPO_ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")


def test_test_sealing_default_off():
    # The runner has NO --allow-test-eval flag (it was removed).
    src = _load_runner_source()
    assert "--allow-test-eval" not in src
    assert "allow_test_eval" not in src


def test_test_sealed_in_all_normal_runners():
    import subprocess

    src = _load_runner_source()
    assert "--allow-test-eval" not in src
    assert "test_loader" not in src
    assert "allow_test_eval" not in src
    # The runner only ever constructs train/val loaders.
    assert "make_loader(" in src
    assert "_test_ids" in src  # the test split is loaded only to be discarded

    # Legacy evaluate_models entry point fails fast (deprecated).
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); "
         "from src.evaluation import evaluate_models; evaluate_models.main()"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode != 0
    assert "DEPRECATED" in (r.stdout + r.stderr)


def _no_test_flag_in_python_invocations(script_path):
    """The --allow-test-eval flag may appear in comments/echo as documentation,
    but it must never be passed to a python invocation."""
    text = (REPO_ROOT / "scripts" / script_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("python ") or stripped.startswith("  python "):
            assert "--allow-test-eval" not in stripped, (
                f"{script_path}: python invocation passes --allow-test-eval:\n{line}"
            )


def test_checkpoint_eval_val_allowed():
    from scripts.evaluate_checkpoint import check_split_allowed

    check_split_allowed("val")
    check_split_allowed("train")


def test_checkpoint_eval_test_refused():
    from scripts.evaluate_checkpoint import check_split_allowed

    try:
        check_split_allowed("test")
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit for --split test")


def test_e2_reuse_never_retrains():
    gate = (REPO_ROOT / "scripts" / "run_backbone_gate.sh").read_text(encoding="utf-8")
    assert "--mode validate-only" in gate
    assert "BLOCKED_MISSING_OR_INCOMPATIBLE_E2" in gate
    assert "run_exp \"E2" not in gate
    assert "run_exp \"I2" not in gate

    axis_i = (REPO_ROOT / "scripts" / "run_axis_i.sh").read_text(encoding="utf-8")
    assert "--mode validate-only" in axis_i
    assert "run_exp \"I2" not in axis_i
    # The official E2 checkpoint is reused through the verifier, never trained.
    assert "verify_experiment_artifact.py" in gate
    assert "verify_experiment_artifact.py" in axis_i
    assert "run_exp \"I2" not in axis_i
    # No test option is ever passed to python in any GPU gate.
    for sh in ("run_backbone_gate.sh", "run_axis_i.sh", "run_axis_ii_c1.sh"):
        _no_test_flag_in_python_invocations(sh)


def test_i5_p0_same_artifact():
    from src.experiments.registry import resolve_alias, load_alias_registry

    reg = load_alias_registry()
    assert resolve_alias("I5", reg) == resolve_alias("P0", reg) == "E5_terrain_geometry"


def test_no_runnable_p4_p5():
    from src.experiments.registry import load_alias_registry

    reg = load_alias_registry()
    assert "P4" not in reg["aliases"] and "P5" not in reg["aliases"]
    for f in (REPO_ROOT / "configs" / "experiments").glob("P[45]*.yaml"):
        raise AssertionError(f"unexpected runnable P4/P5 config: {f}")


def test_nonnegativity_is_architectural_not_training_ablation():
    import torch
    from src.models.baselines import ResConvLSTM
    from src.training.physics_loss import PhysicsInformedLoss

    m = ResConvLSTM(input_channels=4, hidden_dims=[4, 8])
    X = torch.randn(2, 11, 4, 16, 16) * 0.1
    out = m.compute_prediction(X)
    assert (out >= 0).all(), "ReLU(P_last + delta) guarantees P_hat >= 0"

    # No nonnegativity training experiment exists.
    for cfg_path in (REPO_ROOT / "configs" / "experiments").glob("*.yaml"):
        assert "nonneg" not in cfg_path.read_text(encoding="utf-8")
    assert "nonneg" not in PhysicsInformedLoss().components


def test_config_fidelity():
    import yaml
    from src.experiments.registry import validate_formal_config
    from scripts.run_experiment import make_model

    cfg = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/E3_resconvlstm_cma.yaml")
        .read_text(encoding="utf-8"))
    validate_formal_config(cfg, "E3_resconvlstm_cma.yaml")
    model = make_model(cfg["model"]["name"], cfg["model"]["input_channel_indices"],
                       cfg["model"]["hidden_dims"], cfg["model"]["kernel_size"])
    assert model.hidden_dims == [64, 128]
    assert cfg["model"]["kernel_size"] == 3
    assert cfg["data"]["seq_len"] == 11
    assert cfg["physics_loss"]["lambda_smooth"] == 0.01
    assert cfg["physics_loss"]["lambda_extreme"] == 0.5
    assert cfg["physics_loss"]["extreme_threshold"] == 10.0
    assert cfg["training"]["epochs"] == 20
    assert cfg["training"]["checkpoint_selection_metric"] == "rain_mse"
    assert cfg["training"]["use_amp"] == "auto"


def test_formal_config_rejects_deviations():
    import copy
    import yaml
    from src.experiments.registry import validate_formal_config

    base = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/E3_resconvlstm_cma.yaml")
        .read_text(encoding="utf-8"))

    def expect_reject(mutator):
        cfg = copy.deepcopy(base)
        mutator(cfg)
        try:
            validate_formal_config(cfg, "mutated.yaml")
        except ValueError:
            return
        raise AssertionError("expected ValueError for deviating config")

    expect_reject(lambda c: c["training"].__setitem__("epochs", 100))
    expect_reject(lambda c: c["training"].__setitem__("seed", 7))
    expect_reject(lambda c: c["physics_loss"].__setitem__(
        "components", ["rain", "foo"]))
    expect_reject(lambda c: c["model"].__setitem__(
        "input_channel_indices", [1, 2, 3, 4, 5, 6, 7]))
    expect_reject(lambda c: c["training"].__setitem__("use_amp", "off"))


def test_persistence_evaluation_works():
    import torch
    from src.models.baselines import PersistenceBaseline
    from src.evaluation.evaluator import evaluate_model_v2

    class L:
        def __init__(self):
            self.X = torch.rand(2, 11, 1, 8, 8)
            self.Y = torch.rand(2, 1, 8, 8)
            self.meta = {"typhoon_id": torch.tensor([2203, 2203])}

        def __iter__(self):
            yield self.X, self.Y, self.meta

    res = evaluate_model_v2(
        PersistenceBaseline(0), L(), torch.device("cpu"), precip_vmax=100.0,
        thresholds=[5.0, 10.0], channel_indices=[0], split="val")
    assert res["n_windows"] == 2
    assert res["n_events"] == 1
    assert "2203" in res["per_event"]


def test_loader_shuffle():
    from torch.utils.data import RandomSampler, SequentialSampler

    from scripts import run_experiment as run_exp
    stats = {"track_features": {}, "terrain": {}}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = _make_v2_h5(Path(tmp) / "d.h5")
        _, train_loader = run_exp.make_loader(
            p, [1, 2], [0], stats, batch_size=2, num_workers=0,
            shuffle=True, pin_memory=False)
        _, val_loader = run_exp.make_loader(
            p, [1, 2], [0], stats, batch_size=2, num_workers=0,
            shuffle=False, pin_memory=False)
        assert isinstance(train_loader.sampler, RandomSampler)
        assert isinstance(val_loader.sampler, SequentialSampler)


def test_amp_selection():
    from scripts import run_experiment as run_exp
    f = run_exp.resolve_use_amp
    assert f(None, True) is True    # YAML True (CUDA implied)
    assert f(None, False) is False  # YAML False
    assert f("auto", True) is True
    assert f("auto", False) is False


def test_event_aggregation_pooled_counts():
    # v2 per-event aggregation pools counts before computing ratios (the old
    # arithmetic-mean-of-window-metrics assertion encoded v1 behavior).
    from src.evaluation.evaluator import aggregate_v2

    w1 = {"typhoon_id": 1, "sum_abs": 1, "sum_sq": 1, "n": 4, "SSIM": 1.0,
          "peak_error": 0.0, "f_max": 0.0, "y_max": 0.0,
          "counts": {"5mmh": {"a": 1, "b": 1, "c": 0, "d": 2}}}
    w2 = {"typhoon_id": 1, "sum_abs": 1, "sum_sq": 1, "n": 4, "SSIM": 1.0,
          "peak_error": 0.0, "f_max": 0.0, "y_max": 0.0,
          "counts": {"5mmh": {"a": 10, "b": 0, "c": 0, "d": 2}}}
    ev = aggregate_v2([w1, w2], thresholds=[5.0])["per_event"]["1"]
    assert np.isclose(ev["categorical"]["5mmh"]["CSI"], 11 / 12)


def main():
    tests = [
        ("test_channel_subset_shapes", test_channel_subset_shapes),
        ("test_canonical_channel_semantics_preserved",
         test_canonical_channel_semantics_preserved),
        ("test_channel_normalize", test_channel_normalize),
        ("test_model_input_channel_count", test_model_input_channel_count),
        ("test_trajgru_forward", test_trajgru_forward),
        ("test_trajgru_train_eval_semantics_match",
         test_trajgru_train_eval_semantics_match),
        ("test_checkpoint_selection_uses_val_base_mse",
         test_checkpoint_selection_uses_val_base_mse),
        ("test_test_sealing_default_off", test_test_sealing_default_off),
        ("test_test_sealed_in_all_normal_runners",
         test_test_sealed_in_all_normal_runners),
        ("test_checkpoint_eval_val_allowed", test_checkpoint_eval_val_allowed),
        ("test_checkpoint_eval_test_refused", test_checkpoint_eval_test_refused),
        ("test_e2_reuse_never_retrains", test_e2_reuse_never_retrains),
        ("test_i5_p0_same_artifact", test_i5_p0_same_artifact),
        ("test_no_runnable_p4_p5", test_no_runnable_p4_p5),
        ("test_nonnegativity_is_architectural_not_training_ablation",
         test_nonnegativity_is_architectural_not_training_ablation),
        ("test_config_fidelity", test_config_fidelity),
        ("test_formal_config_rejects_deviations",
         test_formal_config_rejects_deviations),
        ("test_persistence_evaluation_works",
         test_persistence_evaluation_works),
        ("test_loader_shuffle", test_loader_shuffle),
        ("test_amp_selection", test_amp_selection),
        ("test_event_aggregation_pooled_counts",
         test_event_aggregation_pooled_counts),
    ]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{sys.argv[0]}: {len(tests) - n_fail}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
