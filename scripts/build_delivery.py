"""Build a paper-ready delivery package for the GeoAI typhoon project.

The script is intentionally conservative:
1. Reuse figures that already exist in the archived competition paper.
2. Regenerate only lightweight summary figures from existing CSV/NPZ files.
3. Export raw/cleaned tabular data and derived paper tables.
4. Create a LaTeX paper folder based on the existing essay, without code appendix.

Run from the repository root:
    python scripts/build_delivery.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PAPER = ROOT / "archive" / "competition" / "essay" / "essay"
DELIVERY = ROOT / "deliverables" / "PI_ResConvLSTM_Paper_Package"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "deliverables" / "_matplotlib_cache"))

RAW_TABLES = [
    "All_Years_Typhoon_Features.csv",
    "TIF_Features_Base.csv",
    "Typhoon_Full_Dataset_Q1.csv",
    "Typhoon_Features_Complete.csv",
    "1.4_Data_Full_Spearman_Matrix.csv",
    "1.4_Data_RF_Raw_Importance.csv",
    "1.4_Data_SHAP_Base_Features.csv",
    "Sensitivity_Analysis_Results.csv",
    "ConvLSTM_Dataset_128_metadata.csv",
    "Virtual_Typhoons_2026.txt",
]

EXISTING_FIGURE_NAMES = [
    "background.png",
    "model.png",
    "PINN.png",
    "Fig1_1_Heatmap_CN.png",
    "Fig1_2_Joint_RegPlot.png",
    "Fig1_3_Radar_Chart_CN.png",
    "Fig2_1_RF_Importance_CN.png",
    "Fig2_2_SHAP_Summary.png",
    "Fig3_1_Evolution_Timeline.png",
    "Fig3_2_Spatial_Snapshots.png",
    "Combined_Peak_Map.png",
    "Trend_KONG-REY.png",
    "Trend_MAN-YI.png",
    "Kong_t1.png",
    "Kong_t2.png",
    "Kong_t3.png",
    "Kong_t4.png",
    "Manyi_t1.png",
    "Manyi_t2.png",
    "Manyi_t3.png",
    "Manyi_t4.png",
    "Climate_Scenario_Comparison.png",
    "Climate_Scenario_Final.png",
    "Fig4_1_Micro_V-SHIFT.png",
    "Fig4_1_Micro_V-INTENSE.png",
    "Fig4_1_Micro_V-COMPOUND.png",
    "Fig4_1_Micro_V-SLOW.png",
    "Fig4_4_Macro_V-SHIFT.png",
    "Fig4_4_Macro_V-INTENSE.png",
    "Fig4_4_Macro_V-COMPOUND.png",
    "Fig4_4_Macro_V-SLOW.png",
    "Fig4_5_China_Landfall_Risk_Expanded_T2.png",
    "Fig4_5_China_Landfall_Risk_Expanded_T3.png",
    "Fig4_6a_Real_Sensitivity_Pressure.png",
    "Fig4_6b_Real_Sensitivity_Latitude.png",
    "Fig4_6c_Real_Sensitivity_Speed.png",
]


def reset_delivery_dir() -> None:
    if DELIVERY.exists():
        shutil.rmtree(DELIVERY)

    for subdir in [
        "code/src",
        "code/configs",
        "code/scripts",
        "figures/existing",
        "figures/generated",
        "tables/raw",
        "tables/derived",
        "paper",
        "paper/figures",
        "docs",
        "models",
    ]:
        (DELIVERY / subdir).mkdir(parents=True, exist_ok=True)


def copy_project_assets() -> None:
    shutil.copytree(ROOT / "src", DELIVERY / "code" / "src", dirs_exist_ok=True)
    shutil.copytree(ROOT / "configs", DELIVERY / "code" / "configs", dirs_exist_ok=True)
    shutil.copytree(ROOT / "scripts", DELIVERY / "code" / "scripts", dirs_exist_ok=True)

    for name in ["README.md", "requirements.txt", "Typhoon_PI_ResConvLSTM_Summary_for_Agent.md"]:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, DELIVERY / "docs" / name)

    if (ROOT / "docs").exists():
        shutil.copytree(ROOT / "docs", DELIVERY / "docs" / "project_docs", dirs_exist_ok=True)

    for model_name in ["typhoon_convlstm_best.pth"]:
        src = ROOT / model_name
        if src.exists():
            shutil.copy2(src, DELIVERY / "models" / model_name)


def copy_raw_tables() -> pd.DataFrame:
    inventory = []
    for name in RAW_TABLES:
        src = ROOT / name
        if not src.exists():
            continue
        dst = DELIVERY / "tables" / "raw" / name
        shutil.copy2(src, dst)
        inventory.append({
            "file": name,
            "bytes": src.stat().st_size,
            "format": src.suffix.lower().lstrip(".") or "txt",
            "role": infer_table_role(name),
        })

    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv(DELIVERY / "tables" / "derived" / "data_inventory.csv", index=False)
    return inventory_df


def infer_table_role(name: str) -> str:
    lower = name.lower()
    if "typhoon_full" in lower:
        return "aligned track-precipitation feature table"
    if "all_years" in lower:
        return "interpolated typhoon best-track feature table"
    if "tif_features" in lower:
        return "GPM raster precipitation feature table"
    if "spearman" in lower:
        return "correlation matrix"
    if "importance" in lower:
        return "random forest feature importance"
    if "shap" in lower:
        return "SHAP intermediate feature table"
    if "sensitivity" in lower:
        return "scenario simulation metrics"
    if "virtual" in lower:
        return "virtual typhoon scenario definitions"
    return "supporting data"


def derive_tables() -> None:
    full_path = ROOT / "Typhoon_Full_Dataset_Q1.csv"
    if full_path.exists():
        df = pd.read_csv(full_path, parse_dates=["Time"])
        df["Year"] = df["Time"].dt.year
        summary = (
            df.groupby("Year")
            .agg(
                samples=("Typhoon_ID", "size"),
                typhoon_events=("Typhoon_ID", "nunique"),
                mean_pmax_mm_h=("P_max", "mean"),
                max_pmax_mm_h=("P_max", "max"),
                mean_extreme_area_km2=("S_ext_Extreme_over_20", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(DELIVERY / "tables" / "derived" / "dataset_summary_by_year.csv", index=False)
        write_hdf5_sample_metadata_from_csv(df)

    sensitivity_path = ROOT / "Sensitivity_Analysis_Results.csv"
    if sensitivity_path.exists():
        sens = pd.read_csv(sensitivity_path)
        base = sens.loc[sens["Scenario"] == "KONG-REY"].iloc[0]
        sens["delta_pmax_vs_kongrey_pct"] = (sens["P_max_mm_h"] / base["P_max_mm_h"] - 1.0) * 100.0
        sens["delta_sext_vs_kongrey_pct"] = (sens["S_ext_km2"] / base["S_ext_km2"] - 1.0) * 100.0
        sens.to_csv(DELIVERY / "tables" / "derived" / "scenario_metrics_with_delta.csv", index=False)

    rf_path = ROOT / "1.4_Data_RF_Raw_Importance.csv"
    if rf_path.exists():
        rf = pd.read_csv(rf_path)
        top = (
            rf.sort_values(["Target_Variable", "Raw_Importance_Score"], ascending=[True, False])
            .groupby("Target_Variable")
            .head(8)
        )
        top.to_csv(DELIVERY / "tables" / "derived" / "rf_top_features_by_target.csv", index=False)

    write_hdf5_metadata()


def write_hdf5_sample_metadata_from_csv(df: pd.DataFrame, seq_len: int = 12) -> None:
    """Recover sample-level metadata for the existing HDF5 sliding windows.

    The archived HDF5 only stores /data. The original builder grouped rows by
    Typhoon_ID, sorted by Time, and emitted stride-1 windows of length 12. This
    reconstruction yields 6921 rows, matching ConvLSTM_Dataset_128.h5 exactly.
    """
    rows = []
    sample_idx = 0
    for typhoon_id, group in df.groupby("Typhoon_ID", sort=True):
        group = group.sort_values("Time").reset_index(drop=True)
        n_frames = len(group)
        for start in range(0, n_frames - seq_len + 1):
            target = group.iloc[start + seq_len - 1]
            source = group.iloc[start]
            rows.append({
                "sample_idx": sample_idx,
                "typhoon_id": int(typhoon_id),
                "year": int(target["Time"].year),
                "start_time": source["Time"].isoformat(),
                "target_time": target["Time"].isoformat(),
                "target_filename": target.get("Filename", ""),
            })
            sample_idx += 1

    pd.DataFrame(rows).to_csv(
        DELIVERY / "tables" / "derived" / "hdf5_sample_metadata_from_csv.csv",
        index=False,
    )


def write_hdf5_metadata() -> None:
    h5_path = ROOT / "ConvLSTM_Dataset_128.h5"
    out = DELIVERY / "tables" / "derived" / "hdf5_metadata.csv"
    rows = []
    if not h5_path.exists():
        return

    try:
        import h5py

        with h5py.File(h5_path, "r") as f:
            def visit(name, obj):
                shape = getattr(obj, "shape", "")
                dtype = str(getattr(obj, "dtype", ""))
                rows.append({"name": name, "shape": str(shape), "dtype": dtype})

            f.visititems(visit)
    except Exception as exc:  # pragma: no cover - diagnostic export
        rows.append({"name": "ERROR", "shape": "", "dtype": repr(exc)})

    pd.DataFrame(rows).to_csv(out, index=False)


def copy_existing_figures() -> None:
    for name in EXISTING_FIGURE_NAMES:
        src = ARCHIVE_PAPER / name
        if not src.exists():
            continue
        shutil.copy2(src, DELIVERY / "figures" / "existing" / name)
        shutil.copy2(src, DELIVERY / "paper" / name)
        shutil.copy2(src, DELIVERY / "paper" / "figures" / name)


def generate_summary_figures() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.15,
        rc={
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "axes.edgecolor": "0.15",
            "axes.linewidth": 0.8,
            "grid.color": "0.88",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        },
    )
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6"]

    generated_dir = DELIVERY / "figures" / "generated"
    paper_dir = DELIVERY / "paper"

    sens_path = DELIVERY / "tables" / "derived" / "scenario_metrics_with_delta.csv"
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
        metrics = [
            ("P_max_mm_h", "Peak precipitation (mm/h)"),
            ("S_ext_km2", "Extreme-rain area (km$^2$)"),
            ("Duration_steps", "Risk duration (steps)"),
            ("P_mean_mm_h", "Mean precipitation (mm/h)"),
        ]
        order = sens.sort_values("P_max_mm_h", ascending=False)["Scenario"]
        for ax, (col, label) in zip(axes.flat, metrics):
            sns.barplot(
                data=sens,
                x="Scenario",
                y=col,
                hue="Scenario",
                order=order,
                hue_order=order,
                palette=palette,
                legend=False,
                ax=ax,
            )
            ax.set_xlabel("")
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=35)
        save_figure(fig, generated_dir / "fig_scenario_metrics_seaborn.png", paper_dir)

        topo = sens[sens["Scenario"].isin(["KONG-REY", "KONG-REY_NoTopo"])]
        if len(topo) == 2:
            topo_long = topo.melt(
                id_vars="Scenario",
                value_vars=["P_max_mm_h", "S_ext_km2"],
                var_name="Metric",
                value_name="Value",
            )
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
            sns.barplot(
                data=topo,
                x="Scenario",
                y="P_max_mm_h",
                hue="Scenario",
                palette=palette[:2],
                legend=False,
                ax=axes[0],
            )
            axes[0].set_ylabel("Peak precipitation (mm/h)")
            axes[0].set_xlabel("")
            sns.barplot(
                data=topo,
                x="Scenario",
                y="S_ext_km2",
                hue="Scenario",
                palette=palette[:2],
                legend=False,
                ax=axes[1],
            )
            axes[1].set_ylabel("Extreme-rain area (km$^2$)")
            axes[1].set_xlabel("")
            for ax in axes:
                ax.tick_params(axis="x", rotation=15)
            save_figure(fig, generated_dir / "fig_topography_ablation_seaborn.png", paper_dir)

    rf_path = ROOT / "1.4_Data_RF_Raw_Importance.csv"
    if rf_path.exists():
        rf = pd.read_csv(rf_path)
        focus = rf[rf["Target_Variable"].isin(["P_max", "S_ext_Extreme_over_20", "D_offset_km", "I_asy_Index"])]
        if focus.empty:
            focus = rf
        top = (
            focus.groupby("Feature_Name", as_index=False)["Raw_Importance_Score"]
            .mean()
            .sort_values("Raw_Importance_Score", ascending=False)
            .head(12)
        )
        fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
        sns.barplot(
            data=top,
            y="Feature_Name",
            x="Raw_Importance_Score",
            hue="Feature_Name",
            palette="deep",
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Mean raw RF importance")
        ax.set_ylabel("")
        save_figure(fig, generated_dir / "fig_rf_top_features_seaborn.png", paper_dir)

    corr_path = ROOT / "1.4_Data_Full_Spearman_Matrix.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path, index_col=0)
        keep_cols = [
            c for c in [
                "Wind_Speed", "Pressure", "Moving_Speed_kmh", "Curvature_deg_per_km",
                "Delta_P_6h", "Delta_V_6h", "P_total", "P_max",
                "S_ext_Extreme_over_20", "D_offset_km", "I_asy_Index",
            ]
            if c in corr.columns
        ]
        corr = corr.loc[keep_cols, keep_cols]
        fig, ax = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
        sns.heatmap(
            corr,
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.35,
            linecolor="white",
            square=True,
            cbar_kws={"label": "Spearman rho"},
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        save_figure(fig, generated_dir / "fig_spearman_heatmap_seaborn.png", paper_dir)

    full_path = ROOT / "Typhoon_Full_Dataset_Q1.csv"
    if full_path.exists():
        df = pd.read_csv(full_path, parse_dates=["Time"])
        df["Year"] = df["Time"].dt.year
        by_year = df.groupby("Year", as_index=False).agg(samples=("Typhoon_ID", "size"))
        fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
        sns.lineplot(data=by_year, x="Year", y="samples", marker="o", linewidth=2.2, color=palette[0], ax=ax)
        ax.set_ylabel("Aligned samples")
        ax.set_xlabel("Year")
        ax.set_title("Temporal coverage of aligned typhoon-precipitation samples")
        save_figure(fig, generated_dir / "fig_dataset_samples_by_year_seaborn.png", paper_dir)

    train_log = ROOT / "train_log.txt"
    if train_log.exists():
        records = parse_train_log(train_log.read_text(encoding="utf-8", errors="ignore"))
        if records:
            log_df = pd.DataFrame(records)
            log_df.to_csv(DELIVERY / "tables" / "derived" / "training_loss_from_existing_log.csv", index=False)
            fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
            sns.lineplot(data=log_df, x="epoch", y="train_loss", marker="o", label="Train", ax=ax)
            sns.lineplot(data=log_df, x="epoch", y="val_loss", marker="s", label="Validation", ax=ax)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE loss")
            ax.set_title("Existing competition training log")
            save_figure(fig, generated_dir / "fig_existing_training_loss_seaborn.png", paper_dir)


def save_figure(fig, path: Path, paper_dir: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    shutil.copy2(path, paper_dir / path.name)
    shutil.copy2(path, paper_dir / "figures" / path.name)
    fig.clf()


def parse_train_log(text: str) -> list[dict[str, float]]:
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+).*?Train Loss:\s*([0-9.]+)\s*\|\s*Val Loss:\s*([0-9.]+)",
        re.S,
    )
    return [
        {
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(3)),
            "val_loss": float(m.group(4)),
        }
        for m in pattern.finditer(text)
    ]


def create_latex_paper() -> None:
    source = ARCHIVE_PAPER / "essay.tex"
    if not source.exists():
        return

    text = source.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(
        r"\n% === 附录代码排版宏包与高级配色 ===.*?\\lstset\{style=pystyle\}\n",
        "\n",
        text,
        flags=re.S,
    )
    text = text.replace(
        "基于多模态同化与 PINN-ConvLSTM 的台风降水时空推演与气候演变情景模拟",
        "基于物理启发残差 ConvLSTM 的台风极端降水时空推演与气候情景模拟",
    )
    text = text.replace("PINN-ConvLSTM", "物理启发 Res-ConvLSTM")
    text = text.replace("PINN convLSTM", "物理启发 Res-ConvLSTM")
    text = text.replace("PINN-Res-ConvLSTM", "PI-ResConvLSTM")
    text = text.replace("PINN 时空引擎", "PI-ResConvLSTM 时空引擎")
    text = text.replace("PINN 物理约束网络", "物理启发残差时空网络")

    text = strip_code_appendix(text)
    text = inject_reproducibility_note(text)

    (DELIVERY / "paper" / "main.tex").write_text(text, encoding="utf-8")
    pdf_src = ARCHIVE_PAPER / "essay.pdf"
    if pdf_src.exists():
        shutil.copy2(pdf_src, DELIVERY / "paper" / "original_competition_essay.pdf")

    readme = """# Paper folder

`main.tex` is adapted from the existing competition essay and removes the source-code appendix.

Compile with XeLaTeX in a TeX Live or MiKTeX environment:

```bash
xelatex main.tex
xelatex main.tex
```

The file `original_competition_essay.pdf` is kept only as a visual reference from the earlier competition version.
"""
    (DELIVERY / "paper" / "README.md").write_text(readme, encoding="utf-8")


def strip_code_appendix(text: str) -> str:
    marker = "\n\\newpage\n\\appendix"
    if marker in text:
        text = text.split(marker)[0].rstrip()
        return text + "\n\\end{document}\n"

    marker = "\n\\appendix"
    if marker in text:
        text = text.split(marker)[0].rstrip()
        return text + "\n\\end{document}\n"

    return text


def inject_reproducibility_note(text: str) -> str:
    note = r"""

\section*{可复现性与交付说明}
本文最终交付包将核心材料划分为代码、表格、图表与论文四类：\texttt{code/} 中包含 PI-ResConvLSTM、物理启发损失函数、评估指标与推理脚本；\texttt{tables/} 中保留原始清洗 CSV、相关性矩阵、随机森林重要性与情景推演指标；\texttt{figures/} 中包含既有论文图与基于 seaborn 重新生成的白底汇总图；\texttt{paper/} 中提供可直接编译的 \LaTeX{} 正文文件。代码附录不再并入论文正文，以保证 paper 文件夹更接近期刊投稿格式。
"""
    ref_marker = "\n\\newpage\n\\renewcommand{\\refname}"
    if ref_marker in text:
        return text.replace(ref_marker, note + ref_marker)
    return text


def write_package_readme(inventory_df: pd.DataFrame) -> None:
    summary = {
        "package": str(DELIVERY),
        "raw_table_count": int(len(inventory_df)),
        "existing_figure_count": len(list((DELIVERY / "figures" / "existing").glob("*.png"))),
        "generated_figure_count": len(list((DELIVERY / "figures" / "generated").glob("*.png"))),
        "paper_main": "paper/main.tex",
        "code_root": "code/src",
    }
    (DELIVERY / "PACKAGE_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = f"""# PI-ResConvLSTM Paper Delivery Package

This folder is generated from the GeoAI workspace.

## Contents

- `code/`: refactored project source, configs, and delivery scripts.
- `figures/existing/`: reused figures from the competition paper.
- `figures/generated/`: lightweight seaborn summary figures regenerated from existing CSV files.
- `tables/raw/`: cleaned/original tabular artifacts kept as CSV/TXT.
- `tables/derived/`: paper-ready derived tables and metadata.
- `paper/`: LaTeX paper folder. Use `main.tex` for the final paper; code appendix has been removed.
- `models/`: existing trained checkpoint when available.

## Notes

The package reuses already generated heavy figures and model outputs. It does not rerun GPU training or reprocess the full GeoTIFF archive.

The existing `ConvLSTM_Dataset_128.h5` in the workspace contains a 4-channel tensor dataset and no embedded `/meta` group. The derived table `tables/derived/hdf5_sample_metadata_from_csv.csv` reconstructs sample-level metadata from the aligned CSV sliding-window order and matches the HDF5 sample count.

Raw table files included: {len(inventory_df)}
Generated seaborn figures: {summary["generated_figure_count"]}
"""
    (DELIVERY / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    reset_delivery_dir()
    copy_project_assets()
    inventory_df = copy_raw_tables()
    derive_tables()
    copy_existing_figures()
    generate_summary_figures()
    create_latex_paper()
    write_package_readme(inventory_df)
    print(f"Delivery package built: {DELIVERY}")


if __name__ == "__main__":
    main()
