"""
Wi-Fi RSSI Indoor Positioning – TEST Evaluation (Full Train → Test)

This script reproduces the TEST set evaluation (Table 2) reported in the study:
"Performance Evaluation of Wi-Fi Indoor Positioning by Regression-Based
Coordinate Estimation in High-Density and Multi-Device Environments".

Pipeline Overview:
- Loads TUJI1 training and testing splits.
- Trains each regression model using the full training set.
- Evaluates performance on the independent test set.
- Performs joint multi-output regression for (x, y) coordinate estimation.
- Computes Euclidean distance errors between predicted and true coordinates.
- Generates:
    (1) Main test results table
    (2) Extended metrics table
    (3) Distance distribution plots
    (4) Model comparison plots
    (5) Sanity report for numerical consistency checks

Data Handling:
- The dataset uses the placeholder value '100' to indicate no received signal.
- These values are replaced with a conservative shadow value (-110 dBm).
- Feature scaling and column filtering are applied consistently based on training data.

Reproducibility:
- All evaluation metrics follow standard statistical definitions.
- No data leakage occurs: training and test sets are strictly separated.
- Results are fully deterministic when the random seed is fixed.

Privacy Notice:
- This repository does not contain personal identifiers.
- If dataset redistribution is restricted, users must obtain TUJI1 independently
  and place it under the expected data directory.

Execution:
- Ensure the dataset is placed under the configured data directory.
- Run: python table2_generation.py
- Outputs will be saved in the directory specified by TableConfig.OUT_DIR.
"""



import os
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from tuji_cv_common_joint import (
    CVConfig,
    load_tuji1_training,
    load_tuji1_testing,
    run_fulltrain_test_joint,
    euclidean_2d,
    HAS_XGB,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

class TableConfig:
    OUT_DIR = "Results_Table2"
    TABLE_MAIN = "table2_test_main.xlsx"
    TABLE_EXTENDED = "table2_test_extended.xlsx"
    PLOT_DIST = "distance_distribution.png"
    PLOT_COMPARE = "model_comparison.png"
    REPORT = "sanity_report.txt"

    MODEL_ORDER = [
        "XGBoost",
        "RandomForest",
        "DeepLearning",
        "KNN",
        "BayesianRidge",
        "ElasticNet",
    ]

    FIGURE_DPI = 300
    SEABORN_STYLE = "whitegrid"
    COLOR_PALETTE = "Set2"


# ============================================================
# METRICS
# ============================================================

def calculate_basic_metrics(distances: np.ndarray) -> Dict[str, float]:
    return {
        "Mean Distance": float(np.mean(distances)),
        "Median Distance": float(np.median(distances)),
        "Min Distance": float(np.min(distances)),
        "Max Distance": float(np.max(distances)),
        "Std Dev Distance": float(np.std(distances, ddof=1)) if len(distances) > 1 else 0.0,
    }


def calculate_percentile_metrics(distances: np.ndarray) -> Dict[str, float]:
    return {
        "P25 Distance": float(np.percentile(distances, 25)),
        "P75 Distance": float(np.percentile(distances, 75)),
        "P90 Distance": float(np.percentile(distances, 90)),
        "P95 Distance": float(np.percentile(distances, 95)),
        "P99 Distance": float(np.percentile(distances, 99)),
    }


def calculate_error_metrics(distances: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(np.mean(distances ** 2))),
        "MAE": float(np.mean(np.abs(distances))),
        "IQR Distance": float(np.percentile(distances, 75) - np.percentile(distances, 25)),
    }


def calculate_confidence_intervals(distances: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
    n = len(distances)
    mean = float(np.mean(distances))
    if n < 2:
        return {
            "Mean Distance CI Lower": mean,
            "Mean Distance CI Upper": mean,
            "CI Width": 0.0,
        }

    std_err = stats.sem(distances)
    ci = stats.t.interval(confidence, n - 1, loc=mean, scale=std_err)
    return {
        "Mean Distance CI Lower": float(ci[0]),
        "Mean Distance CI Upper": float(ci[1]),
        "CI Width": float(ci[1] - ci[0]),
    }


def calculate_accuracy_at_threshold(distances: np.ndarray, thresholds: List[float] = [1.0, 2.0, 5.0, 10.0]) -> Dict[str, float]:
    out = {}
    for t in thresholds:
        out[f"Accuracy @ {t}m (%)"] = float(np.mean(distances <= t) * 100.0)
    return out


def summarize_all_metrics(model_name: str, pack: Dict) -> Dict[str, float]:
    tx = pack["true_x"]
    ty = pack["true_y"]
    px = pack["pred_x"]
    py = pack["pred_y"]

    distances = euclidean_2d(tx, ty, px, py)

    m = {"Model": model_name}
    m.update(calculate_basic_metrics(distances))
    m.update(calculate_percentile_metrics(distances))
    m.update(calculate_error_metrics(distances))
    m.update(calculate_confidence_intervals(distances))
    m.update(calculate_accuracy_at_threshold(distances))
    return m


# ============================================================
# TABLES
# ============================================================

def generate_main_table(test_out: Dict, config: TableConfig) -> pd.DataFrame:
    rows = []
    for model_name in config.MODEL_ORDER:
        if model_name not in test_out:
            continue

        pack = test_out[model_name]
        dist = euclidean_2d(pack["true_x"], pack["true_y"], pack["pred_x"], pack["pred_y"])

        rows.append({
            "Model": model_name,
            "Mean Distance": float(np.mean(dist)),
            "Median Distance": float(np.median(dist)),
            "Min Distance": float(np.min(dist)),
            "Max Distance": float(np.max(dist)),
            "Std Dev Distance": float(np.std(dist, ddof=1)) if len(dist) > 1 else 0.0,
            "P95 Distance": float(np.percentile(dist, 95)),
            "RMSE": float(np.sqrt(np.mean(dist ** 2))),
        })

    return pd.DataFrame(rows).round(2)


def generate_extended_table(test_out: Dict, config: TableConfig) -> pd.DataFrame:
    rows = []
    for model_name in config.MODEL_ORDER:
        if model_name not in test_out:
            continue
        rows.append(summarize_all_metrics(model_name, test_out[model_name]))
    return pd.DataFrame(rows).round(3)


# ============================================================
# PLOTS
# ============================================================

def plot_distance_distributions(test_out: Dict, save_path: str, config: TableConfig):
    sns.set_style(config.SEABORN_STYLE)

    models = [m for m in config.MODEL_ORDER if m in test_out]
    if len(models) == 0:
        print("[WARN] No models to plot (distance distributions).")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    palette = sns.color_palette(config.COLOR_PALETTE, n_colors=max(3, len(models)))

    for idx, model_name in enumerate(models):
        pack = test_out[model_name]
        dist = euclidean_2d(pack["true_x"], pack["true_y"], pack["pred_x"], pack["pred_y"])

        ax = axes[idx]
        ax.hist(dist, bins=50, alpha=0.75, color=palette[idx], edgecolor="black")
        ax.axvline(np.mean(dist), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(dist):.2f}m")
        ax.axvline(np.median(dist), color="blue", linestyle="--", linewidth=2, label=f"Median: {np.median(dist):.2f}m")
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Distance Error (m)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for j in range(len(models), 6):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_model_comparison(test_out: Dict, save_path: str, config: TableConfig):
    sns.set_style(config.SEABORN_STYLE)

    models = [m for m in config.MODEL_ORDER if m in test_out]
    if len(models) == 0:
        print("[WARN] No models to plot (comparison).")
        return

    rows = []
    for model_name in models:
        pack = test_out[model_name]
        dist = euclidean_2d(pack["true_x"], pack["true_y"], pack["pred_x"], pack["pred_y"])
        rows.extend([{"Model": model_name, "Distance Error (m)": float(d)} for d in dist])

    df_plot = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_plot, x="Model", y="Distance Error (m)", palette=config.COLOR_PALETTE, ax=ax, showfliers=False)
    sns.stripplot(data=df_plot, x="Model", y="Distance Error (m)", color="black", alpha=0.12, size=1, ax=ax)

    ax.set_title("Model Performance Comparison (TEST)", fontweight="bold")
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel("Distance Error (m)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {save_path}")


# ============================================================
# SANITY REPORT
# ============================================================

def build_sanity_report(test_out: Dict) -> str:
    lines = []
    lines.append("SANITY REPORT (FULL TRAIN -> TEST)\n")

    for name, pack in test_out.items():
        tx, ty = pack["true_x"], pack["true_y"]
        px, py = pack["pred_x"], pack["pred_y"]
        dist = euclidean_2d(tx, ty, px, py)

        lines.append(f"--- {name} ---")
        lines.append(f"n = {len(dist)}")
        lines.append(f"pred_x finite% = {np.isfinite(px).mean():.4f} | min={np.nanmin(px):.4f} | max={np.nanmax(px):.4f}")
        lines.append(f"pred_y finite% = {np.isfinite(py).mean():.4f} | min={np.nanmin(py):.4f} | max={np.nanmax(py):.4f}")
        lines.append(f"dist  finite% = {np.isfinite(dist).mean():.4f} | mean={np.mean(dist):.4f} | median={np.median(dist):.4f} | p95={np.percentile(dist,95):.4f}")
        if "dropped_cols" in pack:
            lines.append(f"dropped_cols (train-based all-missing) = {len(pack['dropped_cols'])}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("TABLE 2 (TEST) - Using tuji_cv_common_joint MODELS")
    print("FULL TRAIN -> TEST | Joint [x,y]")
    print("=" * 80)

    cfg = CVConfig(
        data_dir="Data",
        out_dir=TableConfig.OUT_DIR,
        n_splits=5,          # burada kullanılmıyor
        seed=42,
        missing_value=100.0,
        impute_strategy="constant",
        impute_fill_value=-110.0,
        scale=True,
        drop_all_missing_cols=True,
        verbose=True,
        fold_verbose=False,
        print_device_counts=True,
        print_floor_counts=True,
        print_building_counts=True,
    )

    table_cfg = TableConfig()
    os.makedirs(table_cfg.OUT_DIR, exist_ok=True)

    print("[1/5] Load TRAIN ...")
    X_train, Y_train = load_tuji1_training(cfg)

    print("[2/5] Load TEST ...")
    X_test, Y_test = load_tuji1_testing(cfg)

    print("[3/5] Run FULL TRAIN -> TEST for all models (your definitions) ...")
    test_out = run_fulltrain_test_joint(cfg, X_train, Y_train, X_test, Y_test)

    print("[4/5] Build tables ...")
    df_main = generate_main_table(test_out, table_cfg)
    df_ext  = generate_extended_table(test_out, table_cfg)

    main_path = os.path.join(table_cfg.OUT_DIR, table_cfg.TABLE_MAIN)
    ext_path  = os.path.join(table_cfg.OUT_DIR, table_cfg.TABLE_EXTENDED)
    df_main.to_excel(main_path, index=False)
    df_ext.to_excel(ext_path, index=False)

    print(f"[OK] Saved: {main_path}")
    print(f"[OK] Saved: {ext_path}")

    print("[5/5] Plots + sanity report ...")
    plot_distance_distributions(test_out, os.path.join(table_cfg.OUT_DIR, table_cfg.PLOT_DIST), table_cfg)
    plot_model_comparison(test_out, os.path.join(table_cfg.OUT_DIR, table_cfg.PLOT_COMPARE), table_cfg)

    rep = build_sanity_report(test_out)
    rep_path = os.path.join(table_cfg.OUT_DIR, table_cfg.REPORT)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep)
    print(f"[OK] Saved: {rep_path}")

    print("\nSummary (Mean Distance):")
    for _, r in df_main.iterrows():
        print(f"  {r['Model']:15s} mean={r['Mean Distance']:6.2f}  median={r['Median Distance']:6.2f}  p95={r['P95 Distance']:6.2f}  rmse={r['RMSE']:6.2f}")

    if len(df_main) > 0:
        best = df_main.loc[df_main["Mean Distance"].idxmin(), "Model"]
        best_mean = df_main["Mean Distance"].min()
        print(f"\nBest Model: {best} (Mean={best_mean:.2f}m)")

    if not HAS_XGB:
        print("\n[WARN] HAS_XGB=False -> XGBoost skipped.")


if __name__ == "__main__":
    main()
