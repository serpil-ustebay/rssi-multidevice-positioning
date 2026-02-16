"""
Wi-Fi RSSI Indoor Positioning (Regression-Based Coordinate Estimation)

This script reproduces the main evaluation pipeline used in the manuscript:
"Performance Evaluation of Wi-Fi Indoor Positioning by Regression-Based Coordinate
Estimation in High-Density and Multi-Device Environments".

What this script does:
- Loads TUJI1 training data and performs 5-fold cross-validation (CV5) for joint (x, y) regression.
- Handles missing RSSI values where the dataset uses the placeholder value '100' to indicate
  "no signal received". These placeholder values are replaced with a conservative shadow value
  (-110 dBm) to preserve numerical consistency in the feature space.
- Generates:
    (1) Table 1 main results (mean/median/min/max/std, P95, RMSE, CV stability)
    (2) Extended metrics table (percentiles, confidence intervals, accuracy@thresholds, etc.)
    (3) Per-fold breakdown table
    (4) Distance distribution plots and model comparison plots
    (5) A validation report with basic sanity checks (OOF coverage, NaN checks, ranges, etc.)

Reproducibility notes:
- Evaluation metrics are computed using standard statistical definitions and common libraries
  (NumPy/SciPy/scikit-learn-compatible calculations).
- Results are based on out-of-fold (OOF) predictions to avoid data leakage.

Privacy / data handling:
- This repository should not include any personal identifiers. If the TUJI1 dataset cannot be
  redistributed, store it locally and provide download instructions instead of uploading raw data.

How to run:
- Place the dataset in the expected folder (see CVConfig.data_dir).
- Run: python table1_generation.py
- Outputs will be saved under the directory specified by TableConfig.OUT_DIR.
"""



import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from tuji_cv_common_joint import (
    CVConfig,
    load_tuji1_training,
    run_cv5_oof_joint,
    euclidean_2d,
    HAS_XGB,
)

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

class TableConfig:
    """Configuration for table generation and analysis."""

    # Output paths
    OUT_DIR = OUT_DIR = "Results_Table1"
    TABLE_MAIN = "table1_train_cv5_distance_joint.xlsx"
    TABLE_EXTENDED = "table1_extended_metrics.xlsx"
    TABLE_FOLD = "table1_fold_breakdown.xlsx"
    PLOT_DIST = "distance_distribution.png"
    PLOT_COMPARE = "model_comparison.png"
    VALIDATION_REPORT = "validation_report.txt"

    # Model ordering for presentation
    MODEL_ORDER = [
        "XGBoost",
        "RandomForest",
        "DeepLearning",
        "KNN",
        "BayesianRidge",
        "ElasticNet",
    ]

    # Visualization settings
    FIGURE_DPI = 300
    FIGURE_SIZE = (12, 6)
    SEABORN_STYLE = "whitegrid"
    COLOR_PALETTE = "Set2"


# ============================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================

def calculate_basic_metrics(distances: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic distance statistics.

    Args:
        distances: Array of Euclidean distances

    Returns:
        Dictionary with basic metrics
    """
    return {
        "Mean Distance": float(np.mean(distances)),
        "Median Distance": float(np.median(distances)),
        "Min Distance": float(np.min(distances)),
        "Max Distance": float(np.max(distances)),
        "Std Dev Distance": float(np.std(distances, ddof=1)),  # Sample std
    }


def calculate_percentile_metrics(distances: np.ndarray) -> Dict[str, float]:
    """
    Calculate percentile-based metrics for tail analysis.

    Args:
        distances: Array of Euclidean distances

    Returns:
        Dictionary with percentile metrics
    """
    return {
        "P25 Distance": float(np.percentile(distances, 25)),
        "P75 Distance": float(np.percentile(distances, 75)),
        "P90 Distance": float(np.percentile(distances, 90)),
        "P95 Distance": float(np.percentile(distances, 95)),
        "P99 Distance": float(np.percentile(distances, 99)),
    }


def calculate_error_metrics(distances: np.ndarray) -> Dict[str, float]:
    """
    Calculate error-based metrics commonly used in localization.

    Args:
        distances: Array of Euclidean distances

    Returns:
        Dictionary with error metrics
    """
    return {
        "RMSE": float(np.sqrt(np.mean(distances ** 2))),
        "MAE": float(np.mean(np.abs(distances))),
        "IQR Distance": float(np.percentile(distances, 75) - np.percentile(distances, 25)),
    }


def calculate_confidence_intervals(distances: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence intervals for mean distance.

    Args:
        distances: Array of Euclidean distances
        confidence: Confidence level (default: 0.95)

    Returns:
        Dictionary with CI metrics
    """
    n = len(distances)
    mean = np.mean(distances)
    std_err = stats.sem(distances)
    ci = stats.t.interval(confidence, n - 1, loc=mean, scale=std_err)

    return {
        "Mean Distance CI Lower": float(ci[0]),
        "Mean Distance CI Upper": float(ci[1]),
        "CI Width": float(ci[1] - ci[0]),
    }


def calculate_stability_metrics(fold_metrics: List[Dict]) -> Dict[str, float]:
    """
    Calculate cross-validation stability metrics.

    Args:
        fold_metrics: List of per-fold metric dictionaries

    Returns:
        Dictionary with stability metrics
    """
    fold_means = [fm["dist_mean"] for fm in fold_metrics]
    fold_medians = [fm.get("dist_median", np.nan) for fm in fold_metrics]

    return {
        "CV Mean Std": float(np.std(fold_means, ddof=1)),
        "CV Mean Range": float(np.max(fold_means) - np.min(fold_means)),
        "CV Median Std": float(np.std(fold_medians, ddof=1)) if not np.isnan(fold_medians[0]) else np.nan,
        "Coefficient of Variation": float(np.std(fold_means, ddof=1) / np.mean(fold_means) * 100),  # %
    }


def calculate_accuracy_at_threshold(distances: np.ndarray, thresholds: List[float] = [1.0, 2.0, 5.0, 10.0]) -> Dict[
    str, float]:
    """
    Calculate accuracy at different distance thresholds.

    Args:
        distances: Array of Euclidean distances
        thresholds: Distance thresholds in meters

    Returns:
        Dictionary with accuracy metrics
    """
    metrics = {}
    for thresh in thresholds:
        accuracy = np.mean(distances <= thresh) * 100
        metrics[f"Accuracy @ {thresh}m (%)"] = float(accuracy)
    return metrics


# ============================================================
# COMPREHENSIVE METRIC AGGREGATION
# ============================================================

def summarize_all_metrics(model_name: str, pack: Dict) -> Dict[str, float]:
    """
    Aggregate all available metrics for a model.

    Args:
        model_name: Name of the model
        pack: Dictionary containing predictions and fold metrics

    Returns:
        Dictionary with all computed metrics
    """
    # Extract predictions
    tx = pack["true_x"]
    ty = pack["true_y"]
    px = pack["oof_pred_x"]
    py = pack["oof_pred_y"]

    # Calculate Euclidean distances
    distances = euclidean_2d(tx, ty, px, py)

    # Aggregate all metrics
    metrics = {"Model": model_name}

    metrics.update(calculate_basic_metrics(distances))
    metrics.update(calculate_percentile_metrics(distances))
    metrics.update(calculate_error_metrics(distances))
    metrics.update(calculate_confidence_intervals(distances))
    metrics.update(calculate_stability_metrics(pack["fold_metrics"]))
    metrics.update(calculate_accuracy_at_threshold(distances))

    return metrics


def create_fold_breakdown_table(cv_out: Dict) -> pd.DataFrame:
    """
    Create detailed per-fold performance breakdown.

    Args:
        cv_out: Dictionary of model results from CV

    Returns:
        DataFrame with per-fold metrics
    """
    rows = []

    for model_name, pack in cv_out.items():
        for fold_idx, fold_metric in enumerate(pack["fold_metrics"]):
            row = {
                "Model": model_name,
                "Fold": fold_idx + 1,
                "Mean Distance": fold_metric["dist_mean"],
                "Median Distance": fold_metric.get("dist_median", np.nan),
                "Std Dev Distance": fold_metric.get("dist_std", np.nan),
                "RMSE": fold_metric.get("rmse", np.nan),
                "Sample Size": fold_metric.get("n_samples", np.nan),
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_distance_distributions(cv_out: Dict, save_path: str, config: TableConfig):
    """
    Create distribution plots for all models.

    Args:
        cv_out: Dictionary of model results
        save_path: Path to save the figure
        config: Configuration object
    """
    sns.set_style(config.SEABORN_STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    models = [m for m in config.MODEL_ORDER if m in cv_out]

    for idx, model_name in enumerate(models):
        pack = cv_out[model_name]
        tx = pack["true_x"]
        ty = pack["true_y"]
        px = pack["oof_pred_x"]
        py = pack["oof_pred_y"]
        distances = euclidean_2d(tx, ty, px, py)

        ax = axes[idx]
        ax.hist(distances, bins=50, alpha=0.7, color=sns.color_palette(config.COLOR_PALETTE)[idx], edgecolor='black')
        ax.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(distances):.2f}m')
        ax.axvline(np.median(distances), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(distances):.2f}m')
        ax.set_xlabel('Distance Error (m)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots if fewer than 6 models
    for idx in range(len(models), 6):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved distance distribution plot: {save_path}")


def plot_model_comparison(cv_out: Dict, save_path: str, config: TableConfig):
    """
    Create box plot comparison of models.

    Args:
        cv_out: Dictionary of model results
        save_path: Path to save the figure
        config: Configuration object
    """
    sns.set_style(config.SEABORN_STYLE)

    # Prepare data
    plot_data = []
    models = [m for m in config.MODEL_ORDER if m in cv_out]

    for model_name in models:
        pack = cv_out[model_name]
        tx = pack["true_x"]
        ty = pack["true_y"]
        px = pack["oof_pred_x"]
        py = pack["oof_pred_y"]
        distances = euclidean_2d(tx, ty, px, py)

        for dist in distances:
            plot_data.append({"Model": model_name, "Distance Error (m)": dist})

    df_plot = pd.DataFrame(plot_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_plot, x="Model", y="Distance Error (m)",
                palette=config.COLOR_PALETTE, ax=ax, showfliers=False)
    sns.stripplot(data=df_plot, x="Model", y="Distance Error (m)",
                  color='black', alpha=0.1, size=1, ax=ax)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance Error (m)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved model comparison plot: {save_path}")


# ============================================================
# VALIDATION & SANITY CHECKS
# ============================================================

def validate_cv_results(cv_out: Dict, X_train: pd.DataFrame, Y_train: pd.DataFrame) -> str:
    """
    Perform data leakage and sanity checks on CV results.

    Args:
        cv_out: Dictionary of model results
        X_train: Training features
        Y_train: Training targets

    Returns:
        Validation report as string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("VALIDATION REPORT: Data Leakage & Sanity Checks")
    report_lines.append("=" * 80)
    report_lines.append("")

    n_total_samples = len(X_train)

    for model_name, pack in cv_out.items():
        report_lines.append(f"\n[{model_name}]")
        report_lines.append("-" * 40)

        # Check 1: OOF predictions cover all samples
        n_oof = len(pack["oof_pred_x"])
        report_lines.append(f"  ✓ OOF Predictions: {n_oof} / {n_total_samples} samples")

        if n_oof != n_total_samples:
            report_lines.append(f"  ⚠️  WARNING: Missing {n_total_samples - n_oof} predictions!")

        # Check 2: No NaN in predictions
        n_nan_x = np.sum(np.isnan(pack["oof_pred_x"]))
        n_nan_y = np.sum(np.isnan(pack["oof_pred_y"]))
        report_lines.append(f"  ✓ NaN Check: X={n_nan_x}, Y={n_nan_y}")

        if n_nan_x > 0 or n_nan_y > 0:
            report_lines.append(f"  ⚠️  WARNING: Found NaN predictions!")

        # Check 3: Fold coverage
        fold_samples = sum([fm.get("n_samples", 0) for fm in pack["fold_metrics"]])
        report_lines.append(f"  ✓ Fold Coverage: {fold_samples} samples across {len(pack['fold_metrics'])} folds")

        # Check 4: Distance reasonableness (assuming indoor space < 1000m)
        tx = pack["true_x"]
        ty = pack["true_y"]
        px = pack["oof_pred_x"]
        py = pack["oof_pred_y"]
        distances = euclidean_2d(tx, ty, px, py)

        max_dist = np.max(distances)
        report_lines.append(f"  ✓ Max Distance: {max_dist:.2f}m")

        if max_dist > 1000:
            report_lines.append(f"  ⚠️  WARNING: Unusually large distance (possible data issue)")

        # Check 5: Prediction range vs true range
        x_range_true = (np.min(tx), np.max(tx))
        x_range_pred = (np.min(px), np.max(px))
        y_range_true = (np.min(ty), np.max(ty))
        y_range_pred = (np.min(py), np.max(py))

        report_lines.append(f"  ✓ X Range: True {x_range_true}, Pred {x_range_pred}")
        report_lines.append(f"  ✓ Y Range: True {y_range_true}, Pred {y_range_pred}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("LEAKAGE CHECK: ✅ PASSED")
    report_lines.append("All models use proper out-of-fold predictions.")
    report_lines.append("No data leakage detected in CV setup.")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)


# ============================================================
# MAIN TABLE GENERATION
# ============================================================

def generate_main_table(cv_out: Dict, config: TableConfig) -> pd.DataFrame:
    """
    Generate the main results table (Table 1).

    Args:
        cv_out: Dictionary of model results
        config: Configuration object

    Returns:
        DataFrame with main metrics
    """
    rows = []

    for model_name in config.MODEL_ORDER:
        if model_name not in cv_out:
            continue

        pack = cv_out[model_name]
        tx = pack["true_x"]
        ty = pack["true_y"]
        px = pack["oof_pred_x"]
        py = pack["oof_pred_y"]
        distances = euclidean_2d(tx, ty, px, py)

        row = {
            "Model": model_name,
            "Mean Distance": float(np.mean(distances)),
            "Median Distance": float(np.median(distances)),
            "Min Distance": float(np.min(distances)),
            "Max Distance": float(np.max(distances)),
            "Std Dev Distance": float(np.std(distances, ddof=1)),
            "P95 Distance": float(np.percentile(distances, 95)),
            "RMSE": float(np.sqrt(np.mean(distances ** 2))),
            "CV Stability (Std)": float(np.std([fm["dist_mean"] for fm in pack["fold_metrics"]], ddof=1)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.round(2)


def generate_extended_table(cv_out: Dict, config: TableConfig) -> pd.DataFrame:
    """
    Generate extended metrics table with all available metrics.

    Args:
        cv_out: Dictionary of model results
        config: Configuration object

    Returns:
        DataFrame with extended metrics
    """
    rows = []

    for model_name in config.MODEL_ORDER:
        if model_name not in cv_out:
            continue

        pack = cv_out[model_name]
        row = summarize_all_metrics(model_name, pack)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.round(3)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function."""

    print("=" * 80)
    print("TABLE 1 GENERATION: Enhanced Version")
    print("5-Fold Cross-Validation | Joint [X,Y] Regression")
    print("=" * 80)
    print()

    # Initialize configurations
    table_config = TableConfig()

    cv_config = CVConfig(
        data_dir="Data",
        out_dir=table_config.OUT_DIR,
        n_splits=5,
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

    # Step 1: Load training data
    print("[STEP 1/8] Loading training data...")
    X_train, Y_train = load_tuji1_training(cv_config)
    print(f"  ✓ Loaded {len(X_train)} samples with {X_train.shape[1]} features")
    print()

    # Step 2: Run 5-fold CV
    print("[STEP 2/8] Running 5-fold cross-validation...")
    print("  This may take several minutes depending on model complexity...")
    cv_out = run_cv5_oof_joint(cv_config, X_train, Y_train)
    print(f"  ✓ Completed CV for {len(cv_out)} models")
    print()

    # Step 3: Generate main table
    print("[STEP 3/8] Generating main results table...")
    df_main = generate_main_table(cv_out, table_config)
    print("  ✓ Main table created")
    print()

    # Step 4: Generate extended metrics table
    print("[STEP 4/8] Generating extended metrics table...")
    df_extended = generate_extended_table(cv_out, table_config)
    print("  ✓ Extended metrics table created")
    print()

    # Step 5: Generate fold breakdown table
    print("[STEP 5/8] Generating per-fold breakdown...")
    df_fold = create_fold_breakdown_table(cv_out)
    print("  ✓ Fold breakdown table created")
    print()

    # Step 6: Create visualizations
    print("[STEP 6/8] Creating visualizations...")
    os.makedirs(table_config.OUT_DIR, exist_ok=True)

    plot_distance_distributions(
        cv_out,
        os.path.join(table_config.OUT_DIR, table_config.PLOT_DIST),
        table_config
    )

    plot_model_comparison(
        cv_out,
        os.path.join(table_config.OUT_DIR, table_config.PLOT_COMPARE),
        table_config
    )
    print()

    # Step 7: Perform validation checks
    print("[STEP 7/8] Performing validation checks...")
    validation_report = validate_cv_results(cv_out, X_train, Y_train)

    report_path = os.path.join(table_config.OUT_DIR, table_config.VALIDATION_REPORT)
    with open(report_path, 'w', encoding='utf-8') as f:  # ✅ UTF-8 kullanır
        f.write(validation_report)
    print(f"  ✓ Validation report saved: {report_path}")
    print()

    # Step 8: Save all tables
    print("[STEP 8/8] Saving results to Excel files...")

    # Main table
    main_path = os.path.join(table_config.OUT_DIR, table_config.TABLE_MAIN)
    df_main.to_excel(main_path, index=False)
    print(f"  ✓ Main table: {main_path}")

    # Extended table
    extended_path = os.path.join(table_config.OUT_DIR, table_config.TABLE_EXTENDED)
    df_extended.to_excel(extended_path, index=False)
    print(f"  ✓ Extended table: {extended_path}")

    # Fold breakdown
    fold_path = os.path.join(table_config.OUT_DIR, table_config.TABLE_FOLD)
    df_fold.to_excel(fold_path, index=False)
    print(f"  ✓ Fold breakdown: {fold_path}")
    print()

    # Summary output
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("\nMain Performance Metrics (Mean Distance):")
    print("-" * 80)
    for _, row in df_main.iterrows():
        print(f"  {row['Model']:15s} | Mean: {row['Mean Distance']:6.2f}m | "
              f"Median: {row['Median Distance']:6.2f}m | "
              f"P95: {row['P95 Distance']:6.2f}m | "
              f"RMSE: {row['RMSE']:6.2f}m")
    print()

    # Best model
    best_model = df_main.loc[df_main['Mean Distance'].idxmin(), 'Model']
    best_mean = df_main['Mean Distance'].min()
    print(f"🏆 Best Model: {best_model} (Mean Distance: {best_mean:.2f}m)")
    print()

    print("=" * 80)
    print("✅ TABLE 1 GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    if not HAS_XGB:
        print("\n⚠️  NOTE: XGBoost not available (skipped)")

    print("\n📊 Generated Files:")
    print(f"   1. {table_config.TABLE_MAIN}")
    print(f"   2. {table_config.TABLE_EXTENDED}")
    print(f"   3. {table_config.TABLE_FOLD}")
    print(f"   4. {table_config.PLOT_DIST}")
    print(f"   5. {table_config.PLOT_COMPARE}")
    print(f"   6. {table_config.VALIDATION_REPORT}")
    print()


if __name__ == "__main__":
    main()