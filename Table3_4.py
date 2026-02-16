"""
Purpose:
- This script evaluates Wi-Fi RSSI-based indoor positioning models by focusing on
  coordinate-wise accuracy (X and Y) rather than Euclidean distance error.

Key idea:
- Instead of reporting only distance-based localization error, this analysis quantifies
  how accurately each model predicts the individual X and Y coordinates. This is useful
  for diagnosing axis-specific bias, variance, and systematic drift that may be hidden
  when using only distance metrics.

What it does:
1) Loads TUJI1 RSSI features and (X, Y) coordinates from separate CSV files.
2) Handles missing RSSI values:
   - The dataset uses the placeholder value '100' to indicate "no signal received".
   - These placeholders are replaced with a conservative shadow RSSI value (-110 dBm).
3) Runs two evaluation phases:
   - TRAIN_CV5: 5-fold cross-validation using out-of-fold predictions (no leakage).
   - TEST: trains on the full training split and evaluates on an independent test split.
4) Computes coordinate-focused metrics for X and Y separately:
   - MAE, RMSE, R², explained variance
   - mean error (bias), error spread (std), median/P95/max absolute errors
   - normalized RMSE (NRMSE) and percentage-based error (MAPE; used only as a reference)
5) Produces diagnostic visualizations (optional):
   - True vs Predicted scatter plots (per axis)
   - Error histograms per axis
   - Residual plots (true/predicted vs residuals)
   - Model comparison charts across key coordinate metrics
6) Saves results to CSV/XLSX under the configured output directory.

Reproducibility and privacy:
- Fixed random seed is used for deterministic CV splits and model training.
- No personal identifiers are included in this repository.
- If dataset redistribution is restricted, do not upload raw TUJI1 files; provide only
  download instructions and expected file paths.

How to run:
- Place input CSV files under the paths defined in the CONFIG section.
- Run: python xy_coordinate_accuracy_focused.py
- Outputs will be written to the OUT_DIR directory.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from copy import deepcopy

from typing import Dict, Any, Tuple

from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import BayesianRidge, MultiTaskElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# --- Optional XGBoost ---
HAS_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False

# -------------------------------
# CONFIG
# -------------------------------
SEED = 42
N_SPLITS = 5

RSS_TRAIN_PATH = "Data/RSS_training.csv"
COORD_TRAIN_PATH = "Data/Coordinates_training.csv"

RSS_TEST_PATH = "Data/RSS_testing.csv"
COORD_TEST_PATH = "Data/Coordinates_testing.csv"

OUT_DIR = "Table 3 4"
os.makedirs(OUT_DIR, exist_ok=True)

MAKE_PLOTS = True


# -------------------------------
# UTIL LOG
# -------------------------------
def log(msg: str):
    print(f"[INFO] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def err(msg: str):
    print(f"[ERROR] {msg}")


# -------------------------------
# DATA LOADING
# -------------------------------
def load_xy_from_separate_files(rss_path: str, coord_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    RSS CSV : features only
    Coord CSV: first two columns are X,Y
    RSS value 100 indicates missing -> replaced with -110
    """
    log(f"Reading RSS: {os.path.abspath(rss_path)}")
    log(f"Reading COORD: {os.path.abspath(coord_path)}")

    if not os.path.isfile(rss_path):
        raise FileNotFoundError(f"RSS file not found: {rss_path}")
    if not os.path.isfile(coord_path):
        raise FileNotFoundError(f"Coordinate file not found: {coord_path}")

    X_raw = pd.read_csv(rss_path, header=None)
    Y_df = pd.read_csv(coord_path, header=None)

    if Y_df.shape[1] < 2:
        raise ValueError("Coordinate file must have at least two columns (X, Y)")

    # --- RSS numeric conversion
    X = X_raw.apply(pd.to_numeric, errors="coerce")

    # Count 100s BEFORE replace
    n_100 = int((X == 100).sum().sum())
    log(f"RSS shape={X.shape} | count(value==100)={n_100}")

    # Replace missing marker
    X = X.replace(100, -110)

    # Fill NaN if any
    n_nan = int(X.isna().sum().sum())
    if n_nan > 0:
        warn(f"RSS has NaN after numeric conversion: {n_nan} cells -> filling with -110")
        X = X.fillna(-110)

    log(f"RSS min={float(np.nanmin(X.values)):.2f} | max={float(np.nanmax(X.values)):.2f}")

    # --- Coordinates: first 2 columns
    y = Y_df.iloc[:, :2].to_numpy(dtype=float)

    log(f"COORD shape={y.shape} | X range=({y[:, 0].min():.2f},{y[:, 0].max():.2f}) "
        f"| Y range=({y[:, 1].min():.2f},{y[:, 1].max():.2f})")

    if len(X) != len(y):
        raise ValueError(f"Row count mismatch: RSS rows={len(X)} vs COORD rows={len(y)}")

    return X, y


# -------------------------------
# MODELS
# -------------------------------
def get_models(seed: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1
        ),
        "DeepLearning": MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=seed
        ),

        "KNN": KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
            metric="euclidean"
        ),

        "BayesianRidge": BayesianRidge(),
        "ElasticNet": MultiTaskElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=seed
        )    }

    return models


def ensure_multioutput(estimator: Any) -> Any:
    """Wrap single-output models with MultiOutputRegressor"""
    single_output_types = (BayesianRidge,)
    if HAS_XGB:
        single_output_types = single_output_types + (XGBRegressor,)
    if isinstance(estimator, single_output_types):
        return MultiOutputRegressor(estimator)
    return estimator


# -------------------------------
# METRICS - X ve Y ODAKLI
# -------------------------------
def compute_xy_focused_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    X ve Y koordinatlarının tahmin doğruluğuna odaklı metrikler.
    Distance metrikleri YOK - sadece X ve Y'nin gerçek değerlere yakınlığı.
    """
    yt_x, yt_y = y_true[:, 0], y_true[:, 1]
    yp_x, yp_y = y_pred[:, 0], y_pred[:, 1]

    # Hatalar
    err_x = yp_x - yt_x
    err_y = yp_y - yt_y

    # Mutlak hatalar
    abs_err_x = np.abs(err_x)
    abs_err_y = np.abs(err_y)

    # Coordinate ranges for normalization
    range_x = yt_x.max() - yt_x.min()
    range_y = yt_y.max() - yt_y.min()

    out = {
        # === X KOORDİNATI DOĞRULUĞU ===
        "MAE_X": float(mean_absolute_error(yt_x, yp_x)),
        "RMSE_X": float(np.sqrt(mean_squared_error(yt_x, yp_x))),
        "R2_X": float(r2_score(yt_x, yp_x)),
        "ExplainedVar_X": float(explained_variance_score(yt_x, yp_x)),
        "Mean_Err_X": float(np.mean(err_x)),  # Bias (sistematik hata)
        "Std_Err_X": float(np.std(err_x)),  # Spread
        "Median_AbsErr_X": float(np.median(abs_err_x)),
        "P95_AbsErr_X": float(np.percentile(abs_err_x, 95)),
        "Max_AbsErr_X": float(np.max(abs_err_x)),
        "MAPE_X": float(np.mean(abs_err_x / (np.abs(yt_x) + 1e-8)) * 100),  # Mean Absolute Percentage Error
        "NRMSE_X": float(np.sqrt(mean_squared_error(yt_x, yp_x)) / range_x) if range_x > 0 else 0.0,

        # === Y KOORDİNATI DOĞRULUĞU ===
        "MAE_Y": float(mean_absolute_error(yt_y, yp_y)),
        "RMSE_Y": float(np.sqrt(mean_squared_error(yt_y, yp_y))),
        "R2_Y": float(r2_score(yt_y, yp_y)),
        "ExplainedVar_Y": float(explained_variance_score(yt_y, yp_y)),
        "Mean_Err_Y": float(np.mean(err_y)),  # Bias
        "Std_Err_Y": float(np.std(err_y)),  # Spread
        "Median_AbsErr_Y": float(np.median(abs_err_y)),
        "P95_AbsErr_Y": float(np.percentile(abs_err_y, 95)),
        "Max_AbsErr_Y": float(np.max(abs_err_y)),
        "MAPE_Y": float(np.mean(abs_err_y / (np.abs(yt_y) + 1e-8)) * 100),
        "NRMSE_Y": float(np.sqrt(mean_squared_error(yt_y, yp_y)) / range_y) if range_y > 0 else 0.0,

        # === GENEL (KOMBİNE) METRİKLER ===
        "MAE_Combined": float((mean_absolute_error(yt_x, yp_x) + mean_absolute_error(yt_y, yp_y)) / 2),
        "RMSE_Combined": float((np.sqrt(mean_squared_error(yt_x, yp_x)) + np.sqrt(mean_squared_error(yt_y, yp_y))) / 2),
        "R2_Combined": float((r2_score(yt_x, yp_x) + r2_score(yt_y, yp_y)) / 2),
    }

    return out


# -------------------------------
# VISUALIZATION - X ve Y ODAKLI
# -------------------------------
def save_true_vs_pred_plots(y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
                            split_name: str, out_path: str):
    """
    True vs Predicted değerleri karşılaştırmalı scatter plot
    Mükemmel tahmin = 45 derece çizgi üzerinde olmalı
    """
    yt_x, yt_y = y_true[:, 0], y_true[:, 1]
    yp_x, yp_y = y_pred[:, 0], y_pred[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === X Koordinatı ===
    ax = axes[0]

    # Scatter plot
    ax.scatter(yt_x, yp_x, alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidth=0.3)

    # Perfect prediction line (45 degree)
    min_val = min(yt_x.min(), yp_x.min())
    max_val = max(yt_x.max(), yp_x.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Mükemmel Tahmin')

    # Metrics
    mae_x = mean_absolute_error(yt_x, yp_x)
    r2_x = r2_score(yt_x, yp_x)

    ax.set_xlabel('Gerçek X Koordinatı', fontsize=12, weight='bold')
    ax.set_ylabel('Tahmin Edilen X Koordinatı', fontsize=12, weight='bold')
    ax.set_title(f'{model_name} - X Koordinatı ({split_name})\nMAE={mae_x:.2f} | R²={r2_x:.3f}',
                 fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    # === Y Koordinatı ===
    ax = axes[1]

    # Scatter plot
    ax.scatter(yt_y, yp_y, alpha=0.5, s=30, c='forestgreen', edgecolors='black', linewidth=0.3)

    # Perfect prediction line
    min_val = min(yt_y.min(), yp_y.min())
    max_val = max(yt_y.max(), yp_y.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Mükemmel Tahmin')

    # Metrics
    mae_y = mean_absolute_error(yt_y, yp_y)
    r2_y = r2_score(yt_y, yp_y)

    ax.set_xlabel('Gerçek Y Koordinatı', fontsize=12, weight='bold')
    ax.set_ylabel('Tahmin Edilen Y Koordinatı', fontsize=12, weight='bold')
    ax.set_title(f'{model_name} - Y Koordinatı ({split_name})\nMAE={mae_y:.2f} | R²={r2_y:.3f}',
                 fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_error_distribution_per_axis(y_true: np.ndarray, y_pred: np.ndarray,
                                     model_name: str, split_name: str, out_path: str):
    """
    X ve Y hatalarının dağılımı - histogram ile
    """
    err_x = y_pred[:, 0] - y_true[:, 0]
    err_y = y_pred[:, 1] - y_true[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === X Hataları ===
    ax = axes[0]
    n, bins, patches = ax.hist(err_x, bins=50, color='steelblue', alpha=0.7, edgecolors='black')

    # Color bars based on error magnitude
    abs_bins = np.abs(bins[:-1])
    for i, patch in enumerate(patches):
        norm_val = abs_bins[i] / abs_bins.max() if abs_bins.max() > 0 else 0
        patch.set_facecolor(plt.cm.RdYlBu_r(norm_val))

    # Statistics
    ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Sıfır Hata')
    ax.axvline(np.mean(err_x), color='red', linestyle='--', linewidth=2,
               label=f'Ortalama: {np.mean(err_x):.2f}')
    ax.axvline(np.median(err_x), color='green', linestyle='--', linewidth=2,
               label=f'Medyan: {np.median(err_x):.2f}')

    ax.set_xlabel('X Hatası (Tahmin - Gerçek)', fontsize=12, weight='bold')
    ax.set_ylabel('Frekans', fontsize=12, weight='bold')
    ax.set_title(f'{model_name} - X Hata Dağılımı ({split_name})\nStd={np.std(err_x):.2f}',
                 fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    # === Y Hataları ===
    ax = axes[1]
    n, bins, patches = ax.hist(err_y, bins=50, color='forestgreen', alpha=0.7, edgecolors='black')

    # Color bars
    abs_bins = np.abs(bins[:-1])
    for i, patch in enumerate(patches):
        norm_val = abs_bins[i] / abs_bins.max() if abs_bins.max() > 0 else 0
        patch.set_facecolor(plt.cm.RdYlBu_r(norm_val))

    # Statistics
    ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Sıfır Hata')
    ax.axvline(np.mean(err_y), color='red', linestyle='--', linewidth=2,
               label=f'Ortalama: {np.mean(err_y):.2f}')
    ax.axvline(np.median(err_y), color='green', linestyle='--', linewidth=2,
               label=f'Medyan: {np.median(err_y):.2f}')

    ax.set_xlabel('Y Hatası (Tahmin - Gerçek)', fontsize=12, weight='bold')
    ax.set_ylabel('Frekans', fontsize=12, weight='bold')
    ax.set_title(f'{model_name} - Y Hata Dağılımı ({split_name})\nStd={np.std(err_y):.2f}',
                 fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_residual_plots(y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str, split_name: str, out_path: str):
    """
    Residual plots - hataların gerçek değerlere göre dağılımı
    İyi bir model: hatalar rastgele dağılmalı, pattern olmamalı
    """
    yt_x, yt_y = y_true[:, 0], y_true[:, 1]
    err_x = y_pred[:, 0] - y_true[:, 0]
    err_y = y_pred[:, 1] - y_true[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # === X Residuals vs True X ===
    ax = axes[0, 0]
    scatter = ax.scatter(yt_x, err_x, c=np.abs(err_x), cmap='RdYlBu_r',
                         s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Gerçek X Koordinatı', fontsize=11)
    ax.set_ylabel('X Hatası', fontsize=11)
    ax.set_title('X Residual Plot', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='|Hata|')

    # === Y Residuals vs True Y ===
    ax = axes[0, 1]
    scatter = ax.scatter(yt_y, err_y, c=np.abs(err_y), cmap='RdYlBu_r',
                         s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Gerçek Y Koordinatı', fontsize=11)
    ax.set_ylabel('Y Hatası', fontsize=11)
    ax.set_title('Y Residual Plot', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='|Hata|')

    # === X Residuals vs Predicted X ===
    ax = axes[1, 0]
    yp_x = y_pred[:, 0]
    scatter = ax.scatter(yp_x, err_x, c=np.abs(err_x), cmap='RdYlBu_r',
                         s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Tahmin X Koordinatı', fontsize=11)
    ax.set_ylabel('X Hatası', fontsize=11)
    ax.set_title('X Residual vs Predicted', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='|Hata|')

    # === Y Residuals vs Predicted Y ===
    ax = axes[1, 1]
    yp_y = y_pred[:, 1]
    scatter = ax.scatter(yp_y, err_y, c=np.abs(err_y), cmap='RdYlBu_r',
                         s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Tahmin Y Koordinatı', fontsize=11)
    ax.set_ylabel('Y Hatası', fontsize=11)
    ax.set_title('Y Residual vs Predicted', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='|Hata|')

    fig.suptitle(f'{model_name} - Residual Analysis ({split_name})',
                 fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_model_comparison_chart(metrics_df: pd.DataFrame, split_name: str, out_dir: str):
    """
    Tüm modellerin X ve Y metriklerini karşılaştırmalı bar chart
    """
    # Filter for specific split
    df = metrics_df[metrics_df['Split'] == split_name].copy()

    if df.empty:
        return

    # Key metrics to compare
    metrics_to_plot = ['MAE_X', 'MAE_Y', 'RMSE_X', 'RMSE_Y', 'R2_X', 'R2_Y']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Sort by metric value
        df_sorted = df.sort_values(metric, ascending=(metric not in ['R2_X', 'R2_Y']))

        # Create bar plot
        bars = ax.barh(df_sorted['Model'], df_sorted[metric],
                       color=plt.cm.RdYlBu_r(df_sorted[metric] / df_sorted[metric].max()))

        # Color bars - lower is better (except R2)
        if metric in ['R2_X', 'R2_Y']:
            bars = ax.barh(df_sorted['Model'], df_sorted[metric],
                           color=plt.cm.RdYlGn(df_sorted[metric]))

        ax.set_xlabel(metric, fontsize=11, weight='bold')
        ax.set_title(f'{metric} Karşılaştırması', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)

    fig.suptitle(f'Model Performans Karşılaştırması - {split_name}',
                 fontsize=16, weight='bold')
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"model_comparison_{split_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Saved comparison chart: {os.path.basename(out_path)}")


# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_train_cv(models: Dict[str, Any], X: pd.DataFrame, y: np.ndarray,
                      n_splits: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Cross-validation on training data"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    metrics_rows = []
    oof_preds: Dict[str, np.ndarray] = {}

    for name, base_model in models.items():
        log(f"TRAIN CV: {name} ...")
        try:
            preds = np.zeros((len(X), 2), dtype=float)

            for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
                log(f"  fold {fold}/{n_splits}")
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr = y[tr_idx]

                # Clone model for each fold
                fold_model = ensure_multioutput(deepcopy(base_model))
                fold_model.fit(X_tr, y_tr)
                preds[va_idx] = fold_model.predict(X_va)

            oof_preds[name] = preds
            met = compute_xy_focused_metrics(y_true=y, y_pred=preds)
            met["Split"] = f"TRAIN_CV{n_splits}"
            met["Model"] = name
            metrics_rows.append(met)

            log(f"  ✓ X: MAE={met['MAE_X']:.3f}, R²={met['R2_X']:.3f} | "
                f"Y: MAE={met['MAE_Y']:.3f}, R²={met['R2_Y']:.3f}")

        except Exception as e:
            err(f"TRAIN CV failed for {name}: {e}")

    return pd.DataFrame(metrics_rows), oof_preds


def evaluate_test(models: Dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray,
                  X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Final evaluation on test set - AYNI TRAIN DATA ile FIT"""
    metrics_rows = []
    test_preds: Dict[str, np.ndarray] = {}

    for name, base_model in models.items():
        log(f"TEST: {name} ...")
        try:
            # Her model için AYNI train data ile fit - ama BAĞIMSIZ instance
            model = ensure_multioutput(deepcopy(base_model))
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            test_preds[name] = pred

            met = compute_xy_focused_metrics(y_true=y_test, y_pred=pred)
            met["Split"] = "TEST"
            met["Model"] = name
            metrics_rows.append(met)

            log(f"  ✓ X: MAE={met['MAE_X']:.3f}, R²={met['R2_X']:.3f} | "
                f"Y: MAE={met['MAE_Y']:.3f}, R²={met['R2_Y']:.3f}")

        except Exception as e:
            err(f"TEST failed for {name}: {e}")

    return pd.DataFrame(metrics_rows), test_preds


# -------------------------------
# MAIN
# -------------------------------
def main():
    log("=" * 80)
    log("X ve Y KOORDİNAT TAHMİN DOĞRULUĞU ANALİZİ")
    log("ODAK: True X/Y ile Predicted X/Y arasındaki fark (Distance DEĞİL!)")
    log("=" * 80)
    log(f"Output Directory: {os.path.abspath(OUT_DIR)}")
    log(f"Random Seed: {SEED}")
    log(f"CV Folds: {N_SPLITS}")
    log("")

    # Initialize models
    models = get_models(SEED)
    log(f"Models: {list(models.keys())}")
    log("")

    # Load data
    log("Loading training data...")
    X_train, y_train = load_xy_from_separate_files(RSS_TRAIN_PATH, COORD_TRAIN_PATH)
    log("")

    log("Loading test data...")
    X_test, y_test = load_xy_from_separate_files(RSS_TEST_PATH, COORD_TEST_PATH)
    log("")

    # TRAIN CV
    log("=" * 80)
    log("PHASE 1: Cross-Validation")
    log("=" * 80)
    train_metrics_df, train_oof = evaluate_train_cv(models, X_train, y_train, N_SPLITS, SEED)
    log("")

    # TEST
    log("=" * 80)
    log("PHASE 2: Test Evaluation (Tüm modeller AYNI train data ile fit ediliyor)")
    log("=" * 80)
    test_metrics_df, test_preds = evaluate_test(models, X_train, y_train, X_test, y_test)
    log("")

    if train_metrics_df.empty and test_metrics_df.empty:
        err("No metrics produced.")
        return

    # Combine metrics
    all_metrics = pd.concat([train_metrics_df, test_metrics_df], axis=0, ignore_index=True)

    # Round for readability
    round_cols = [c for c in all_metrics.columns if c not in ["Model", "Split"]]
    all_metrics[round_cols] = all_metrics[round_cols].round(4)

    # Reorder columns - X ve Y metrikleri önce
    col_order = [
        "Split", "Model",
        # X Koordinatı
        "MAE_X", "RMSE_X", "R2_X", "ExplainedVar_X",
        "Mean_Err_X", "Std_Err_X", "Median_AbsErr_X", "P95_AbsErr_X", "Max_AbsErr_X",
        "MAPE_X", "NRMSE_X",
        # Y Koordinatı
        "MAE_Y", "RMSE_Y", "R2_Y", "ExplainedVar_Y",
        "Mean_Err_Y", "Std_Err_Y", "Median_AbsErr_Y", "P95_AbsErr_Y", "Max_AbsErr_Y",
        "MAPE_Y", "NRMSE_Y",
        # Kombine
        "MAE_Combined", "RMSE_Combined", "R2_Combined"
    ]
    all_metrics = all_metrics[[c for c in col_order if c in all_metrics.columns]]
    all_metrics = all_metrics.sort_values(["Split", "MAE_Combined"])

    # Save results
    log("=" * 80)
    log("Saving Results")
    log("=" * 80)

    out_csv = os.path.join(OUT_DIR, "XY_Coordinate_Accuracy_Metrics.csv")
    out_xlsx = os.path.join(OUT_DIR, "XY_Coordinate_Accuracy_Metrics.xlsx")

    all_metrics.to_csv(out_csv, index=False)
    log(f"✓ CSV:  {os.path.basename(out_csv)}")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        all_metrics.to_excel(writer, sheet_name="All_Metrics", index=False)
        train_metrics_df.sort_values("MAE_Combined").to_excel(writer, sheet_name="Train_CV", index=False)
        test_metrics_df.sort_values("MAE_Combined").to_excel(writer, sheet_name="Test", index=False)
    log(f"✓ XLSX: {os.path.basename(out_xlsx)}")
    log("")

    # Print summary
    log("=" * 80)
    log("RESULTS SUMMARY - En İyi Performans (MAE_Combined'a göre)")
    log("=" * 80)

    # Show top models for each split
    for split in all_metrics['Split'].unique():
        print(f"\n{split}:")
        split_df = all_metrics[all_metrics['Split'] == split].sort_values('MAE_Combined')
        print(split_df[['Model', 'MAE_X', 'MAE_Y', 'R2_X', 'R2_Y', 'MAE_Combined']].head(3).to_string(index=False))
    log("")

    # Visualizations
    if MAKE_PLOTS:
        log("=" * 80)
        log("Generating Visualizations")
        log("=" * 80)

        # 1. True vs Predicted plots (her model için)
        log("Creating True vs Predicted plots...")
        for name, pred in train_oof.items():
            out_path = os.path.join(OUT_DIR, f"true_vs_pred_train_{name}.png")
            save_true_vs_pred_plots(y_train, pred, name, "TRAIN CV", out_path)

        for name, pred in test_preds.items():
            out_path = os.path.join(OUT_DIR, f"true_vs_pred_test_{name}.png")
            save_true_vs_pred_plots(y_test, pred, name, "TEST", out_path)

        # 2. Error distribution plots
        log("Creating error distribution plots...")
        for name, pred in train_oof.items():
            out_path = os.path.join(OUT_DIR, f"error_dist_train_{name}.png")
            save_error_distribution_per_axis(y_train, pred, name, "TRAIN CV", out_path)

        for name, pred in test_preds.items():
            out_path = os.path.join(OUT_DIR, f"error_dist_test_{name}.png")
            save_error_distribution_per_axis(y_test, pred, name, "TEST", out_path)

        # 3. Residual plots
        log("Creating residual plots...")
        for name, pred in test_preds.items():
            out_path = os.path.join(OUT_DIR, f"residuals_test_{name}.png")
            save_residual_plots(y_test, pred, name, "TEST", out_path)

        # 4. Model comparison charts
        log("Creating model comparison charts...")
        save_model_comparison_chart(all_metrics, f"TRAIN_CV{N_SPLITS}", OUT_DIR)
        save_model_comparison_chart(all_metrics, "TEST", OUT_DIR)

        log("")
        log("✓ All visualizations completed!")

    log("")
    log("=" * 80)
    log("ANALİZ TAMAMLANDI!")
    log("=" * 80)
    log(f"Tüm sonuçlar: {os.path.abspath(OUT_DIR)}")
    log("")
    log("ÖNEMLİ: Bu kod X ve Y koordinatlarının tahmin doğruluğuna odaklanır.")
    log("Distance metrikleri dahil DEĞİLDİR - sadece X ve Y'nin gerçek değerlere yakınlığı.")
    log("")


if __name__ == "__main__":
    main()