# tuji_cv_common_joint.py
# ============================================================
# TUJI1 - CV5 common utilities (TRAIN only) - JOINT (X,Y) MODEL
# - Reads TUJI1 training files
# - Leakage-free pipeline: imputer + scaler inside fold
# - Missing RSSI: 100 -> NaN -> constant (-110 dBm)
# - Optionally drops all-missing AP columns (based on TRAIN only)
# - Fits ONE multi-output regressor to predict [x, y] together
# - Returns OOF predictions for X and Y + fold metrics
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# XGBoost optional
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Coordinates column indices (as you described)
COL_X = 0
COL_Y = 1
COL_Z = 2
COL_FLOOR = 3
COL_BUILDING = 4
COL_DEVICE = 5


@dataclass
class CVConfig:
    data_dir: str = "Data"
    rss_train: str = "RSS_training.csv"
    coord_train: str = "Coordinates_training.csv"
    out_dir: str = "results_cv5"
    n_splits: int = 5
    seed: int = 42

    missing_value: float = 100.0
    impute_strategy: str = "constant"
    impute_fill_value: float = -110.0
    scale: bool = True
    drop_all_missing_cols: bool = True

    verbose: bool = True
    fold_verbose: bool = False
    print_device_counts: bool = True
    print_floor_counts: bool = True
    print_building_counts: bool = True


def load_tuji1_training(cfg: CVConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rss_path = os.path.join(cfg.data_dir, cfg.rss_train)
    coord_path = os.path.join(cfg.data_dir, cfg.coord_train)

    if not os.path.isfile(rss_path):
        raise FileNotFoundError(f"RSS file not found: {rss_path}")
    if not os.path.isfile(coord_path):
        raise FileNotFoundError(f"Coord file not found: {coord_path}")

    X = pd.read_csv(rss_path)
    Y = pd.read_csv(coord_path)

    if len(X) != len(Y):
        raise ValueError(f"Row mismatch: RSS rows={len(X)} vs Coord rows={len(Y)}")

    if Y.shape[1] < 6:
        raise ValueError("Coordinates_training.csv must have 6 columns: x,y,z,floor,building,device")

    if cfg.verbose:
        print(f"[INFO] TRAIN RSSI: {X.shape} | TRAIN COORD: {Y.shape}")

        if cfg.print_building_counts:
            b = Y.iloc[:, COL_BUILDING].unique()
            print(f"[INFO] TRAIN unique buildings: {len(b)} | {sorted(b.tolist())[:20]}")

        if cfg.print_floor_counts:
            f = Y.iloc[:, COL_FLOOR].unique()
            print(f"[INFO] TRAIN unique floors: {len(f)} | {sorted(f.tolist())[:20]}")

        if cfg.print_device_counts:
            d = Y.iloc[:, COL_DEVICE].unique()
            print(f"[INFO] TRAIN unique devices: {len(d)} | {sorted(d.tolist())[:20]}")

    return X, Y

def run_fulltrain_test_joint(cfg: CVConfig, X_train: pd.DataFrame, Y_train: pd.DataFrame,
                             X_test: pd.DataFrame, Y_test: pd.DataFrame) -> Dict[str, Dict]:
    """
    Fits each model on FULL TRAIN and predicts on TEST.
    Returns a dict similar in spirit to cv_out but contains TEST predictions.
    """

    # 100 -> NaN
    X_tr_df = X_train.replace(cfg.missing_value, np.nan)
    X_te_df = X_test.replace(cfg.missing_value, np.nan)

    # TRAIN bazlı all-missing kolonları düş (testte de aynı kolonları düş)
    dropped_cols = []
    if cfg.drop_all_missing_cols:
        dropped_cols = X_tr_df.columns[X_tr_df.isna().all(axis=0)].tolist()
        if dropped_cols:
            print(f"[INFO] Dropping all-missing AP columns (TRAIN-based): {len(dropped_cols)}")
            X_tr_df = X_tr_df.drop(columns=dropped_cols)
            X_te_df = X_te_df.drop(columns=dropped_cols)

    X_tr = X_tr_df.to_numpy(dtype=np.float32)
    X_te = X_te_df.to_numpy(dtype=np.float32)

    tx_tr = Y_train.iloc[:, COL_X].to_numpy(dtype=np.float32)
    ty_tr = Y_train.iloc[:, COL_Y].to_numpy(dtype=np.float32)
    Y_tr = np.column_stack([tx_tr, ty_tr]).astype(np.float32)

    tx_te = Y_test.iloc[:, COL_X].to_numpy(dtype=np.float32)
    ty_te = Y_test.iloc[:, COL_Y].to_numpy(dtype=np.float32)

    models = get_models(cfg.seed)

    out: Dict[str, Dict] = {}
    for model_name, base_model in models.items():
        print(f"[MODEL] {model_name} | fit FULL TRAIN -> predict TEST")

        pipe = build_pipeline(cfg, base_model)
        pipe.fit(X_tr, Y_tr)
        pred = pipe.predict(X_te).astype(np.float32)

        out[model_name] = {
            "true_x": tx_te,
            "true_y": ty_te,
            "pred_x": pred[:, 0],
            "pred_y": pred[:, 1],
            "dropped_cols": dropped_cols
        }

    return out



def load_tuji1_testing(cfg: CVConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rss_path = os.path.join(cfg.data_dir, "RSS_testing.csv")
    coord_path = os.path.join(cfg.data_dir, "Coordinates_testing.csv")

    if not os.path.isfile(rss_path):
        raise FileNotFoundError(f"RSS testing file not found: {rss_path}")
    if not os.path.isfile(coord_path):
        raise FileNotFoundError(f"Coordinates testing file not found: {coord_path}")

    X = pd.read_csv(rss_path)
    Y = pd.read_csv(coord_path)

    if len(X) != len(Y):
        raise ValueError(f"Row mismatch: RSS_TEST rows={len(X)} vs Coord_TEST rows={len(Y)}")

    return X, Y



def euclidean_2d(x_true, y_true, x_pred, y_pred) -> np.ndarray:
    return np.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)


def reg_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def get_models(seed: int) -> Dict[str, Any]:
    """
    Returns base estimators.
    We'll wrap single-output estimators with MultiOutputRegressor if needed.
    """
    models: Dict[str, Any] = {
        "BayesianRidge": BayesianRidge(),  # single-output -> wrapper
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1
        ),  # native multi-output
        "KNN": KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
            metric="euclidean"
        ),  # native multi-output
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
        ),  # native multi-output
        "ElasticNet": MultiTaskElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=seed
        )  # true joint linear model for multiple targets
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1
        )  # single-output -> wrapper

    return models


def needs_multioutput_wrapper(estimator) -> bool:
    """
    Wrap estimators that don't support multioutput natively.
    """
    # quick heuristic: some sklearn estimators expose multioutput through n_outputs_ after fit,
    # but before fit we decide based on known types.
    return isinstance(estimator, (BayesianRidge,)) or (HAS_XGB and estimator.__class__.__name__ == "XGBRegressor")


def build_pipeline(cfg: CVConfig, base_model) -> Pipeline:
    steps = []

    if cfg.impute_strategy == "constant":
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=cfg.impute_fill_value)))
    else:
        steps.append(("imputer", SimpleImputer(strategy=cfg.impute_strategy)))

    if cfg.scale:
        steps.append(("scaler", StandardScaler()))

    # Wrap if needed
    model = MultiOutputRegressor(base_model) if needs_multioutput_wrapper(base_model) else base_model
    steps.append(("model", model))

    return Pipeline(steps)


def model_params_for_log(models: Dict[str, Any]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, m in models.items():
        out[name] = m.get_params() if hasattr(m, "get_params") else {}
    return out


def run_cv5_oof_joint(cfg: CVConfig, X_df: pd.DataFrame, Y_df: pd.DataFrame) -> Dict[str, dict]:
    """
    Joint CV runner: one model predicts [x, y] together.

    Returns dict keyed by model name:
      {
        'oof_pred_x': np.ndarray,
        'oof_pred_y': np.ndarray,
        'true_x': np.ndarray,
        'true_y': np.ndarray,
        'fold_metrics': list
      }
    """
    if cfg.verbose:
        print(f"[INFO] CV: KFold(n_splits={cfg.n_splits}, shuffle=True, seed={cfg.seed})")

    models = get_models(cfg.seed)
    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    # missing code -> NaN
    X_all_df = X_df.copy().replace(cfg.missing_value, np.nan)

    dropped_cols = []
    if cfg.drop_all_missing_cols:
        all_nan_mask = X_all_df.isna().all(axis=0)
        dropped_cols = X_all_df.columns[all_nan_mask].tolist()
        if cfg.verbose:
            print(f"[INFO] All-missing AP columns (TRAIN): {len(dropped_cols)}")
        if dropped_cols:
            X_all_df = X_all_df.drop(columns=dropped_cols)

    X_all = X_all_df.to_numpy(dtype=np.float32)

    true_x = Y_df.iloc[:, COL_X].to_numpy(dtype=np.float32)
    true_y = Y_df.iloc[:, COL_Y].to_numpy(dtype=np.float32)
    Y_all = np.column_stack([true_x, true_y]).astype(np.float32)

    results: Dict[str, dict] = {}

    for model_name, base_model in models.items():
        if cfg.verbose:
            wrap_flag = needs_multioutput_wrapper(base_model)
            print(f"\n[MODEL] {model_name} | joint [x,y] | wrapper={wrap_flag}")

        oof_pred = np.zeros((len(Y_all), 2), dtype=np.float32)
        fold_logs: List[dict] = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all), start=1):
            X_tr, X_va = X_all[tr_idx], X_all[va_idx]
            Y_tr, Y_va = Y_all[tr_idx], Y_all[va_idx]

            pipe = build_pipeline(cfg, base_model)
            pipe.fit(X_tr, Y_tr)
            pred_va = pipe.predict(X_va).astype(np.float32)

            oof_pred[va_idx, :] = pred_va

            tx_va, ty_va = Y_va[:, 0], Y_va[:, 1]
            px_va, py_va = pred_va[:, 0], pred_va[:, 1]

            dist = euclidean_2d(tx_va, ty_va, px_va, py_va)
            mx = reg_metrics(tx_va, px_va)
            my = reg_metrics(ty_va, py_va)

            fm = {
                "fold": fold,
                "dist_mean": float(np.mean(dist)),
                "dist_min": float(np.min(dist)),
                "dist_max": float(np.max(dist)),
                "dist_std": float(np.std(dist)),
                "x": mx,
                "y": my,
            }
            fold_logs.append(fm)

            if cfg.fold_verbose:
                print(f"   [FOLD {fold}] meanDist={fm['dist_mean']:.3f} | R2x={mx['R2']:.3f} | R2y={my['R2']:.3f}")

        results[model_name] = {
            "oof_pred_x": oof_pred[:, 0],
            "oof_pred_y": oof_pred[:, 1],
            "true_x": true_x,
            "true_y": true_y,
            "fold_metrics": fold_logs,
            "dropped_cols": dropped_cols
        }

    # logs
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "cv_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    with open(os.path.join(cfg.out_dir, "hyperparams.json"), "w", encoding="utf-8") as f:
        json.dump(model_params_for_log(get_models(cfg.seed)), f, indent=2)

    preprocess_report = {
        "missing_value_code": cfg.missing_value,
        "impute_strategy": cfg.impute_strategy,
        "impute_fill_value": cfg.impute_fill_value if cfg.impute_strategy == "constant" else None,
        "scale": cfg.scale,
        "drop_all_missing_cols": cfg.drop_all_missing_cols,
        "n_features_original": int(X_df.shape[1]),
        "n_features_after_drop": int(X_all_df.shape[1]),
        "n_dropped_all_missing_cols": int(len(dropped_cols)),
        "dropped_cols_example": dropped_cols[:50],
    }
    with open(os.path.join(cfg.out_dir, "preprocess_report.json"), "w", encoding="utf-8") as f:
        json.dump(preprocess_report, f, indent=2)

    return results
