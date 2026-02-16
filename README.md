## Performance Evaluation of Wi-Fi Indoor Positioning by Regression-Based Coordinate Estimation in High-Density and Multi-Device Environments

---

## Overview

This repository contains the experimental implementation of a regression-based Wi-Fi RSSI indoor positioning framework designed to analyze device heterogeneity in multi-device environments.

The study investigates how device-specific antenna characteristics influence coordinate estimation accuracy using multiple regression-based models.

The main methodological focus includes:

- Joint (X, Y) regression
- Device-specific vs device-agnostic modeling
- Cross-device generalization
- Transfer learning
- RSSI normalization strategies
- Ensemble modeling
- Feature engineering effects
- Coordinate-level accuracy analysis (beyond distance-only metrics)

---

## Repository Structure

| File | Description |
|------|------------|
| `Table1.py` | 5-Fold Cross-Validation (Joint X,Y regression) |
| `Table2.py` | Full Train → Test evaluation |
| `Table3_4.py` | Coordinate-focused accuracy analysis (MAE_X, RMSE_X, R²_X, etc.) |
| `Table5_6.py` | Multi-scenario device heterogeneity analysis |
| `tuji_cv_common_joint.py` | Shared CV configuration and joint regression utilities |

---

## Implemented Scenarios

### Scenario A – Device-Specific
Separate model trained and tested per device.

### Scenario B – Device-Agnostic
Single pooled model trained on all devices.

### Scenario C – Cross-Device
Train on one device and test on another.

### Scenario D – Transfer Learning
Fine-tuning with limited target-device samples.

### Scenario E – RSSI Normalization
Device-wise RSSI standardization.

### Scenario F – Ensemble
Weighted combination of device-specific models.

### Scenario G – Feature Engineering
Includes:
- RSSI pairwise differences  
- Ranking-based transformations  
- Normalized RSSI representations  
- Statistical feature summaries  

---

## Evaluation Metrics

### Distance-Based Metrics

- Mean positioning error  
- Median error  
- Root Mean Square Error (RMSE)  
- Percentile errors (P50, P75, P90, P95)

### Coordinate-Based Metrics

- MAE_X / MAE_Y  
- RMSE_X / RMSE_Y  
- R²_X / R²_Y  
- Axis-level bias (mean error per axis)  
- Axis-level variance  

---

## Models Used

- XGBoost  
- Random Forest  
- Multi-Layer Perceptron (MLP Regressor)  
- K-Nearest Neighbors (KNN)  
- Bayesian Ridge  
- MultiTask ElasticNet  

All models use fixed hyperparameters to ensure reproducibility and comparability.

---

## Data Handling

- RSSI value `100` (no signal) is replaced with `-110 dBm`.
- Train and test splits remain strictly separated.
- No data leakage across folds.
- All cross-validation results use proper out-of-fold predictions.

---

## Reproducibility

- Random seed: `42`
- 5-Fold Cross-Validation
- Fixed hyperparameter configuration
- Deterministic evaluation pipeline
