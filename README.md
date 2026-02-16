Performance Evaluation of Wi-Fi Indoor Positioning by Regression-Based Coordinate Estimation in High-Density and Multi-Device Environments

This repository contains the experimental implementation for evaluating Wi-Fi RSSI-based indoor positioning under device heterogeneity (antenna variability).

The study investigates how device-specific antenna characteristics affect coordinate estimation accuracy using multiple regression-based models.



The main focus is:

Joint (X,Y) regression

Device-specific vs device-agnostic modeling

Cross-device generalization

Transfer learning

Normalization strategies

Ensemble modeling

Feature engineering effects

Coordinate-level accuracy analysis (not only distance)



📂 Repository Structure
File	Description
Table1.py	5-Fold Cross-Validation (Joint X,Y regression)
Table2.py	Full Train → Test evaluation
Table3_4.py	Coordinate-focused accuracy analysis (MAE_X, R²_X, etc.)
Table5_6.py	Multi-scenario device heterogeneity analysis
tuji_cv_common_joint.py	Shared CV configuration and joint regression utilities


📊 Implemented Scenarios
Scenario A – Device-Specific

Separate model trained and tested per device.

Scenario B – Device-Agnostic

Single pooled model trained on all devices.

Scenario C – Cross-Device

Train on one device, test on another.

Scenario D – Transfer Learning

Fine-tuning with limited target-device samples.

Scenario E – RSSI Normalization

Device-wise RSSI standardization.

Scenario F – Ensemble

Weighted combination of device-specific models.

Scenario G – Feature Engineering

RSSI pairwise differences, ranking, normalization patterns, statistical summaries.

📈 Evaluation Metrics

Distance-based:

Mean positioning error

Median error

RMSE

Percentile errors (P50, P75, P90, P95)

Coordinate-based:

MAE_X / MAE_Y

RMSE_X / RMSE_Y

R²_X / R²_Y

Bias and variance per axis

🧠 Models Used

XGBoost

Random Forest

MLP Regressor

KNN

Bayesian Ridge

ElasticNet

⚙️ Data Handling

RSSI value 100 (no signal) is replaced with -110 dBm

Train and test splits remain strictly separated

No data leakage across folds

All reported CV results use proper out-of-fold predictions

Reproducibility

Random seed = 42

5-Fold Cross-Validation

Fixed hyperparameters (no aggressive tuning)
