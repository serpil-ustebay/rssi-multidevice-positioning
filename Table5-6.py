"""
IndoorPositioningAnalyzer – Device Heterogeneity (Antenna Effect) Analysis

This script analyzes the impact of device/antenna heterogeneity on Wi-Fi RSSI-based
indoor positioning by comparing multiple training/evaluation scenarios.

Core approach:
- Models estimate (X, Y) coordinates from RSSI fingerprints.
- Separate regressors are trained for X and Y (two-model approach).
- Performance is reported using Euclidean positioning error on the test set
  (mean/median/std, RMSE, and tail percentiles such as P90/P95).

Data handling:
- The dataset uses the placeholder value '100' to indicate "no signal received".
- All occurrences of 100 are replaced with a conservative shadow value (-110 dBm).
- Train and test splits are kept strictly separated.

Scenarios implemented:
A) Device-Specific:
   Train and test using data from the same device ID (one model per device).

B) Device-Agnostic:
   Train a single model on pooled training data (all devices), then evaluate per device.

C) Cross-Device Generalization:
   Train on one device and test on a different device to quantify domain shift.

D) Transfer Learning (Data-Limited Adaptation):
   Train on pooled data excluding the target device, plus a small fraction of target-device
   samples (e.g., 20%), then evaluate on the target-device test set.

E) RSSI Normalization (Device-wise Standardization):
   Apply per-device standardization (fit on device-specific training data, apply to that
   device’s test data) and evaluate a pooled model.

F) Ensemble:
   Combine predictions from device-specific models using weighted averaging
   (higher weight for the target device’s own model).

G) Feature Engineering:
   Create additional RSSI-derived features (pairwise differences, rank-based features,
   normalized RSSI patterns, mean/std summaries) and evaluate pooled models.

Outputs:
- Per-device metric summaries for each scenario
- CDF plots of positioning errors
- Scenario comparison plots and tables
- Statistical tests (paired t-test, ANOVA) for scenario-level comparisons
- Excel export of aggregated results

Reproducibility & privacy:
- Avoid uploading restricted datasets if redistribution is not permitted.
- This repository should not contain personal identifiers.

Run:
- Place the required CSV files under the Data/ directory.
- Execute: python your_script_name.py
- Outputs are written under the configured results folder.
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
from xgboost import XGBRegressor

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class IndoorPositioningAnalyzer:

    def __init__(self, rssi_train_path, coor_train_path, rssi_test_path, coor_test_path):
        self.load_data(rssi_train_path, coor_train_path, rssi_test_path, coor_test_path)
        self.preprocess_data()

        self.results = {
            'scenario_a': {},
            'scenario_b': {},
            'scenario_c': {},
            'scenario_d': {},
            'scenario_e': {},
            'scenario_f': {},
            'scenario_g': {}
        }

    def load_data(self, rssi_train_path, coor_train_path, rssi_test_path, coor_test_path):
        print("Veriler yükleniyor...")
        self.rssi_train = pd.read_csv(rssi_train_path, header=None)
        self.coor_train = pd.read_csv(coor_train_path, header=None)
        self.rssi_test = pd.read_csv(rssi_test_path, header=None)
        self.coor_test = pd.read_csv(coor_test_path, header=None)

        self.coor_train.columns = ['X', 'Y', 'Z', 'buildingID', 'floor', 'DeviceID']
        self.coor_test.columns = ['X', 'Y', 'Z', 'buildingID', 'floor', 'DeviceID']

        print(f"Train RSSI shape: {self.rssi_train.shape}")
        print(f"Test RSSI shape: {self.rssi_test.shape}")

    def preprocess_data(self):
        print("\nVeri ön işleme yapılıyor...")
        self.rssi_train = self.rssi_train.replace(100, -110)
        self.rssi_test = self.rssi_test.replace(100, -110)
        print("100 dBm → -110 dBm")

        self.train_data = pd.concat([self.rssi_train, self.coor_train], axis=1)
        self.test_data = pd.concat([self.rssi_test, self.coor_test], axis=1)

        print(f"\nCihaz dağılımı (Train):\n{self.train_data['DeviceID'].value_counts()}")
        print(f"\nCihaz dağılımı (Test):\n{self.test_data['DeviceID'].value_counts()}")

    def train_model(self, X_train, y_train):
        model =XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def calculate_metrics(self, y_true, y_pred):
        metrics = {}
        euclidean_errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
        metrics['mean_error'] = np.mean(euclidean_errors)
        metrics['median_error'] = np.median(euclidean_errors)
        metrics['std_error'] = np.std(euclidean_errors)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['error_50'] = np.percentile(euclidean_errors, 50)
        metrics['error_75'] = np.percentile(euclidean_errors, 75)
        metrics['error_90'] = np.percentile(euclidean_errors, 90)
        metrics['error_95'] = np.percentile(euclidean_errors, 95)
        metrics['euclidean_errors'] = euclidean_errors
        return metrics

    def scenario_a_device_specific(self):
        print("\n" + "="*80)
        print("SENARYO A: Device-Specific Models")
        print("="*80)

        device_ids = sorted(self.train_data['DeviceID'].unique())
        rssi_cols = list(range(len(self.rssi_train.columns)))

        for device_id in device_ids:
            print(f"\nCihaz {device_id} eğitiliyor...")

            train_device = self.train_data[self.train_data['DeviceID'] == device_id]
            test_device = self.test_data[self.test_data['DeviceID'] == device_id]

            if len(test_device) == 0:
                continue

            X_train = train_device[rssi_cols].values
            y_train_xy = train_device[['X', 'Y']].values
            X_test = test_device[rssi_cols].values
            y_test_xy = test_device[['X', 'Y']].values

            model_x = self.train_model(X_train, y_train_xy[:, 0])
            model_y = self.train_model(X_train, y_train_xy[:, 1])

            y_pred_xy = np.column_stack([model_x.predict(X_test), model_y.predict(X_test)])

            metrics = self.calculate_metrics(y_test_xy, y_pred_xy)
            self.results['scenario_a'][device_id] = metrics

            print(f"  Mean: {metrics['mean_error']:.2f}m | Median: {metrics['median_error']:.2f}m | 90%: {metrics['error_90']:.2f}m")

    def scenario_b_device_agnostic(self):
        print("\n" + "="*80)
        print("SENARYO B: Device-Agnostic Model")
        print("="*80)

        rssi_cols = list(range(len(self.rssi_train.columns)))

        X_train_all = self.train_data[rssi_cols].values
        y_train_xy_all = self.train_data[['X', 'Y']].values

        print("Tüm verilerle tek model eğitiliyor...")
        model_x_all = self.train_model(X_train_all, y_train_xy_all[:, 0])
        model_y_all = self.train_model(X_train_all, y_train_xy_all[:, 1])

        for device_id in sorted(self.test_data['DeviceID'].unique()):
            print(f"\nCihaz {device_id} test ediliyor...")

            test_device = self.test_data[self.test_data['DeviceID'] == device_id]
            if len(test_device) == 0:
                continue

            X_test = test_device[rssi_cols].values
            y_test_xy = test_device[['X', 'Y']].values

            y_pred_xy = np.column_stack([model_x_all.predict(X_test), model_y_all.predict(X_test)])

            metrics = self.calculate_metrics(y_test_xy, y_pred_xy)
            self.results['scenario_b'][device_id] = metrics

            print(f"  Mean: {metrics['mean_error']:.2f}m | Median: {metrics['median_error']:.2f}m | 90%: {metrics['error_90']:.2f}m")

    def scenario_c_cross_device(self):
        print("\n" + "="*80)
        print("SENARYO C: Cross-Device Generalization")
        print("="*80)

        device_ids = sorted(self.train_data['DeviceID'].unique())
        rssi_cols = list(range(len(self.rssi_train.columns)))
        cross_device_results = {}

        for train_device_id in device_ids:
            train_device = self.train_data[self.train_data['DeviceID'] == train_device_id]
            X_train = train_device[rssi_cols].values
            y_train_xy = train_device[['X', 'Y']].values

            model_x = self.train_model(X_train, y_train_xy[:, 0])
            model_y = self.train_model(X_train, y_train_xy[:, 1])

            cross_device_results[train_device_id] = {}

            for test_device_id in device_ids:
                if test_device_id == train_device_id:
                    continue

                test_device = self.test_data[self.test_data['DeviceID'] == test_device_id]
                if len(test_device) == 0:
                    continue

                X_test = test_device[rssi_cols].values
                y_test_xy = test_device[['X', 'Y']].values

                y_pred_xy = np.column_stack([model_x.predict(X_test), model_y.predict(X_test)])
                metrics = self.calculate_metrics(y_test_xy, y_pred_xy)
                cross_device_results[train_device_id][test_device_id] = metrics

                print(f"Train:{train_device_id} Test:{test_device_id} -> {metrics['mean_error']:.2f}m")

        self.results['scenario_c'] = cross_device_results

    def scenario_d_transfer_learning(self, ratio=0.2):
        print("\n" + "="*80)
        print(f"SENARYO D: Transfer Learning ({ratio*100:.0f}% fine-tuning)")
        print("="*80)

        rssi_cols = list(range(len(self.rssi_train.columns)))

        for device_id in sorted(self.train_data['DeviceID'].unique()):
            train_device = self.train_data[self.train_data['DeviceID'] == device_id]
            test_device = self.test_data[self.test_data['DeviceID'] == device_id]

            if len(test_device) == 0:
                continue

            size = max(10, int(len(train_device) * ratio))
            sample = train_device.sample(n=size, random_state=42)
            others = self.train_data[self.train_data['DeviceID'] != device_id]
            combined = pd.concat([others, sample])

            X_train = combined[rssi_cols].values
            y_train_xy = combined[['X', 'Y']].values

            model_x = self.train_model(X_train, y_train_xy[:, 0])
            model_y = self.train_model(X_train, y_train_xy[:, 1])

            X_test = test_device[rssi_cols].values
            y_test_xy = test_device[['X', 'Y']].values

            y_pred_xy = np.column_stack([model_x.predict(X_test), model_y.predict(X_test)])

            metrics = self.calculate_metrics(y_test_xy, y_pred_xy)
            self.results['scenario_d'][device_id] = metrics

            print(f"Cihaz {device_id}: {metrics['mean_error']:.2f}m")

    def scenario_e_normalization(self):
        print("\n" + "="*80)
        print("SENARYO E: RSSI Normalization")
        print("="*80)

        rssi_cols = list(range(len(self.rssi_train.columns)))

        train_norm = self.train_data.copy()
        test_norm = self.test_data.copy()

        for device_id in self.train_data['DeviceID'].unique():
            scaler = StandardScaler()

            train_mask = train_norm['DeviceID'] == device_id
            train_norm.loc[train_mask, rssi_cols] = scaler.fit_transform(
                train_norm.loc[train_mask, rssi_cols]
            )

            test_mask = test_norm['DeviceID'] == device_id
            if test_mask.sum() > 0:
                test_norm.loc[test_mask, rssi_cols] = scaler.transform(
                    test_norm.loc[test_mask, rssi_cols]
                )

        X_train = train_norm[rssi_cols].values
        y_train_xy = train_norm[['X', 'Y']].values

        model_x = self.train_model(X_train, y_train_xy[:, 0])
        model_y = self.train_model(X_train, y_train_xy[:, 1])

        for device_id in sorted(test_norm['DeviceID'].unique()):
            test_device = test_norm[test_norm['DeviceID'] == device_id]
            if len(test_device) == 0:
                continue

            X_test = test_device[rssi_cols].values
            y_test_xy = test_device[['X', 'Y']].values

            y_pred_xy = np.column_stack([model_x.predict(X_test), model_y.predict(X_test)])

            metrics = self.calculate_metrics(y_test_xy, y_pred_xy)
            self.results['scenario_e'][device_id] = metrics

            print(f"Cihaz {device_id}: {metrics['mean_error']:.2f}m")

    def scenario_f_ensemble(self):
        print("\n" + "="*80)
        print("SENARYO F: Ensemble")
        print("="*80)

        rssi_cols = list(range(len(self.rssi_train.columns)))
        device_ids = sorted(self.train_data['DeviceID'].unique())

        models_x = {}
        models_y = {}

        for device_id in device_ids:
            train_device = self.train_data[self.train_data['DeviceID'] == device_id]
            X_train = train_device[rssi_cols].values
            y_train_xy = train_device[['X', 'Y']].values

            models_x[device_id] = self.train_model(X_train, y_train_xy[:, 0])
            models_y[device_id] = self.train_model(X_train, y_train_xy[:, 1])

        for test_device_id in device_ids:
            test_device = self.test_data[self.test_data['DeviceID'] == test_device_id]
            if len(test_device) == 0:
                continue

            X_test = test_device[rssi_cols].values
            y_test_xy = test_device[['X', 'Y']].values

            own_w = 0.7
            other_w = 0.3 / (len(device_ids) - 1) if len(device_ids) > 1 else 0

            pred_x = np.zeros(len(X_test))
            pred_y = np.zeros(len(X_test))

            for device_id in device_ids:
                w = own_w if device_id == test_device_id else other_w
                pred_x += models_x[device_id].predict(X_test) * w
                pred_y += models_y[device_id].predict(X_test) * w

            y_pred_xy = np.column_stack([pred_x, pred_y])

            metrics = self.calculate_metrics(y_test_xy, y_pred_xy)
            self.results['scenario_f'][test_device_id] = metrics

            print(f"Cihaz {test_device_id}: {metrics['mean_error']:.2f}m")

    def scenario_g_feature_engineering(self):
        print("\n" + "="*80)
        print("SENARYO G: Feature Engineering")
        print("="*80)

        rssi_cols = list(range(len(self.rssi_train.columns)))

        def create_features(df):
            features = []
            rssi = df[rssi_cols].values

            for i in range(min(10, len(rssi_cols)-1)):
                for j in range(i+1, min(10, len(rssi_cols))):
                    features.append(rssi[:, i] - rssi[:, j])

            features.append(np.argsort(np.argsort(rssi, axis=1), axis=1))

            rssi_min = rssi.min(axis=1, keepdims=True)
            rssi_max = rssi.max(axis=1, keepdims=True)
            rssi_range = rssi_max - rssi_min
            rssi_range[rssi_range == 0] = 1
            features.append((rssi - rssi_min) / rssi_range)

            features.append(rssi.mean(axis=1, keepdims=True))
            features.append(rssi.std(axis=1, keepdims=True))

            return np.concatenate([f.reshape(len(df), -1) for f in features], axis=1)

        X_train = create_features(self.train_data)
        X_test = create_features(self.test_data)

        y_train_xy = self.train_data[['X', 'Y']].values

        print(f"Özellik sayısı: {X_train.shape[1]}")

        model_x = self.train_model(X_train, y_train_xy[:, 0])
        model_y = self.train_model(X_train, y_train_xy[:, 1])

        for device_id in sorted(self.test_data['DeviceID'].unique()):
            mask = self.test_data['DeviceID'] == device_id
            test_device = self.test_data[mask]

            if len(test_device) == 0:
                continue

            X_test_dev = X_test[mask]
            y_test_xy = test_device[['X', 'Y']].values

            y_pred_xy = np.column_stack([model_x.predict(X_test_dev), model_y.predict(X_test_dev)])

            metrics = self.calculate_metrics(y_test_xy, y_pred_xy)
            self.results['scenario_g'][device_id] = metrics

            print(f"Cihaz {device_id}: {metrics['mean_error']:.2f}m")

    def run_all_scenarios(self):
        print("\n" + "="*80)
        print("TÜM SENARYOLAR BAŞLIYOR")
        print("="*80)

        self.scenario_a_device_specific()
        self.scenario_b_device_agnostic()
        self.scenario_c_cross_device()
        self.scenario_d_transfer_learning()
        self.scenario_e_normalization()
        self.scenario_f_ensemble()
        self.scenario_g_feature_engineering()

        print("\n" + "="*80)
        print("TÜM SENARYOLAR TAMAMLANDI")
        print("="*80)

    def plot_cdf(self, errors_dict, title, filename):
        plt.figure(figsize=(10, 6))

        for label, errors in errors_dict.items():
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            plt.plot(sorted_errors, cdf, label=label, linewidth=2)

        plt.xlabel('Positioning Error (m)', fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_comparison_table(self, scen_a, scen_b, filename):
        devices = sorted(list(scen_a.keys()))

        fig, ax = plt.subplots(figsize=(14, len(devices) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')

        table_data = [['Device', 'A: Mean', 'B: Mean', 'A: Median', 'B: Median', 'A: 90%', 'B: 90%']]

        for dev in devices:
            if dev in scen_b:
                table_data.append([
                    str(dev),
                    f"{scen_a[dev]['mean_error']:.2f}",
                    f"{scen_b[dev]['mean_error']:.2f}",
                    f"{scen_a[dev]['median_error']:.2f}",
                    f"{scen_b[dev]['median_error']:.2f}",
                    f"{scen_a[dev]['error_90']:.2f}",
                    f"{scen_b[dev]['error_90']:.2f}"
                ])

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.14, 0.14])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        for i in range(7):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('Scenario A vs B Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_scenarios(self, filename):
        scenarios = ['scenario_a', 'scenario_b', 'scenario_e', 'scenario_f', 'scenario_g']
        names = ['A: Device-Spec', 'B: Device-Agn', 'E: Normalized', 'F: Ensemble', 'G: Features']

        devices = sorted(list(self.results['scenario_a'].keys()))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        x = np.arange(len(devices))
        width = 0.15

        # Mean Error
        ax = axes[0]
        for i, (scen, name) in enumerate(zip(scenarios, names)):
            means = [self.results[scen].get(dev, {}).get('mean_error', 0) for dev in devices]
            ax.bar(x + i * width, means, width, label=name)
        ax.set_xlabel('Device ID', fontweight='bold')
        ax.set_ylabel('Mean Error (m)', fontweight='bold')
        ax.set_title('Mean Positioning Error', fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(devices)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Median Error
        ax = axes[1]
        for i, (scen, name) in enumerate(zip(scenarios, names)):
            medians = [self.results[scen].get(dev, {}).get('median_error', 0) for dev in devices]
            ax.bar(x + i * width, medians, width, label=name)
        ax.set_xlabel('Device ID', fontweight='bold')
        ax.set_ylabel('Median Error (m)', fontweight='bold')
        ax.set_title('Median Positioning Error', fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(devices)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 90th Percentile
        ax = axes[2]
        for i, (scen, name) in enumerate(zip(scenarios, names)):
            p90s = [self.results[scen].get(dev, {}).get('error_90', 0) for dev in devices]
            ax.bar(x + i * width, p90s, width, label=name)
        ax.set_xlabel('Device ID', fontweight='bold')
        ax.set_ylabel('90th Percentile (m)', fontweight='bold')
        ax.set_title('90th Percentile Error', fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(devices)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # RMSE
        ax = axes[3]
        for i, (scen, name) in enumerate(zip(scenarios, names)):
            rmses = [self.results[scen].get(dev, {}).get('rmse', 0) for dev in devices]
            ax.bar(x + i * width, rmses, width, label=name)
        ax.set_xlabel('Device ID', fontweight='bold')
        ax.set_ylabel('RMSE (m)', fontweight='bold')
        ax.set_title('Root Mean Square Error', fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(devices)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def statistical_tests(self):
        print("\n" + "="*80)
        print("İSTATİSTİKSEL TESTLER")
        print("="*80)

        devices = list(self.results['scenario_a'].keys())

        errors_a = []
        errors_b = []

        for dev in devices:
            if dev in self.results['scenario_b']:
                errors_a.extend(self.results['scenario_a'][dev]['euclidean_errors'])
                errors_b.extend(self.results['scenario_b'][dev]['euclidean_errors'])

        min_len = min(len(errors_a), len(errors_b))

        t_stat, p_val = stats.ttest_rel(errors_a[:min_len], errors_b[:min_len])
        print(f"\nA vs B (t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4e}")
        print(f"  Sonuç: {'ANLAMLI' if p_val < 0.05 else 'ANLAMLI DEĞİL'} (p<0.05)")

        scenario_errors = []
        for scen in ['scenario_a', 'scenario_b', 'scenario_e', 'scenario_f', 'scenario_g']:
            errors = []
            for dev in devices:
                if dev in self.results[scen]:
                    errors.extend(self.results[scen][dev]['euclidean_errors'])
            if errors:
                scenario_errors.append(errors)

        if len(scenario_errors) >= 2:
            f_stat, p_val = stats.f_oneway(*scenario_errors)
            print(f"\nTüm Senaryolar (ANOVA):")
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  p-value: {p_val:.4e}")
            print(f"  Sonuç: {'ANLAMLI' if p_val < 0.05 else 'ANLAMLI DEĞİL'} (p<0.05)")

    def generate_report(self, output_dir='Results_Cc'):
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print("RAPOR OLUŞTURULUYOR")
        print("="*80)

        devices = list(self.results['scenario_a'].keys())

        # CDF Grafikleri
        errors_a = {f"Device {dev}": self.results['scenario_a'][dev]['euclidean_errors'] for dev in devices}
        self.plot_cdf(errors_a, 'CDF - Scenario A (Device-Specific)', f'{output_dir}/cdf_scenario_a.png')

        errors_b = {f"Device {dev}": self.results['scenario_b'][dev]['euclidean_errors']
                   for dev in devices if dev in self.results['scenario_b']}
        self.plot_cdf(errors_b, 'CDF - Scenario B (Device-Agnostic)', f'{output_dir}/cdf_scenario_b.png')

        errors_d = {f"Device {dev}": self.results['scenario_d'][dev]['euclidean_errors']
                    for dev in devices if dev in self.results['scenario_d']}
        self.plot_cdf(errors_d, 'CDF - Scenario C (Transfer Learning)', f'{output_dir}/cdf_scenario_d.png')

        errors_e = {f"Device {dev}": self.results['scenario_e'][dev]['euclidean_errors']
                    for dev in devices if dev in self.results['scenario_b']}
        self.plot_cdf(errors_e, 'CDF - Scenario D (RSSI Normalization)', f'{output_dir}/cdf_scenario_e.png')


        # Karşılaştırma
        self.plot_comparison_table(self.results['scenario_a'], self.results['scenario_b'],
                                   f'{output_dir}/comparison_a_vs_b.png')

        # Tüm senaryolar
        self.plot_all_scenarios(f'{output_dir}/all_scenarios.png')

        # İstatistiksel testler
        self.statistical_tests()

        # Excel
        self.export_excel(f'{output_dir}/results.xlsx')

    def export_excel(self, filename):
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for scen_name, scen_results in self.results.items():
                if not scen_results:
                    continue

                if scen_name == 'scenario_c':  # Cross-device has different structure
                    continue

                df = pd.DataFrame([
                    {
                        'Device': dev,
                        'Mean_Error': metrics['mean_error'],
                        'Median_Error': metrics['median_error'],
                        'Std_Error': metrics['std_error'],
                        'RMSE': metrics['rmse'],
                        'Error_50': metrics['error_50'],
                        'Error_75': metrics['error_75'],
                        'Error_90': metrics['error_90'],
                        'Error_95': metrics['error_95']
                    }
                    for dev, metrics in scen_results.items()
                ])
                sheet_name = scen_name.replace('scenario_', 'Scen_').upper()
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    analyzer = IndoorPositioningAnalyzer(
        rssi_train_path='Data/RSS_training.csv',
        coor_train_path='Data/Coordinates_training.csv',
        rssi_test_path='Data/RSS_testing.csv',
        coor_test_path='Data/Coordinates_testing.csv'
    )

    analyzer.run_all_scenarios()
    analyzer.generate_report()

    print("\n✅ ANALİZ TAMAMLANDI!")


if __name__ == "__main__":
    main()