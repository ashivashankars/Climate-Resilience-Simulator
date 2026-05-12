"""
Feature engineering and preprocessing pipeline for ClimateResilienceNet.

Follows sklearn transformer API so it plugs cleanly into MLflow pipelines
and can be serialised with joblib.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

# Feature groups — kept explicit so ablation studies can drop groups cleanly
GEO_FEATURES = [
    "lat", "lon", "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    "elevation", "coastal_distance", "coastal_distance_log",
]
CLIMATE_FEATURES = [
    "baseline_temp", "baseline_humidity",
    "baseline_solar_radiation", "baseline_wind_speed",
]
ZONE_FEATURES = [
    "zone_tropical", "zone_subtropical", "zone_temperate", "zone_cold",
]
TEMPORAL_FEATURES = ["year_norm", "year_sin", "year_cos"]
INTERVENTION_FEATURES = [
    "intervention_green_roof", "intervention_solar_panels",
    "intervention_flood_walls", "intervention_permeable",
    "n_interventions",
]
BUILDING_FEATURES = [
    "building_age", "building_area", "building_area_log", "building_type",
]
INTERACTION_FEATURES = [
    "solar_x_green", "solar_x_solar_panel", "coast_x_flood_wall",
]

ALL_FEATURE_GROUPS = {
    "geo":          GEO_FEATURES,
    "climate":      CLIMATE_FEATURES,
    "zone":         ZONE_FEATURES,
    "temporal":     TEMPORAL_FEATURES,
    "intervention": INTERVENTION_FEATURES,
    "building":     BUILDING_FEATURES,
    "interaction":  INTERACTION_FEATURES,
}

TARGET_COLS = [
    "resilience_score",      # 0–100
    "temp_reduction_f",      # 0–15 °F
    "flood_risk_reduction",  # 0–90 %
    "energy_savings_usd",    # 0–5000 $/yr
]

# Targets that require log-normalisation before training (heavy right skew)
LOG_TARGETS = {"energy_savings_usd"}

# Continuous numeric features to standardise (binary / one-hot excluded)
NUMERIC_SCALE_FEATURES = (
    GEO_FEATURES[6:]        # elevation, coastal_distance, coastal_distance_log
    + CLIMATE_FEATURES
    + TEMPORAL_FEATURES[:1]  # year_norm only (sin/cos already bounded)
    + BUILDING_FEATURES[:3]  # age, area, area_log
)


class ClimatePreprocessor:
    """
    Fit-transform pipeline for the climate resilience dataset.

    Usage
    -----
    pre = ClimatePreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = pre.fit_transform_splits(df_X, df_y)
    pre.save("ml/data/preprocessor.joblib")
    """

    def __init__(self,
                 feature_groups: Optional[List[str]] = None,
                 target_scaler: bool = True,
                 val_size: float = 0.10,
                 test_size: float = 0.10,
                 seed: int = 42):
        self.feature_groups  = feature_groups or list(ALL_FEATURE_GROUPS.keys())
        self.target_scaler   = target_scaler
        self.val_size        = val_size
        self.test_size       = test_size
        self.seed            = seed

        self._feature_scaler = RobustScaler()  # Robust to outliers (coastal_distance)
        self._target_scaler  = StandardScaler()
        self.feature_names_: List[str] = []
        self.fitted_         = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def selected_features(self) -> List[str]:
        cols = []
        for g in self.feature_groups:
            cols += ALL_FEATURE_GROUPS[g]
        return cols

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)

    @property
    def n_targets(self) -> int:
        return len(TARGET_COLS)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def fit_transform_splits(
        self,
        df_X: pd.DataFrame,
        df_y: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:

        X_raw = self._select_features(df_X)
        y_raw = self._prepare_targets(df_y)

        # Stratified split by climate zone (preserve distribution)
        zone_col = df_X["zone_tropical"].astype(str) + df_X["zone_subtropical"].astype(str) + \
                   df_X["zone_temperate"].astype(str) + df_X["zone_cold"].astype(str)
        strat = zone_col.values

        X_tr, X_tmp, y_tr, y_tmp, s_tr, s_tmp = train_test_split(
            X_raw, y_raw, strat, test_size=self.val_size + self.test_size,
            random_state=self.seed, stratify=strat,
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=self.test_size / (self.val_size + self.test_size),
            random_state=self.seed,
        )

        # Fit scalers only on training data to avoid leakage
        numeric_idx = [i for i, c in enumerate(self.feature_names_)
                       if c in NUMERIC_SCALE_FEATURES]
        self._numeric_idx = numeric_idx
        self._feature_scaler.fit(X_tr[:, numeric_idx])
        if self.target_scaler:
            self._target_scaler.fit(y_tr)
        self.fitted_ = True

        return (
            self._scale_X(X_tr),
            self._scale_X(X_val),
            self._scale_X(X_te),
            self._scale_y(y_tr),
            self._scale_y(y_val),
            self._scale_y(y_te),
        )

    def transform(self, df_X: pd.DataFrame) -> np.ndarray:
        assert self.fitted_, "Call fit_transform_splits first."
        X_raw = self._select_features(df_X)
        return self._scale_X(X_raw)

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.target_scaler:
            y = self._target_scaler.inverse_transform(y_scaled)
        else:
            y = y_scaled.copy()
        # Reverse log transform on energy savings column
        energy_idx = TARGET_COLS.index("energy_savings_usd")
        y[:, energy_idx] = np.expm1(y[:, energy_idx])
        return np.clip(y, 0, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = self.selected_features
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self.feature_names_ = cols
        return df[cols].values.astype(np.float32)

    def _prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        y = df[TARGET_COLS].values.astype(np.float32)
        # Log-normalise skewed targets
        energy_idx = TARGET_COLS.index("energy_savings_usd")
        y[:, energy_idx] = np.log1p(y[:, energy_idx])
        return y

    def _scale_X(self, X: np.ndarray) -> np.ndarray:
        Xs = X.copy()
        Xs[:, self._numeric_idx] = self._feature_scaler.transform(X[:, self._numeric_idx])
        return Xs.astype(np.float32)

    def _scale_y(self, y: np.ndarray) -> np.ndarray:
        if self.target_scaler:
            return self._target_scaler.transform(y).astype(np.float32)
        return y.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        print(f"Preprocessor saved → {path}")

    @staticmethod
    def load(path: str) -> "ClimatePreprocessor":
        return joblib.load(path)

    def feature_info(self) -> dict:
        return {
            "feature_names":  self.feature_names_,
            "n_features":     self.n_features,
            "n_targets":      self.n_targets,
            "target_names":   TARGET_COLS,
            "feature_groups": {g: ALL_FEATURE_GROUPS[g] for g in self.feature_groups},
        }
