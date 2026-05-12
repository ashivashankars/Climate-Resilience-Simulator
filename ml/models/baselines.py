"""
Baseline models for ablation study comparison.

Purpose
-------
Comparing ClimateResilienceNet against these baselines quantifies the
contribution of: (a) deep learning over shallow models, (b) attention
mechanism over plain MLP, (c) physics-informed priors over pure learning.

Models
------
LinearBaseline       — multivariate linear regression per output
RandomForestBaseline — sklearn RF (strong non-linear baseline)
MLPBaseline          — plain feedforward net (no attention, no physics)
GradientBoostBaseline— XGBoost-style GBT (gold standard for tabular data)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


# ---------------------------------------------------------------------------
# Scikit-learn baselines (for tabular comparison)
# ---------------------------------------------------------------------------

class LinearBaseline:
    """Ridge regression with default alpha — upper bound on linear capacity."""

    def __init__(self, alpha: float = 1.0):
        self.model = MultiOutputRegressor(Ridge(alpha=alpha))
        self.name  = "LinearBaseline"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestBaseline:
    """Random forest — strong shallow non-linear baseline for tabular data."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 12,
                 n_jobs: int = -1, random_state: int = 42):
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_leaf=5, n_jobs=n_jobs, random_state=random_state,
            )
        )
        self.name = "RandomForestBaseline"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        importances = np.mean(
            [est.feature_importances_ for est in self.model.estimators_], axis=0
        )
        return dict(sorted(zip(feature_names, importances),
                            key=lambda x: x[1], reverse=True))


class GradientBoostBaseline:
    """Gradient boosting — current SOTA for tabular regression."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 max_depth: int = 6, random_state: int = 42):
        self.model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth, subsample=0.8, random_state=random_state,
            )
        )
        self.name = "GradientBoostBaseline"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# ---------------------------------------------------------------------------
# PyTorch MLP baseline
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    """
    Plain MLP with residual connections — used as the 'no_attention' ablation.
    Architecture: Input → [Linear → BN → GELU → Dropout → Linear + skip] × 4 → Heads
    """

    def __init__(self, n_features: int = 36, hidden_dim: int = 256,
                 n_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.name = "ResidualMLP"

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SELU(),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = F.gelu(h + block(h))
        return self.head(h)   # (B, 4) raw logits — apply activations outside

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
