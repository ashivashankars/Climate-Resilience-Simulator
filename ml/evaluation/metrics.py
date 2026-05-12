"""Per-output and aggregate metrics for climate resilience prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_NAMES = ["resilience", "temp", "flood", "energy"]
TARGET_DISPLAY = {
    "resilience": "Resilience Score",
    "temp":       "Temp Reduction (°F)",
    "flood":      "Flood Risk Reduction (%)",
    "energy":     "Energy Savings ($/yr)",
}


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute per-output and aggregate metrics.

    Parameters
    ----------
    preds   : (N, 4) — model predictions (normalised scale)
    targets : (N, 4) — ground truth (normalised scale)

    Returns
    -------
    flat dict with keys: r2_{name}, rmse_{name}, mae_{name}, + aggregate r2_mean
    """
    metrics: Dict[str, float] = {}
    r2_list = []

    for i, name in enumerate(TARGET_NAMES):
        p = preds[:, i]
        t = targets[:, i]

        r2   = float(r2_score(t, p))
        rmse = float(np.sqrt(mean_squared_error(t, p)))
        mae  = float(mean_absolute_error(t, p))

        metrics[f"r2_{name}"]   = r2
        metrics[f"rmse_{name}"] = rmse
        metrics[f"mae_{name}"]  = mae
        r2_list.append(r2)

    metrics["r2_mean"]   = float(np.mean(r2_list))
    metrics["rmse_mean"] = float(np.mean([metrics[f"rmse_{n}"] for n in TARGET_NAMES]))
    return metrics


def metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics as a markdown table."""
    header = "| Output | R² | RMSE | MAE |"
    sep    = "|---|---|---|---|"
    rows   = [header, sep]
    for name in TARGET_NAMES:
        disp = TARGET_DISPLAY[name]
        r2   = metrics.get(f"r2_{name}",   0.0)
        rmse = metrics.get(f"rmse_{name}", 0.0)
        mae  = metrics.get(f"mae_{name}",  0.0)
        rows.append(f"| {disp} | {r2:.4f} | {rmse:.4f} | {mae:.4f} |")
    rows.append(f"| **Mean** | **{metrics.get('r2_mean', 0):.4f}** | **{metrics.get('rmse_mean', 0):.4f}** | — |")
    return "\n".join(rows)
