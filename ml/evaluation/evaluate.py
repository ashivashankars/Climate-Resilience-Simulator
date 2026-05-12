"""Full evaluation pipeline: test set metrics, residual analysis, calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data.preprocessor import ClimatePreprocessor
from ml.evaluation.metrics import compute_metrics, metrics_table


def evaluate_model(
    model:       nn.Module,
    test_loader: DataLoader,
    preprocessor: ClimatePreprocessor,
    output_dir:  str = "ml/artifacts",
    device:      Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run full test set evaluation and save results.

    Produces
    --------
    - test_metrics.json  — all scalar metrics
    - predictions.npz    — pred + target arrays for downstream plotting
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds  = []
    all_tgts   = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds   = model(X_batch)

            pred_r = preds["resilience"].cpu().numpy()
            pred_t = preds["temp"].cpu().numpy()
            pred_f = preds["flood"].cpu().numpy()
            energy = preds["energy"]
            pred_e = energy[:, :1].cpu().numpy() if energy.shape[1] == 2 else energy.cpu().numpy()

            batch_preds = np.concatenate([pred_r, pred_t, pred_f, pred_e], axis=1)
            all_preds.append(batch_preds)
            all_tgts.append(y_batch.numpy())

    preds_np  = np.vstack(all_preds)
    tgts_np   = np.vstack(all_tgts)

    # Metrics on normalised scale
    norm_metrics = compute_metrics(preds_np, tgts_np)

    # Inverse-transform to original scale for interpretable RMSE
    preds_orig = preprocessor.inverse_transform_y(preds_np)
    tgts_orig  = preprocessor.inverse_transform_y(tgts_np)
    orig_metrics = compute_metrics(preds_orig, tgts_orig)
    orig_metrics = {f"orig_{k}": v for k, v in orig_metrics.items()}

    all_metrics = {**norm_metrics, **orig_metrics}

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "test_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    np.savez(
        out / "predictions.npz",
        preds_norm=preds_np, tgts_norm=tgts_np,
        preds_orig=preds_orig, tgts_orig=tgts_orig,
    )

    print("\nTest Set Results (original scale):")
    print(metrics_table(orig_metrics))

    return all_metrics
