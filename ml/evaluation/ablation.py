"""
Ablation study framework.

Systematically disables model components to quantify each one's contribution.
Results are logged to MLflow for side-by-side comparison.

Ablation variants
-----------------
1. full              — Full ClimateResilienceNet (baseline)
2. no_physics_prior  — Remove IPCC physics prior injection
3. no_attention      — Replace transformer with plain MLP (no feature interaction)
4. no_uncertainty    — Disable aleatoric uncertainty on energy head
5. shallow           — 2-layer transformer (vs 4-layer)
6. narrow            — d_model=64 (vs 128); tests model capacity
7. no_augmentation   — Train without data augmentation
8. linear_baseline   — Ridge regression (upper bound of linear models)
9. random_forest     — RF (strong non-linear baseline)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ml.data.dataset import build_dataloaders
from ml.data.preprocessor import ClimatePreprocessor
from ml.evaluation.metrics import compute_metrics, TARGET_NAMES
from ml.models.baselines import LinearBaseline, RandomForestBaseline
from ml.models.climate_net import build_model_variant
from ml.models.losses import ClimateAwareLoss
from ml.training.trainer import Trainer


ABLATION_VARIANTS = [
    "full",
    "no_physics_prior",
    "no_attention",
    "no_uncertainty",
    "shallow",
    "narrow",
]

SKLEARN_BASELINES = ["linear_baseline", "random_forest"]


def run_single_variant(
    variant: str,
    X_tr: np.ndarray, X_val: np.ndarray, X_te: np.ndarray,
    y_tr: np.ndarray, y_val: np.ndarray, y_te: np.ndarray,
    preprocessor: ClimatePreprocessor,
    config: Dict,
    output_dir: str = "ml/artifacts/ablation",
) -> Dict[str, float]:
    """Run one ablation variant and return test metrics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    # ---- sklearn baselines ----
    if variant == "linear_baseline":
        model = LinearBaseline()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        metrics = compute_metrics(preds, y_te)
        print(f"[{variant}] R²_mean={metrics['r2_mean']:.4f}")
        return metrics

    if variant == "random_forest":
        model = RandomForestBaseline(n_estimators=100)  # Fast for ablation
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        metrics = compute_metrics(preds, y_te)
        print(f"[{variant}] R²_mean={metrics['r2_mean']:.4f}")
        return metrics

    # ---- PyTorch variants ----
    use_aug = variant != "no_augmentation"
    train_loader, val_loader, test_loader = build_dataloaders(
        X_tr, X_val, X_te, y_tr, y_val, y_te,
        batch_size=config.get("batch_size", 512),
        augment=use_aug,
    )

    model = build_model_variant(variant, n_features=X_tr.shape[1])
    loss_fn = ClimateAwareLoss()

    ablation_cfg = {
        **config,
        "epochs":   config.get("ablation_epochs", 40),
        "patience": 10,
        "experiment": "climate-resilience-ablation",
        "output_dir": f"{output_dir}/{variant}",
    }

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=ablation_cfg,
        output_dir=ablation_cfg["output_dir"],
        experiment="climate-resilience-ablation",
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=ablation_cfg["epochs"],
        run_name=f"ablation_{variant}",
    )

    # Evaluate on test
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            out = model(X_b)
            pr  = out["resilience"].cpu().numpy()
            pt  = out["temp"].cpu().numpy()
            pf  = out["flood"].cpu().numpy()
            en  = out["energy"]
            pe  = en[:, :1].cpu().numpy() if en.shape[1] == 2 else en.cpu().numpy()
            all_preds.append(np.concatenate([pr, pt, pf, pe], axis=1))
            all_tgts.append(y_b.numpy())

    preds_np = np.vstack(all_preds)
    tgts_np  = np.vstack(all_tgts)
    metrics  = compute_metrics(preds_np, tgts_np)
    print(f"[{variant}] R²_mean={metrics['r2_mean']:.4f}  params={model.count_parameters():,}")
    return metrics


def run_ablation_study(
    X_tr: np.ndarray, X_val: np.ndarray, X_te: np.ndarray,
    y_tr: np.ndarray, y_val: np.ndarray, y_te: np.ndarray,
    preprocessor: ClimatePreprocessor,
    config: Dict,
    variants: Optional[List[str]] = None,
    output_dir: str = "ml/artifacts/ablation",
) -> pd.DataFrame:
    """
    Run all ablation variants and produce a comparison table.

    Returns a DataFrame suitable for matplotlib plotting.
    """
    if variants is None:
        variants = ABLATION_VARIANTS + SKLEARN_BASELINES

    all_results: Dict[str, Dict] = {}

    for variant in variants:
        print(f"\n{'─'*50}")
        print(f"Ablation: {variant}")
        print(f"{'─'*50}")
        try:
            metrics = run_single_variant(
                variant, X_tr, X_val, X_te, y_tr, y_val, y_te,
                preprocessor, config, output_dir,
            )
            all_results[variant] = metrics
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[variant] = {}

    # Build comparison DataFrame
    rows = []
    for variant, metrics in all_results.items():
        row = {"variant": variant}
        for name in TARGET_NAMES:
            row[f"r2_{name}"]   = metrics.get(f"r2_{name}",   np.nan)
            row[f"rmse_{name}"] = metrics.get(f"rmse_{name}", np.nan)
        row["r2_mean"] = metrics.get("r2_mean", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("variant")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{output_dir}/ablation_results.csv")
    with open(f"{output_dir}/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\nAblation Study Summary")
    print("=" * 80)
    print(df[["r2_mean"] + [f"r2_{n}" for n in TARGET_NAMES]].to_string())

    return df
