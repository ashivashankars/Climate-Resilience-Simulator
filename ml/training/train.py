"""
Entry point for ClimateResilienceNet training.

Usage
-----
# Standard training
python -m ml.training.train

# Override config via CLI
python -m ml.training.train --epochs 150 --lr 1e-3 --d-model 256

# Optuna hyperparameter sweep (30 trials)
python -m ml.training.train --sweep --n-trials 30
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

# ---- Optuna (hyper-parameter sweep) ----
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from ml.data.dataset import build_dataloaders
from ml.data.generate_dataset import generate_dataset
from ml.data.preprocessor import ClimatePreprocessor
from ml.models.climate_net import ClimateResilienceNet
from ml.models.losses import ClimateAwareLoss
from ml.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "n_samples":      5000,
    "seed":           42,
    "epochs":         100,
    "batch_size":     128,
    "lr":             3e-4,
    "lr_min":         1e-6,
    "weight_decay":   1e-4,
    "T_0":            10,
    "T_mult":         2,
    "patience":       20,
    "min_delta":      1e-4,
    "grad_clip":      1.0,
    "d_model":        128,
    "n_heads":        8,
    "n_layers":       4,
    "d_ff":           512,
    "dropout":        0.10,
    "augment":        True,
    "use_physics_prior": True,
    "use_attention":     True,
    "use_uncertainty":   True,
    "output_dir":     "ml/artifacts",
    "data_dir":       "ml/data",
    "experiment":     "climate-resilience",
}


def load_config(config_path: Optional[str] = None) -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f)
        cfg.update(file_cfg)
    return cfg


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(cfg: Dict):
    data_dir  = cfg["data_dir"]
    feat_path = Path(data_dir) / "features.csv"
    lbl_path  = Path(data_dir) / "labels.csv"

    if not feat_path.exists() or not lbl_path.exists():
        print("Generating dataset...")
        generate_dataset(
            n_samples=cfg["n_samples"],
            seed=cfg["seed"],
            output_dir=data_dir,
        )

    print("Loading dataset...")
    df_X = pd.read_csv(feat_path)
    df_y = pd.read_csv(lbl_path)
    print(f"  Features: {df_X.shape}  Labels: {df_y.shape}")
    return df_X, df_y


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_training(cfg: Dict) -> Dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # --- Data ---
    df_X, df_y = prepare_data(cfg)
    pre = ClimatePreprocessor(seed=cfg["seed"])
    X_tr, X_val, X_te, y_tr, y_val, y_te = pre.fit_transform_splits(df_X, df_y)
    pre.save(f"{cfg['output_dir']}/preprocessor.joblib")

    print("\nDataset splits:")
    print(f"  Train : {X_tr.shape[0]:,}")
    print(f"  Val   : {X_val.shape[0]:,}")
    print(f"  Test  : {X_te.shape[0]:,}")
    print(f"  Features: {X_tr.shape[1]}")

    train_loader, val_loader, test_loader = build_dataloaders(
        X_tr, X_val, X_te, y_tr, y_val, y_te,
        batch_size=cfg["batch_size"],
        augment=cfg["augment"],
    )

    # --- Model ---
    model = ClimateResilienceNet(
        n_features        = X_tr.shape[1],
        d_model           = cfg["d_model"],
        n_heads           = cfg["n_heads"],
        n_layers          = cfg["n_layers"],
        d_ff              = cfg["d_ff"],
        dropout           = cfg["dropout"],
        use_physics_prior = cfg["use_physics_prior"],
        use_attention     = cfg["use_attention"],
        use_uncertainty   = cfg["use_uncertainty"],
    )

    loss_fn = ClimateAwareLoss(
        weights=(1.0, 1.5, 1.2, 0.8),
        uncertainty_weight=0.3,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=cfg,
        output_dir=cfg["output_dir"],
        experiment=cfg["experiment"],
    )

    # --- Train ---
    val_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=cfg["epochs"],
    )

    # --- Test evaluation ---
    from ml.evaluation.evaluate import evaluate_model
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        preprocessor=pre,
        output_dir=cfg["output_dir"],
        device=trainer.device,
    )

    # Save feature info for inference server
    info = pre.feature_info()
    info["n_features"] = X_tr.shape[1]
    with open(f"{cfg['output_dir']}/feature_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return {"val": val_metrics, "test": test_metrics}


# ---------------------------------------------------------------------------
# Optuna sweep
# ---------------------------------------------------------------------------

def run_sweep(cfg: Dict, n_trials: int = 30) -> None:
    if not HAS_OPTUNA:
        raise ImportError("Install optuna: pip install optuna")

    def objective(trial: "optuna.Trial") -> float:
        trial_cfg = cfg.copy()
        trial_cfg.update({
            "lr":           trial.suggest_float("lr",           1e-4, 1e-2, log=True),
            "d_model":      trial.suggest_categorical("d_model",  [64, 128, 256]),
            "n_heads":      trial.suggest_categorical("n_heads",  [4, 8]),
            "n_layers":     trial.suggest_int("n_layers",         2, 6),
            "dropout":      trial.suggest_float("dropout",        0.05, 0.3),
            "weight_decay": trial.suggest_float("weight_decay",   1e-5, 1e-2, log=True),
            "batch_size":   trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "epochs":       30,  # Shorter for sweep
            "patience":     10,
            "experiment":   "climate-resilience-sweep",
        })
        metrics = run_training(trial_cfg)
        return metrics["val"].get("loss", float("inf"))

    sampler = TPESampler(seed=cfg["seed"])
    pruner  = MedianPruner(n_startup_trials=5)
    study   = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="climate-resilience-hpo",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nBest trial:")
    print(f"  Value: {study.best_value:.4f}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Save best params
    best_path = Path(cfg["output_dir"]) / "best_sweep_params.json"
    with open(best_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest params saved → {best_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ClimateResilienceNet")
    p.add_argument("--config",    default="ml/configs/config.yaml")
    p.add_argument("--epochs",    type=int)
    p.add_argument("--lr",        type=float)
    p.add_argument("--d-model",   type=int)
    p.add_argument("--n-layers",  type=int)
    p.add_argument("--batch-size",type=int)
    p.add_argument("--dropout",   type=float)
    p.add_argument("--sweep",     action="store_true")
    p.add_argument("--n-trials",  type=int, default=30)
    p.add_argument("--no-physics-prior", action="store_true")
    p.add_argument("--no-attention",     action="store_true")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.lr:
        cfg["lr"] = args.lr
    if args.d_model:
        cfg["d_model"] = args.d_model
    if args.n_layers:
        cfg["n_layers"] = args.n_layers
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.dropout:
        cfg["dropout"] = args.dropout
    if args.no_physics_prior:
        cfg["use_physics_prior"] = False
    if args.no_attention:
        cfg["use_attention"] = False
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    os.makedirs(cfg["output_dir"], exist_ok=True)

    if args.sweep:
        run_sweep(cfg, n_trials=args.n_trials)
    else:
        results = run_training(cfg)
        print("\nFinal Results:")
        print(json.dumps(results, indent=2, default=str))
