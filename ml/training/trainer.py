"""
Training loop for ClimateResilienceNet.

MLOps integration
-----------------
- MLflow: tracks every run (params, metrics, artifacts, model registry)
- TensorBoard: real-time training curves (loss, per-output R², LR)
- Checkpointing: saves best model by val_loss + final checkpoint
- Early stopping: patience-based, prevents over-fitting
- Gradient clipping: stabilises transformer training
- Cosine annealing with warm restarts (SGDR): escapes local minima
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.models.climate_net import ClimateResilienceNet
from ml.models.losses import ClimateAwareLoss
from ml.evaluation.metrics import compute_metrics, TARGET_NAMES


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4,
                 restore_best: bool = True):
        self.patience     = patience
        self.min_delta    = min_delta
        self.restore_best = restore_best
        self.best_loss    = float("inf")
        self.counter      = 0
        self.best_state   = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            if self.restore_best:
                self.best_state = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}
            return False  # Continue
        self.counter += 1
        return self.counter >= self.patience  # Stop

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    """
    Orchestrates training, validation, checkpointing, and MLflow logging.

    Parameters
    ----------
    model        : ClimateResilienceNet instance
    loss_fn      : ClimateAwareLoss instance
    config       : training hyper-parameters dict
    output_dir   : where to save checkpoints and artifacts
    experiment   : MLflow experiment name
    """

    def __init__(
        self,
        model:       ClimateResilienceNet,
        loss_fn:     ClimateAwareLoss,
        config:      Dict,
        output_dir:  str = "ml/artifacts",
        experiment:  str = "climate-resilience",
    ):
        self.model      = model
        self.loss_fn    = loss_fn
        self.config     = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # MPS (Apple Silicon) has known allocator bugs with Python 3.14 + complex
        # transformer ops. CPU is reliable; still fast enough for this model size.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        print(f"Training on: {self.device}")
        print(self.model.summary())

        # Optimizer: AdamW with decoupled weight decay (Loshchilov & Hutter 2019)
        # Chosen over Adam for transformer training — better regularisation
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 1e-4),
            betas=(0.9, 0.999),
        )

        # Cosine Annealing with Warm Restarts (SGDR, Loshchilov 2017)
        # T_0=10: restart every 10 epochs, T_mult=2: double period each restart
        # Chosen because it explores different loss basins — helpful when
        # physics-prior and attention interact non-convexly
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get("T_0", 10),
            T_mult=config.get("T_mult", 2),
            eta_min=config.get("lr_min", 1e-6),
        )

        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 20),
            min_delta=config.get("min_delta", 1e-4),
        )

        # MLflow setup
        mlflow.set_experiment(experiment)
        self.run_id: Optional[str] = None

        # TensorBoard
        tb_dir = self.output_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard logs → {tb_dir}")
        print(f"  Run: tensorboard --logdir {tb_dir}")

        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # Training primitives
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss  = 0.0
        loss_parts  = {"mse_loss": 0.0, "uncertainty_loss": 0.0,
                       "monotonicity_loss": 0.0, "physics_loss": 0.0}
        n_batches   = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            preds  = self.model(X_batch)
            losses = self.loss_fn(preds, y_batch, X_batch)
            if not torch.isfinite(losses["total"]):
                continue  # Skip batch if loss is NaN/Inf
            losses["total"].backward()

            # Gradient clipping (essential for transformer stability)
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     max_norm=self.config.get("grad_clip", 1.0))
            self.optimizer.step()

            total_loss += losses["total"].item()
            for k in loss_parts:
                val = losses[k]
                loss_parts[k] += val.item() if isinstance(val, torch.Tensor) else val
            n_batches += 1

        self.scheduler.step()

        n_batches = max(n_batches, 1)
        return {"total": total_loss / n_batches,
                **{k: v / n_batches for k, v in loss_parts.items()}}

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Tuple[Dict, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds  = []
        all_tgts   = []
        n_batches  = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            preds  = self.model(X_batch)
            losses = self.loss_fn(preds, y_batch, X_batch)
            total_loss += losses["total"].item()

            # Collect predictions (use only μ for uncertainty heads)
            pred_r = preds["resilience"].cpu().numpy()
            pred_t = preds["temp"].cpu().numpy()
            pred_f = preds["flood"].cpu().numpy()
            energy = preds["energy"]
            pred_e = energy[:, :1].cpu().numpy() if energy.shape[1] == 2 else energy.cpu().numpy()
            batch_preds = np.concatenate([pred_r, pred_t, pred_f, pred_e], axis=1)

            # Guard against NaN outputs (can occur in early epochs before convergence)
            if not np.isfinite(batch_preds).all():
                batch_preds = np.nan_to_num(batch_preds, nan=0.0, posinf=0.0, neginf=0.0)

            all_preds.append(batch_preds)
            all_tgts.append(y_batch.cpu().numpy())
            n_batches += 1

        preds_np = np.vstack(all_preds)
        tgts_np  = np.vstack(all_tgts)
        metrics  = compute_metrics(preds_np, tgts_np)
        metrics["loss"] = total_loss / n_batches
        return metrics, preds_np, tgts_np

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        n_epochs:     int = 100,
        run_name:     Optional[str] = None,
    ) -> Dict:
        """
        Full training loop with MLflow tracking and TensorBoard logging.

        Returns best validation metrics dict.
        """
        with mlflow.start_run(run_name=run_name) as run:
            self.run_id = run.info.run_id
            print(f"\nMLflow run ID: {self.run_id}")

            # Log all hyper-parameters
            mlflow.log_params({
                "n_features":        self.model.n_features,
                "d_model":           self.model.d_model,
                "use_physics_prior": self.model.use_physics_prior,
                "use_attention":     self.model.use_attention,
                "use_uncertainty":   self.model.use_uncertainty,
                "n_parameters":      self.model.count_parameters(),
                **{f"hp/{k}": v for k, v in self.config.items()},
            })

            best_val_metrics = None
            t0 = time.time()

            for epoch in range(1, n_epochs + 1):
                # Train
                train_losses = self._train_epoch(train_loader)

                # Validate
                val_metrics, val_preds, val_tgts = self._eval_epoch(val_loader)

                # Current LR
                lr_now = self.optimizer.param_groups[0]["lr"]

                # ---- Console output ----
                if epoch % 5 == 0 or epoch <= 3:
                    elapsed = time.time() - t0
                    r2_mean = np.mean([val_metrics.get(f"r2_{n}", 0) for n in TARGET_NAMES])
                    print(
                        f"[{epoch:>3}/{n_epochs}] "
                        f"train_loss={train_losses['total']:.4f}  "
                        f"val_loss={val_metrics['loss']:.4f}  "
                        f"val_R²={r2_mean:.4f}  "
                        f"lr={lr_now:.2e}  "
                        f"({elapsed:.0f}s)"
                    )

                # ---- TensorBoard ----
                self.writer.add_scalar("Loss/train",      train_losses["total"],    epoch)
                self.writer.add_scalar("Loss/train_mse",  train_losses["mse_loss"], epoch)
                self.writer.add_scalar("Loss/train_mono", train_losses["monotonicity_loss"], epoch)
                self.writer.add_scalar("Loss/train_phys", train_losses["physics_loss"], epoch)
                self.writer.add_scalar("Loss/val",        val_metrics["loss"],      epoch)
                self.writer.add_scalar("LR",              lr_now,                   epoch)

                for name in TARGET_NAMES:
                    self.writer.add_scalar(f"R2/val_{name}",  val_metrics.get(f"r2_{name}",  0), epoch)
                    self.writer.add_scalar(f"RMSE/val_{name}", val_metrics.get(f"rmse_{name}", 0), epoch)

                # ---- MLflow (every 5 epochs to reduce overhead) ----
                if epoch % 5 == 0:
                    flat_metrics = {
                        "train_loss": train_losses["total"],
                        "val_loss":   val_metrics["loss"],
                        "val_r2_mean": np.mean([val_metrics.get(f"r2_{n}", 0) for n in TARGET_NAMES]),
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }
                    mlflow.log_metrics(flat_metrics, step=epoch)

                # ---- History ----
                self.history.append({
                    "epoch":      epoch,
                    "train_loss": train_losses["total"],
                    "val_loss":   val_metrics["loss"],
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                })

                # ---- Early stopping ----
                stop = self.early_stopping.step(val_metrics["loss"], self.model)
                if stop:
                    print(f"\nEarly stopping at epoch {epoch} (patience={self.config.get('patience',20)})")
                    self.early_stopping.restore(self.model)
                    # Final eval with restored best weights
                    best_val_metrics, _, _ = self._eval_epoch(val_loader)
                    break

                if best_val_metrics is None or val_metrics["loss"] < best_val_metrics.get("loss", 1e9):
                    best_val_metrics = val_metrics.copy()
                    self._save_checkpoint("best_model.pt")

            # ---- Final logging ----
            best_metrics_flat = {f"best_{k}": v for k, v in (best_val_metrics or val_metrics).items()}
            mlflow.log_metrics(best_metrics_flat)

            # Save final model to MLflow model registry
            mlflow.pytorch.log_model(
                self.model,
                artifact_path="climate_resilience_model",
                registered_model_name="ClimateResilienceNet",
            )

            # Save training history
            history_path = self.output_dir / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)
            mlflow.log_artifact(str(history_path))

            # Save config
            cfg_path = self.output_dir / "run_config.json"
            with open(cfg_path, "w") as f:
                json.dump(self.config, f, indent=2)
            mlflow.log_artifact(str(cfg_path))

            print(f"\nBest val metrics:")
            for k, v in (best_val_metrics or val_metrics).items():
                print(f"  {k}: {v:.4f}")

        self.writer.close()
        return best_val_metrics or val_metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str) -> None:
        path = self.output_dir / filename
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config":               self.config,
            "n_features":           self.model.n_features,
            "d_model":              self.model.d_model,
            "use_physics_prior":    self.model.use_physics_prior,
            "use_attention":        self.model.use_attention,
            "use_uncertainty":      self.model.use_uncertainty,
        }, path)

    def load_checkpoint(self, filename: str = "best_model.pt") -> None:
        path = self.output_dir / filename
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {path}")
