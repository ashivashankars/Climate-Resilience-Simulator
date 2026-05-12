"""
Inference pipeline for ClimateResilienceNet.

Wraps the trained model + preprocessor into a single callable that
accepts raw location/intervention inputs and returns human-readable predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ml.data.generate_dataset import (
    CITIES, PROJECTION_YEARS,
    build_feature_row,
)
from ml.data.preprocessor import ClimatePreprocessor
from ml.models.climate_net import ClimateResilienceNet


class ClimatePredictor:
    """
    End-to-end inference: raw inputs → climate resilience predictions.

    Loads trained model and preprocessor from disk and provides a simple
    .predict() API used by both the FastAPI server and the Gradio demo.

    Parameters
    ----------
    model_path  : path to best_model.pt checkpoint
    pre_path    : path to preprocessor.joblib
    device      : 'cpu' | 'cuda' | 'mps'
    n_mc_samples: number of MC-Dropout samples for uncertainty (0 = disabled)
    """

    def __init__(
        self,
        model_path:    str = "ml/artifacts/best_model.pt",
        pre_path:      str = "ml/artifacts/preprocessor.joblib",
        device:        str = "cpu",
        n_mc_samples:  int = 0,
    ):
        self.device       = torch.device(device)
        self.n_mc_samples = n_mc_samples

        # Load preprocessor
        self.pre = ClimatePreprocessor.load(pre_path)

        # Load model
        ckpt = torch.load(model_path, map_location=self.device)
        self.model = ClimateResilienceNet(
            n_features        = ckpt["n_features"],
            d_model           = ckpt.get("d_model", 128),
            use_physics_prior = ckpt.get("use_physics_prior", True),
            use_attention     = ckpt.get("use_attention",     True),
            use_uncertainty   = ckpt.get("use_uncertainty",   True),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Predictor ready ({self.model.count_parameters():,} params)")

    # ------------------------------------------------------------------

    def _build_input(
        self,
        location:           str,
        year:               int,
        interventions:      Dict[str, bool],
        building_type:      str = "residential",
        building_age:       float = 20.0,
        building_area:      float = 150.0,
    ) -> np.ndarray:
        """Build raw feature row from human-readable inputs."""
        # Look up city data; fall back to approximate lat/lon
        loc_key = location.lower().replace(" ", "_")
        city = CITIES.get(loc_key)
        if city is None:
            # Try partial match
            for key, data in CITIES.items():
                if loc_key in key or key in loc_key:
                    city = data
                    break
        if city is None:
            # Default: temperate city approximation
            city = dict(lat=37.77, lon=-122.42, elev=52, coast=50,
                        temp=14.5, hum=70, solar=180, wind=5.0)

        year = min(PROJECTION_YEARS, key=lambda y: abs(y - year))  # Snap to valid year

        row = build_feature_row(
            city, loc_key, year, interventions,
            building_type, building_age, building_area,
        )
        df_row = {k: [v] for k, v in row.items() if k != "city"}
        import pandas as pd
        return pd.DataFrame(df_row)

    def predict(
        self,
        location:       str,
        year:           int = 2050,
        interventions:  Optional[Dict[str, bool]] = None,
        building_type:  str = "residential",
        building_age:   float = 20.0,
        building_area:  float = 150.0,
    ) -> Dict:
        """
        Run inference for a single location.

        Parameters
        ----------
        location      : city name (e.g. 'Miami', 'Phoenix')
        year          : projection year (2030/2050/2070/2100)
        interventions : dict with keys: green_roof, solar_panels, flood_walls, permeable_pavement
        building_type : 'residential' | 'commercial' | 'industrial' | 'mixed_use' | 'institutional'
        building_age  : years since construction (0–80)
        building_area : building footprint m² (50–5000)

        Returns
        -------
        dict with human-readable predictions + uncertainty ranges
        """
        if interventions is None:
            interventions = {"green_roof": True, "solar_panels": True,
                             "flood_walls": True, "permeable_pavement": True}

        df_feat = self._build_input(location, year, interventions,
                                     building_type, building_age, building_area)
        X = self.pre.transform(df_feat)
        X_t = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            if self.n_mc_samples > 0:
                # MC Dropout for epistemic uncertainty
                self.model.train()  # Enable dropout
                mc_outputs = [self.model(X_t) for _ in range(self.n_mc_samples)]
                self.model.eval()

                def mc_mean_std(key):
                    vals = torch.cat([o[key][:, :1] for o in mc_outputs], dim=0)
                    return vals.mean().item(), vals.std().item()

                r_mu, r_std = mc_mean_std("resilience")
                t_mu, t_std = mc_mean_std("temp")
                f_mu, f_std = mc_mean_std("flood")
                e_mu, e_std = mc_mean_std("energy")
            else:
                out = self.model(X_t)
                r_mu = out["resilience"][0, 0].item()
                t_mu = out["temp"][0, 0].item()
                f_mu = out["flood"][0, 0].item()
                e_mu = out["energy"][0, 0].item()
                r_std = t_std = f_std = e_std = 0.0

        # Back to original scale
        pred_norm = np.array([[r_mu, t_mu, f_mu, e_mu]], dtype=np.float32)
        pred_orig = self.pre.inverse_transform_y(pred_norm)[0]

        # Aleatoric uncertainty from energy head (if available)
        energy_out = self.model(X_t)["energy"]
        if energy_out.shape[1] == 2:
            log_var = energy_out[0, 1].item()
            e_aleatoric_std = float(np.exp(0.5 * log_var))
        else:
            e_aleatoric_std = 0.0

        return {
            "location":             location,
            "year":                 year,
            "interventions":        interventions,
            "building_type":        building_type,
            "predictions": {
                "resilience_score": {
                    "value": round(float(pred_orig[0]), 1),
                    "unit":  "/100",
                    "uncertainty_std": round(r_std, 2),
                },
                "temp_reduction": {
                    "value": round(float(pred_orig[1]), 2),
                    "unit":  "°F",
                    "uncertainty_std": round(t_std, 2),
                },
                "flood_risk_reduction": {
                    "value": round(float(pred_orig[2]), 1),
                    "unit":  "%",
                    "uncertainty_std": round(f_std, 2),
                },
                "energy_savings": {
                    "value": round(float(pred_orig[3]), 0),
                    "unit":  "$/yr",
                    "uncertainty_std": round(e_std + e_aleatoric_std, 0),
                },
            },
            "model_version": "ClimateResilienceNet-v1",
        }

    def batch_predict(self, inputs: List[Dict]) -> List[Dict]:
        """Run inference on a list of input dicts."""
        return [self.predict(**inp) for inp in inputs]


# ---------------------------------------------------------------------------
# Singleton loader (cached for server use)
# ---------------------------------------------------------------------------
_predictor: Optional[ClimatePredictor] = None


def get_predictor(
    model_path: str = "ml/artifacts/best_model.pt",
    pre_path:   str = "ml/artifacts/preprocessor.joblib",
) -> Optional[ClimatePredictor]:
    """Return cached predictor, or None if model files don't exist yet."""
    global _predictor
    if _predictor is None:
        if Path(model_path).exists() and Path(pre_path).exists():
            try:
                _predictor = ClimatePredictor(model_path, pre_path)
            except Exception as e:
                print(f"Warning: Could not load ML model: {e}")
    return _predictor
