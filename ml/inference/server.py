"""
FastAPI inference microservice for ClimateResilienceNet.

The Node.js Express server proxies /api/ml-predict requests to this service.
Run separately: uvicorn ml.inference.server:app --port 8001 --host 0.0.0.0

Endpoints
---------
GET  /health          — model health check + version
POST /predict         — single location prediction
POST /predict/batch   — batch predictions
GET  /model/info      — architecture + parameter count
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from ml.inference.predictor import get_predictor

app = FastAPI(
    title="ClimateResilienceNet Inference API",
    description="ML-powered climate adaptation prediction service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "ml/artifacts/best_model.pt")
PRE_PATH   = os.environ.get("PRE_PATH",   "ml/artifacts/preprocessor.joblib")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class InterventionFlags(BaseModel):
    green_roof:          bool = True
    solar_panels:        bool = True
    flood_walls:         bool = False
    permeable_pavement:  bool = False


class PredictRequest(BaseModel):
    location:       str   = Field(..., example="Miami")
    year:           int   = Field(2050, ge=2025, le=2100)
    interventions:  Optional[InterventionFlags] = None
    building_type:  str   = Field("residential",
                                   example="residential",
                                   description="residential|commercial|industrial|mixed_use|institutional")
    building_age:   float = Field(20.0, ge=0, le=100)
    building_area:  float = Field(150.0, ge=10, le=50000)

    @validator("building_type")
    def valid_building_type(cls, v):
        valid = ["residential", "commercial", "industrial", "mixed_use", "institutional"]
        if v not in valid:
            raise ValueError(f"building_type must be one of {valid}")
        return v


class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    predictor = get_predictor(MODEL_PATH, PRE_PATH)
    return {
        "status":      "ready" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "message":     "Model ready" if predictor else "Model not trained yet — using rule-based fallback",
    }


@app.post("/predict")
def predict(req: PredictRequest):
    predictor = get_predictor(MODEL_PATH, PRE_PATH)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Run: python -m ml.training.train"
        )
    iv = req.interventions.dict() if req.interventions else None
    try:
        result = predictor.predict(
            location      = req.location,
            year          = req.year,
            interventions = iv,
            building_type = req.building_type,
            building_age  = req.building_age,
            building_area = req.building_area,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    predictor = get_predictor(MODEL_PATH, PRE_PATH)
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model not available.")
    results = []
    for r in req.requests:
        iv = r.interventions.dict() if r.interventions else None
        results.append(predictor.predict(
            location=r.location, year=r.year, interventions=iv,
            building_type=r.building_type, building_age=r.building_age,
            building_area=r.building_area,
        ))
    return {"results": results, "count": len(results)}


@app.get("/model/info")
def model_info():
    predictor = get_predictor(MODEL_PATH, PRE_PATH)
    if predictor is None:
        return {"status": "not_loaded"}
    m = predictor.model
    return {
        "architecture":       "ClimateResilienceNet",
        "n_parameters":       m.count_parameters(),
        "n_features":         m.n_features,
        "d_model":            m.d_model,
        "use_physics_prior":  m.use_physics_prior,
        "use_attention":      m.use_attention,
        "use_uncertainty":    m.use_uncertainty,
        "outputs": [
            "resilience_score (0-100)",
            "temp_reduction_f (°F)",
            "flood_risk_reduction (%)",
            "energy_savings_usd ($/yr)",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ml.inference.server:app", host="0.0.0.0", port=8001, reload=False)
