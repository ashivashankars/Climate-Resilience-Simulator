import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";

const API_TIMEOUT       = 5000;
const ML_INFERENCE_URL  = process.env.ML_INFERENCE_URL || "http://localhost:8001";
const ML_TIMEOUT        = 8000;

function fetchWithTimeout(url: string, timeout = API_TIMEOUT, opts?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  return fetch(url, { signal: controller.signal, ...opts }).finally(() => clearTimeout(id));
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {

  // ── Mapbox token ──────────────────────────────────────────────────────────
  app.get("/api/mapbox/token", (_req: Request, res: Response) => {
    const token = process.env.MAPBOX_ACCESS_TOKEN;
    if (!token) return res.status(500).json({ error: "Mapbox token not configured" });
    res.json({ token });
  });

  // ── Mapbox geocoding proxy ────────────────────────────────────────────────
  app.get("/api/mapbox/geocode", async (req: Request, res: Response) => {
    const query = req.query.q as string;
    const token = process.env.MAPBOX_ACCESS_TOKEN;
    if (!query || !token) return res.json({ features: [] });
    try {
      const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json?access_token=${token}&country=US&types=place,locality,neighborhood&limit=5`;
      const response = await fetchWithTimeout(url);
      if (!response.ok) return res.status(response.status).json({ error: "Geocoding service error", features: [] });
      res.json(await response.json());
    } catch (err: any) {
      const isTimeout = err?.name === "AbortError";
      res.status(isTimeout ? 504 : 500).json({
        error: isTimeout ? "Geocoding request timed out" : "Geocoding failed",
        features: [],
      });
    }
  });

  // ── Open-Meteo climate data ───────────────────────────────────────────────
  app.get("/api/climate/data", async (req: Request, res: Response) => {
    const lat = parseFloat(req.query.lat as string);
    const lon = parseFloat(req.query.lon as string);
    if (isNaN(lat) || isNaN(lon)) return res.status(400).json({ error: "Invalid coordinates" });

    const results: { weather?: any; elevation?: any; solar?: any } = {};
    await Promise.all([
      fetchWithTimeout(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m&daily=temperature_2m_max,temperature_2m_min&timezone=auto`)
        .then(r => r.json()).then(d => { results.weather = d; }).catch(() => { results.weather = null; }),
      fetchWithTimeout(`https://api.open-meteo.com/v1/elevation?latitude=${lat}&longitude=${lon}`)
        .then(r => r.json()).then(d => { results.elevation = d; }).catch(() => { results.elevation = null; }),
      fetchWithTimeout(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&daily=shortwave_radiation_sum&timezone=auto`)
        .then(r => r.json()).then(d => { results.solar = d; }).catch(() => { results.solar = null; }),
    ]);
    res.json(results);
  });

  // ── ML Model inference (proxies to FastAPI Python server) ─────────────────
  app.post("/api/ml-predict", async (req: Request, res: Response) => {
    const {
      location     = "San Francisco",
      year         = 2050,
      interventions = { green_roof: true, solar_panels: true, flood_walls: false, permeable_pavement: false },
      building_type = "residential",
      building_age  = 20,
      building_area = 150,
    } = req.body || {};

    try {
      const response = await fetchWithTimeout(
        `${ML_INFERENCE_URL}/predict`,
        ML_TIMEOUT,
        {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ location, year, interventions, building_type, building_age, building_area }),
        },
      );

      if (!response.ok) {
        // ML server unavailable — fall back to rule-based computation
        return res.json(ruleBasedFallback(location, year, interventions));
      }
      res.json(await response.json());
    } catch {
      // FastAPI server not running yet (model not trained) — graceful degradation
      res.json(ruleBasedFallback(location, year, interventions));
    }
  });

  // ── ML model health check ─────────────────────────────────────────────────
  app.get("/api/ml-health", async (_req: Request, res: Response) => {
    try {
      const response = await fetchWithTimeout(`${ML_INFERENCE_URL}/health`, 3000);
      if (response.ok) {
        res.json(await response.json());
      } else {
        res.json({ status: "degraded", model_loaded: false, message: "ML server unhealthy" });
      }
    } catch {
      res.json({ status: "offline", model_loaded: false, message: "ML inference server not running. Train model first: python -m ml.training.train" });
    }
  });

  // ── ML model info ─────────────────────────────────────────────────────────
  app.get("/api/ml-model-info", async (_req: Request, res: Response) => {
    try {
      const response = await fetchWithTimeout(`${ML_INFERENCE_URL}/model/info`, 3000);
      res.json(response.ok ? await response.json() : { status: "not_loaded" });
    } catch {
      res.json({ status: "not_loaded", message: "Train model first" });
    }
  });

  return httpServer;
}

// ---------------------------------------------------------------------------
// Rule-based fallback (mirrors original logic — used before model is trained)
// ---------------------------------------------------------------------------
function ruleBasedFallback(
  location: string,
  year: number,
  interventions: Record<string, boolean>,
): object {
  const zoneMult: Record<string, number> = { tropical: 1.3, subtropical: 1.4, temperate: 1.0, cold: 1.6 };
  const coastalRisk: Record<string, number> = {
    miami: 2.2, "new orleans": 2.0, houston: 1.6, tampa: 1.9, "new york": 1.5,
    boston: 1.4, "san francisco": 1.3, seattle: 1.2, default: 0.8,
  };
  const baseTempMap: Record<number, number> = { 2030: 1.5, 2050: 2.8, 2070: 4.2, 2100: 5.8 };
  const snapYear = [2030, 2050, 2070, 2100].reduce((a, b) =>
    Math.abs(b - year) < Math.abs(a - year) ? b : a
  );
  const loc  = location.toLowerCase();
  const mult = zoneMult[loc.includes("miami") || loc.includes("houston") ? "subtropical" : "temperate"];
  const risk = coastalRisk[loc] ?? coastalRisk.default;
  const temp = (baseTempMap[snapYear] ?? 2.8) * mult;
  const iv   = interventions;
  const n    = Object.values(iv).filter(Boolean).length;
  const score = Math.min(99, 40 + n * 12 + (iv.green_roof ? 6 : 0) + (iv.solar_panels ? 5 : 0));

  return {
    location,
    year: snapYear,
    source: "rule_based_fallback",
    predictions: {
      resilience_score:     { value: score,                  unit: "/100"  },
      temp_reduction:       { value: n > 0 ? 3.2 * n : 0.0, unit: "°F"   },
      flood_risk_reduction: { value: risk > 1.3 ? 35 * (n > 0 ? 1 : 0) : 15, unit: "%" },
      energy_savings:       { value: iv.solar_panels ? 1200 : 300,  unit: "$/yr" },
    },
    note: "ClimateResilienceNet not loaded — using rule-based approximation. Run: python -m ml.training.train",
  };
}
